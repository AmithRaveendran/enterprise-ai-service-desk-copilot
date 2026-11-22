# Performance

Measured numbers, where the time goes, and what to change as the load or the knowledge base
grows. All figures are from a laptop (Windows 11, Docker Desktop) with the sample KB
(19 documents, 108 chunks), gpt-5-mini (`LLM_REASONING_EFFORT=low`) and
text-embedding-3-small, on 22–23 November 2026. Treat them as orders of magnitude.

## Measured

### Search (`GET /search`, k = 5, the API running in Docker, 30 queries per mode)

| Mode | Server time p50 | p95 | What it does |
|---|---|---|---|
| keyword | 9 ms | 11 ms | Postgres full-text (GIN index) |
| vector | 211 ms | 1,846 ms | embed the query with OpenAI, then cosine top 20 |
| hybrid | 191 ms | 294 ms | both, fused with Reciprocal Rank Fusion |

Splitting vector search into its two steps (20 queries):

| Step | p50 | p95 |
|---|---|---|
| OpenAI embedding call | 206 ms | 818 ms |
| pgvector query | 10 ms | 16 ms |

**The database is not the bottleneck: the network call to embed the query is ~95% of search
time**, and its tail (p95) is the provider's, not ours.

### Answers and agent turns (from `servicedesk eval`, see [evals/report.md](../evals/report.md))

| Operation | Mean | p95 | Tokens per call |
|---|---|---|---|
| `/ask` (retrieve + one structured LLM call) | 3.8 s | 5.7 s | ~1,200 (42,897 over 36 model calls) |
| agent turn (`/chat`, one or more model calls + tools) | 7.1 s | 15.9 s | ~4,800 (57,314 over 12 tasks) |

Out-of-scope questions below the similarity threshold are refused **without** a chat-model
call (only the embedding), so they cost ~200 ms and almost nothing.

### Container
- API restart → healthy: **8.8 s**. Most of it is importing the LLM libraries (openai,
  LangChain, LangGraph), not our code.
- Image: 465 MB (python:3.13-slim base + dependencies).
- Code-only rebuild: seconds (dependency layer cached); clean build ~2 minutes.

## Where the time goes in a chat turn

```
 user message
   └─ model call #1: decide tools ................... 1.5–4 s   (LLM)
        └─ tools: status / tickets (Postgres) ......... 5–20 ms
        └─ tool: search_knowledge_base
             ├─ embed question ........................ ~0.2 s   (LLM provider)
             ├─ hybrid search ......................... ~10 ms   (Postgres)
             └─ answer model call ..................... 2–4 s    (LLM)
   └─ model call #2: write the reply ................ 1.5–4 s   (LLM)
   └─ checkpoint writes after each step ............. a few ms each (Postgres)
```

Agent turns are sequential model calls, so latency is roughly the **number of model calls ×
model latency**. That is why the p95 (15.9 s) is about twice the mean: some tasks take two
or three tool rounds.

## What is already in place
- **Indexes:** HNSW on `chunks.embedding` (cosine) and GIN on `chunks.search_vector`. At 108
  rows Postgres rightly chooses a sequential scan + sort (cheaper than the index), so these
  matter only as the KB grows; HNSW keeps vector search in milliseconds up to millions of rows.
- **Refuse early:** the similarity threshold skips the chat model for off-topic questions.
- **Small context:** top 5 chunks of ≤ 500 tokens; only the last 40 messages of a conversation
  are sent to the model (`AGENT_HISTORY_MESSAGES`), and tool rounds are capped.
- **Connection pools:** SQLAlchemy (`pool_pre_ping`) and psycopg for the checkpointer, created
  once per process.
- **Clients built once:** the embedder, the LLM clients and the chat model are cached per
  process.
- **Batching and retries in ingestion:** up to 100 chunks per embedding request, with
  exponential back-off on rate limits; unchanged files are skipped by hash, so re-ingestion
  costs nothing.
- **Bounded waits:** `LLM_TIMEOUT_SECONDS` (60) and a 5 s database connect timeout.

## Scaling: what to do, in order of payoff
1. **Stream responses** (Server-Sent Events): the total time stays the same, but the first
   words appear in ~1 s instead of 4–8 s. This matters most to how fast it feels.
2. **Cache query embeddings** (an LRU cache in process, or Redis shared across replicas, keyed
   by normalised text): repeat questions skip ~200 ms. Caching whole answers is riskier,
   because the KB and service status change.
3. **Fewer model calls per turn:** route plain KB questions straight to `/ask`, use a
   smaller model for tool selection, and lower the reasoning effort.
4. **Horizontal scaling:** the API is stateless (state lives in Postgres), so add replicas.
   Model-provider rate limits (tokens per minute) are the first ceiling, so request a higher
   quota or spread load across deployments.
5. **Async I/O:** endpoints are synchronous and run in FastAPI's thread pool (40 threads by
   default). That is fine while the waits are on the LLM. Thousands of concurrent
   conversations would justify async clients (`AsyncOpenAI`, an async checkpointer).
6. **Bigger KB:** tune HNSW (`m`, `ef_construction`, `hnsw.ef_search`) for recall versus
   speed, and re-run the retrieval eval after each change. Past a few million chunks,
   consider partitioning by category or a dedicated vector store.
7. **Database:** PgBouncer in front of Postgres once replicas × pools approach
   `max_connections`. Add a retention job for old conversation checkpoints.

## How to measure again
```bash
uv run servicedesk eval                  # answer + agent latency and tokens → evals/report.md
curl "http://localhost:8000/search?q=vpn&mode=vector"   # took_ms in the response
docker compose logs api                  # duration_ms on every request line
```
