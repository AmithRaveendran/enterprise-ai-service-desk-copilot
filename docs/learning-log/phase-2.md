# Phase 2: Semantic + hybrid search and the first API

## What was built
```
question
  ├─ vector search:  embed question → ORDER BY embedding <=> :vector  (cosine, HNSW index)
  ├─ keyword search: websearch_to_tsquery → search_vector @@ query      (full-text, GIN index)
  └─ hybrid:         both rankings → Reciprocal Rank Fusion → top k
        ↓
  SearchHit: chunk text + doc_id, title, section, page, source path (everything a citation needs)
```
- `search/retriever.py`: `Retriever.search(query, k, mode, filters)` with modes `vector`,
  `keyword` and `hybrid`, and filters on category, doc_type and service.
- Migration `0002`: an HNSW index on `chunks.embedding`, plus a generated `search_vector`
  (tsvector) column with a GIN index.
- FastAPI app (`api/`): `GET /health`, `GET /search`, OpenAPI docs at `/docs`.
- CLI: `servicedesk search "question" [--mode] [-k] [--category]` and `servicedesk serve`.
- 34 new tests (125 in total): RRF maths, every search mode and filter against the test
  database, and API tests for responses, validation (422), and failures (503).

## Results on hand-written questions (real embeddings, top result)
| Question | Vector | Keyword | Hybrid |
|---|---|---|---|
| I typed my password wrong too many times and now I can't log in | KB-0002 Account lockout | POL-0003 ✗ | KB-0002 Account lockout |
| my headset has no sound in meetings when I plug into the dock | KI-0012 | KI-0012 | KI-0012 |
| how fast will you respond to a critical outage | POL-0002 SLA targets | POL-0002 | POL-0002 |
| can I install Spotify on my work laptop | KB-0008 Company Portal | POL-0006 ✗ | KB-0008 |
| vpn keeps dropping on my macbook | KI-0007 | KI-0007 | KI-0007 |
| who do I ask for access to the finance folder | KB-0011 | KB-0011 | KB-0011 |

Keyword search alone missed 2 of 6: the questions share few exact words with the right
article. Vector search got all 6, and hybrid kept all 6 while also surfacing chunks that
contain the exact terms. Latency: about 10 ms in Postgres, plus about 200 ms to embed the
question (the first call takes longer while the HTTPS connection is set up).

## Key concepts

**Cosine distance in SQL.** pgvector adds operators: `<=>` cosine distance, `<->` L2
distance, `<#>` negative inner product. Similarity = `1 - (embedding <=> :query)`. OpenAI
vectors have length 1, so cosine and inner product give the same ranking.

**HNSW (Hierarchical Navigable Small World).** An approximate nearest-neighbour index: a
layered graph where each vector links to its near neighbours. A query walks the graph
instead of comparing against every row, so it reads hundreds of vectors instead of millions.
It is *approximate*: it can occasionally miss a true neighbour (tunable with `m`,
`ef_construction` and `hnsw.ef_search`). The index must use the same operator class as the
query (`vector_cosine_ops` ↔ `<=>`), or Postgres won't use it.

**The planner chooses.** With 108 rows, `EXPLAIN` shows a sequential scan plus sort: exact
and cheaper than walking a graph. With `enable_seqscan = off` it uses
`ix_chunks_embedding_hnsw`. The index pays off at tens of thousands of rows; it's in place
now so search stays fast as the knowledge base grows.

**Filtered vector search.** HNSW finds the nearest vectors first and filters after, so a
narrow filter can return fewer than k rows. pgvector 0.8's `hnsw.iterative_scan` keeps
searching until enough rows pass; we turn it on (`SET LOCAL`) only when filters are used.

**Postgres full-text search.** `to_tsvector('english', text)` lower-cases, removes stop words
("the", "my") and stems ("printers" → `printer`). A generated column stores this, so
ingestion code didn't change. Section words get weight A, body words weight B, so a match
in a heading ranks higher. `websearch_to_tsquery` parses user text safely; we turn its AND
into OR so a question needn't contain every word, and `ts_rank` rewards chunks matching more.

**Why hybrid?** Vector search understands meaning but can blur exact tokens (error codes,
product names, IDs). Keyword search is exact but misses paraphrases. Combining them is the
standard production approach.

**Reciprocal Rank Fusion.** `score(chunk) = Σ 1 / (60 + rank in each list)`. It uses only
positions, so cosine similarity (0..1) and ts_rank (unbounded) never need to be normalised
against each other. A chunk ranked 3rd by both beats one ranked 1st by only one.

**FastAPI basics used.**
- *Path operations*: `@router.get("/search")`; query parameters come from function arguments.
- *Validation*: `Query(min_length=2, le=20)`, `Literal[...]` types → automatic 422 with the
  exact field and reason. This is the "structured input validation" on the resume.
- *Response models*: Pydantic models define and document the JSON shape (`/docs`).
- *Dependency injection*: `Depends(get_retriever)`; tests replace it with
  `app.dependency_overrides` to use the test DB and fake embedder.
- *Sync endpoints* (`def`, not `async def`) run in a thread pool, which suits the
  synchronous SQLAlchemy and OpenAI clients.

**Status codes.** 422 = the caller's input is invalid. 503 = we can't serve right now
(database down, or no API key for vector search). 502 = an upstream dependency (the
embedding API) failed. `/health` returns 503 when the database is unreachable, so a load
balancer stops routing traffic to that instance.

## Design decisions
- **Retriever is independent of FastAPI.** The CLI and API use it now; the LangGraph agent
  (Phase 5) and MCP server (Phase 7) will reuse it.
- **Keyword search works without an API key.** The API starts without OpenAI configured, and
  only vector/hybrid return 503.
- **Each method contributes 20 candidates to fusion**, more than k, so a chunk ranked 8th by
  one method and 2nd by the other can still surface.
- **Results carry citation fields** (doc_id, title, section, page, path) ready for Phase 3.

## Interview talking points
- "How does your retrieval work?" Hybrid: pgvector cosine search plus Postgres full-text,
  fused with RRF, with metadata filters.
- "Why not only vectors?" Exact identifiers and rare terms: keyword search finds "KI-0007"
  literally. "Why not only keywords?" It missed 2 of 6 paraphrased questions; vectors got 6.
- "What is HNSW and when does it help?" Approximate graph index; the planner skips it on tiny
  tables; recall vs speed trade-off.
- "How do you validate API input?" Pydantic/FastAPI declarations → 422 with details, tested.

## Known limitations (to revisit)
- Every vector/hybrid search costs one embedding call (~200 ms). A cache of recent query
  embeddings would help repeated questions.
- RRF scores aren't a confidence measure. Phase 3 will use `similarity` to decide when the
  evidence is too weak to answer.
- No reranking model yet (a cross-encoder could reorder the top 20 more precisely).
- English-only text search configuration.

## Commands
```bash
uv run alembic upgrade head                                   # adds the search indexes
uv run servicedesk search "vpn keeps dropping on my macbook"  # hybrid, top 5
uv run servicedesk search "secure print" --mode keyword -k 3  # no API key needed
uv run servicedesk search "password" --category policy
uv run servicedesk serve --reload                             # API at http://127.0.0.1:8000/docs
curl "http://127.0.0.1:8000/search?q=vpn+keeps+dropping&k=3"
curl http://127.0.0.1:8000/health
```
