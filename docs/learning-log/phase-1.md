# Phase 1: Document ingestion

## What was built
```
data/kb/*.md|html|pdf
  → loaders.py     parse each format, strip boilerplate, keep metadata + heading paths
  → chunker.py     token-bounded chunks (max 500, overlap 50) + contextual header
  → embeddings.py  OpenAI / Azure OpenAI, batched, retried with backoff
  → pipeline.py    idempotent sync into Postgres (documents + chunks with vector(1536))
```
- `servicedesk ingest [PATH] [--dry-run] [--force] [--no-prune]`: the ingestion CLI.
- `servicedesk chunks FILE`: shows exactly how one file is chunked (no API, no DB).
- Alembic migration `0001` creates the `documents` and `chunks` tables.
- 91 tests: loaders, chunker, embedder (with a fake OpenAI client), and integration tests
  against a separate `servicedesk_test` database.

Result on the sample KB: 19 documents → 108 chunks → 11,384 tokens embedded (about $0.0002).
A second run embeds 0 tokens.

## Key concepts

**Embeddings.** A model turns text into a vector (1536 numbers here). Similar meanings give
vectors pointing in similar directions, measured by *cosine similarity*. That is how "I typed
the wrong thing too many times" finds the *Account lockout* section with no shared keywords.

**Why chunk?** One vector per whole document blurs many topics together. Small, focused
chunks give sharper matches, and later the LLM only receives the relevant passages, which
is cheaper and leaves less room for distraction.

**Tokens.** Models measure text in tokens (about ¾ of a word). We count with `tiktoken`
using `cl100k_base`, the tokenizer of the text-embedding-3 models, so chunk limits are exact.

**Chunking strategy used:**
1. Never cross a section boundary, so each chunk has one clear heading for citations.
2. Split at the largest natural unit that fits: paragraph → line → raw tokens.
3. Overlap consecutive chunks of a long section by up to 50 tokens, so an idea on a
   boundary is retrievable from either side.
4. **Contextual header**: `Document: <title> (<id>)\nSection: <heading path>` is prepended to
   the text we embed. A chunk saying "Click Connect" is otherwise ambiguous; with the header
   it is clearly about the VPN.

**Cleaning per format:**
- Markdown: front matter → metadata; tables → one "Column: value; ..." line per row, so a
  row is still understandable without its header row.
- HTML: remove `nav`, `footer`, `script`…; read `<meta>` tags; number `<ol>` items.
- PDF: remove lines repeated on every page (after replacing digits, so "Page 1/2" and
  "Page 2/2" match); re-join lines the layout wrapped; detect "N. Heading" sections; keep
  page numbers for citations.

**Idempotency.** Identity is the business `doc_id` (KB-0001), not the file path.
| Case | Action |
|---|---|
| New doc_id | embed + insert |
| Same SHA-256 and same embedding model | skip (free) |
| Content or model changed | re-embed, replace chunks in one transaction |
| Renamed/moved, same content | update path only |
| File deleted | delete document (chunks cascade) |

**Transactions.** Embedding happens *before* the write transaction opens. If the API fails,
the old version stays. Deleting old chunks and inserting new ones commit together, so a
reader never sees half a document.

**Retries.** Only transient errors (rate limit, connection, timeout, 5xx) are retried, with
exponential backoff plus jitter. An auth error fails immediately. The SDK's own retries
are turned off (`max_retries=0`) so the two retry layers don't multiply.

**Migrations.** Alembic versions the schema in code (`migrations/versions/0001_*.py`).
`alembic upgrade head` brings any database up to date, including the test database.

## Design decisions
- **pgvector in the same Postgres** as everything else: one system, and document + chunk
  updates are transactional together.
- **No vector index yet.** With 108 rows an exact scan is instant. HNSW comes in Phase 2,
  where its trade-offs can be measured.
- **Failure isolation.** One broken file is reported and the rest still ingest. Pruning is
  skipped in a run with failures, so a temporarily unreadable file is never deleted.
- **Dimension check up front.** A 3072-d model against a `vector(1536)` column fails before
  any API spend.
- **Fake embedder in tests.** Feature hashing gives deterministic vectors with no network or
  cost, while integration tests still exercise real Postgres.

## Interview talking points
- "How did you chunk, and why?" Section-aware, token-bounded, with overlap and contextual
  headers, and why each choice matters for retrieval quality and citations.
- "How do you avoid re-embedding everything?" Content hash + model version per document.
- "What happens if the embedding API fails mid-run?" Per-document isolation, embedding before
  the transaction, all-or-nothing chunk replacement, and retries for transient errors only.
- "How did you handle PDFs?" Repeated-line detection for headers/footers, de-wrapping, and
  page tracking for citations.

## Known limitations (to revisit)
- Overlap works at paragraph/line granularity, so a paragraph bigger than 50 tokens is not
  partially overlapped.
- PDF heading detection is a pattern match for our policy format, not general layout analysis.
- Single-page PDFs can't use repeated-line detection.
- Changing chunk settings doesn't trigger re-ingestion automatically; use `--force`.

## Commands
```bash
uv run alembic upgrade head            # create/upgrade tables
uv run servicedesk ingest --dry-run    # free preview: docs, chunks, tokens
uv run servicedesk ingest              # embed + store (idempotent)
uv run servicedesk chunks data/kb/how-to/vpn-setup.md
uv run pytest                          # integration tests need `docker compose up -d`
```
