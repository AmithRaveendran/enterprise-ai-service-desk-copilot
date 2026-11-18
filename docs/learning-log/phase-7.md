# Phase 7: MCP server

## What was built
```
 Claude Desktop · Claude Code · VS Code · MCP Inspector        (any MCP client)
        │  stdio: the client starts `servicedesk mcp`      http: http://127.0.0.1:8001/mcp
        ▼
 mcp_server/server.py   build_mcp_server(desk, ask_knowledge_base)
   tools      get_service_status · search_tickets · get_ticket · search_knowledge_base
              create_ticket · update_ticket · add_comment  (preview → confirm)
   resources  kb://articles (the list) · kb://articles/{doc_id} (one article as Markdown)
   prompt     report_it_issue(problem, requester_email)
        │
 agent/tools.py         AgentTools.run() / preview()  ← the same code our own agent uses
        │
 desk/tools.py → desk/service.py → Postgres            rag/answerer.py → pgvector + OpenAI
```
- `mcp_server/server.py`: an `MCPServer` (the official `mcp` SDK, v2; FastMCP was renamed
  MCPServer in v2) with **7 tools, 2 resources and 1 prompt**. Built by a factory,
  `build_mcp_server()`, so tests pass in a test database and a fake knowledge base.
- **Nothing is redefined.** Each tool calls the Phase 4/5 code through `AgentTools`: same
  input models, same "errors are data" results (`{"ok": false, "error": {...}}`, sent with
  `isError: true`), same handling of database outages (`unavailable`). Tool descriptions come
  from the registry (`desk/tools.py`).
- **Writes need a confirmation token** (`mcp_server/confirmation.py`). The first call returns
  a preview and a token and changes nothing; the client's model shows the person the change;
  the second call, with the same arguments and the token, writes. The token is an HMAC
  signature over the tool name and the validated arguments, so it can't be invented and
  doesn't fit any other change. Its random part is the write's idempotency key (`mcp:<nonce>`),
  so a repeated confirmed call writes once.
- **Tool annotations** tell the client what each tool does: reads are `readOnlyHint`, writes
  are not read-only, not destructive, idempotent; none reach outside our systems
  (`openWorldHint: false`). Clients use these to decide when to ask the person first.
- **Resources:** `kb://articles` lists the 19 articles with their URIs;
  `kb://articles/KB-0001` returns one as Markdown (re-read with the Phase 1 loaders, so PDFs
  come back as text with headings). `search_knowledge_base` sources include the `uri`, so a
  client can open the whole article behind a citation.
- **Prompt:** `report_it_issue` gives a person a starting point that follows the desk's
  process (status → knowledge base → existing tickets → preview a ticket).
- **Transports:** `servicedesk mcp` (stdio, started by the client) and
  `servicedesk mcp --transport http` (Streamable HTTP on 127.0.0.1:8001/mcp).
  `.mcp.json` registers the stdio server for Claude Code.
- 23 new tests (344 in total): 22 through the SDK's real client connected in-process, and
  one that the `mcp` command never prints to stdout, even when it can't start.

## How a write runs over MCP
"Raise INC-1002 to P2" from an MCP client:
1. The client's model calls `update_ticket {ticket_id: "INC-1002", priority: "P2"}`.
2. The SDK validates the arguments against the published schema (built from our `TicketId`,
   `Priority` types). Invalid → `isError: true` with the problems, nothing else happens.
3. `write()`: `AgentTools.preview()` validates with `UpdateTicketInput` and normalises
   (`{ticket_id: 1002, priority: "P2"}`). No token → result `confirmation_required` with the
   preview and a token. **Nothing written.**
4. The model shows the person the change. They agree.
5. Same call plus `confirmation_token`. `Confirmations.check()` re-signs the normalised
   arguments: match → the nonce. Different arguments, a made-up token or one from before a
   restart → `invalid_confirmation`, with the new preview and a fresh token.
6. `AgentTools.run(..., idempotency_key="mcp:<nonce>")` → `ServiceDesk.update_ticket` in one
   transaction with the key. Sent again (a client retry) → the saved result, no second write.

## Key concepts
- **MCP**: a standard protocol (JSON-RPC messages) between an AI app (the *client*/host)
  and a *server* that offers context. Write the server once; every MCP client can use it.
- **Three primitives:** *tools* (the model decides to call them: actions), *resources*
  (the app or person reads them by URI: data), *prompts* (the person picks one: templates).
- **Transports:** *stdio*: the client starts the server as a child process and talks over
  stdin/stdout (local, no network, one client). *Streamable HTTP*: the server listens on a
  URL (remote or shared clients). Same server code for both.
- **stdout is the protocol** under stdio: one stray `print()` corrupts it. All our messages
  go to stderr (`err_console` in cli.py; the SDK logs to stderr too).
- **Schemas from type hints:** the SDK builds each tool's JSON Schema from the function's
  annotations. Reusing `Title`, `Email`, `TicketId`… puts our limits (`minLength`, enums,
  examples) in front of the client's model.
- **HMAC** (hash-based message authentication code): a signature only the secret holder can
  make. Stateless: the server stores no tokens, it re-computes the signature to check one.
  `hmac.compare_digest` compares in constant time.
- **DNS rebinding protection:** a web page could make a browser send requests to
  `127.0.0.1`. The SDK checks `Host`/`Origin` on localhost and answers 421 otherwise.

## Design decisions
- **Reuse `AgentTools`, not the HTTP API.** The MCP server calls the same Python code as the
  agent: one validation path, one error format, no network hop. It's just another front
  door, like FastAPI and the CLI.
- **Explicit tool signatures + a drift test**, instead of generating functions from the
  input models. The SDK needs a function signature per tool; writing them out keeps them
  readable, and `test_tool_parameters_match_the_input_models` fails if a field is added to a
  model but not to the MCP tool. The input model still validates every call, so rules across
  fields (resolving needs a resolution) are enforced even though a schema can't express them.
- **A confirmation token instead of a plain `confirm: true` flag.** A flag can be sent on the
  first call without anyone seeing the change; a token can only come from a preview of exactly
  these arguments, and gives each confirmed write its idempotency key for free.
- **Why not MCP elicitation** (the server asking the person directly)? Tried: on the current
  protocol version (2026-07-28) the server can't send its own requests to the client
  mid-call (`NoBackChannelError`); that moved to a multi-round "input required" flow that
  clients support unevenly. The token flow works with every client, and the client's own
  "allow this tool?" prompt (driven by the annotations) is the person's gate.
- **Sync tool functions.** Our service layer is synchronous SQLAlchemy; the SDK runs sync
  tools in a worker thread, so the event loop isn't blocked. No need for async here.
- **Articles re-read from their source files**, not stitched together from chunks (which
  overlap). The path comes from our database, never from the client.

## Results
Run by hand against the dev database:
- stdio (a real subprocess, as a desktop client starts it): 7 tools listed;
  `search_knowledge_base("How do I reset my password?")` answered from KB-0002 with 4 cited
  sources, each with its `kb://articles/KB-0002` URI.
- HTTP: same tools, a status check and an article read; a request with a foreign `Host`
  header → 421.
- Preview → confirm on INC-1002; a token with different arguments and a forged token were
  both rejected; invalid arguments came back with one error per field.

## Interview talking points
- "The service desk is one Python core with four front doors: CLI, REST API, a LangGraph
  agent and an MCP server. MCP made it usable from Claude Desktop or VS Code without writing
  any client code; each tool is the same function the agent calls."
- "Writes over MCP are two calls: preview, then confirm with an HMAC token bound to the exact
  arguments. The model can't confirm a change the person hasn't been shown, and the token's
  nonce doubles as the idempotency key, so client retries never duplicate a ticket."
- "I tried elicitation first and found it doesn't work over the newest protocol version
  without the multi-round input flow, so I chose a mechanism every client supports. I'd
  revisit it once clients support it widely."
- "Annotations matter: readOnly/idempotent/openWorld hints let the client decide what to ask
  the person about; the server still enforces its own rules."
- "stdio servers must never print to stdout: it's the protocol channel."

## Known limitations (to revisit)
- **No authentication or identity.** Anyone who can start the server (stdio) or reach the
  port (HTTP, localhost only by default) can use every tool, and `requester`/`author` are
  whatever the caller sends. Real use needs MCP's OAuth support and the requester taken from
  the signed-in user (Phase 9 security doc).
- The confirmation is a protocol step, not proof a human agreed: a model could preview and
  confirm by itself. The human gate is the client's tool-approval prompt.
- `update_ticket` can't clear the assignee over MCP (null means "not given").
- Tokens are signed with a per-process secret: a restart invalidates open previews (the
  client just previews again); several HTTP workers would need a shared secret.
- Slow cold start (~10 s, mostly importing `openai`/LangChain on this machine); clients wait.
- MCP clients start the server with a clean environment: the OpenAI key must be in `.env`.

## Commands
```bash
uv run servicedesk mcp                                # stdio: for a client to start
uv run servicedesk mcp --transport http --port 8001   # http://127.0.0.1:8001/mcp
uv run pytest tests/test_mcp.py -v
```
