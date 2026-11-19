"""Command-line interface: `uv run servicedesk --help`."""

import json
import logging
from collections.abc import Callable
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import openai
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError

from servicedesk.agent.checkpointer import postgres_checkpointer
from servicedesk.agent.copilot import ChatError, ChatTurn, Copilot, build_copilot, new_thread_id
from servicedesk.agent.llm import get_chat_model
from servicedesk.config import get_settings
from servicedesk.db.session import get_engine, get_sessionmaker
from servicedesk.desk.schemas import GetServiceStatusInput, ServiceHealth
from servicedesk.desk.seed import seed as seed_database
from servicedesk.desk.service import DeskError, ServiceDesk
from servicedesk.desk.tools import TOOLS, run_tool
from servicedesk.ingestion.chunker import Chunker
from servicedesk.ingestion.embeddings import ConfigurationError, get_embedder
from servicedesk.ingestion.loaders import load_document
from servicedesk.ingestion.pipeline import IngestionPipeline, preview
from servicedesk.rag.answerer import Answerer
from servicedesk.rag.llm import get_llm
from servicedesk.search.retriever import Retriever, SearchFilters, SearchMode

app = typer.Typer(help="Enterprise AI Service Desk Copilot tools.", no_args_is_help=True)
console = Console()
err_console = Console(stderr=True)  # for `mcp`, whose stdout is the protocol


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debug logs.")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _chunker() -> Chunker:
    settings = get_settings()
    return Chunker(settings.chunk_max_tokens, settings.chunk_overlap_tokens)


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(exists=True, help="File or folder to ingest.")] = Path(
        "data/kb"
    ),
    force: Annotated[
        bool, typer.Option("--force", help="Re-embed every document, even unchanged ones.")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Parse and chunk only: no OpenAI calls, no DB writes."),
    ] = False,
    prune: Annotated[
        bool, typer.Option("--prune/--no-prune", help="Remove documents whose files are gone.")
    ] = True,
) -> None:
    """Load, chunk, embed and store knowledge-base documents."""
    chunker = _chunker()
    if dry_run:
        _print_preview(path, chunker)
        return

    _check_database()
    try:
        embedder = get_embedder()
    except ConfigurationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    pipeline = IngestionPipeline(get_sessionmaker(), embedder, chunker)
    with console.status(f"Ingesting {path} ..."):
        report = pipeline.run(path, force=force, prune=prune)

    console.print(
        f"[green]added {len(report.added)}[/green]  "
        f"[yellow]updated {len(report.updated)}[/yellow]  "
        f"unchanged {len(report.unchanged)}  "
        f"[magenta]removed {len(report.removed)}[/magenta]  "
        f"[red]failed {len(report.failed)}[/red]"
    )
    for label, ids in [("Added", report.added), ("Updated", report.updated)]:
        if ids:
            console.print(f"{label}: {', '.join(ids)}")
    if report.removed:
        console.print(f"Removed: {', '.join(report.removed)}")
    console.print(
        f"Chunks written: {report.chunks_written}  Tokens embedded: {report.tokens_embedded}"
    )
    for source, error in report.failed:
        console.print(f"[red]FAILED[/red] {source}: {error}")
    if report.pruning_skipped:
        console.print("[yellow]Pruning skipped because some files failed.[/yellow]")
    if report.failed:
        raise typer.Exit(1)


@app.command()
def chunks(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="One document.")],
) -> None:
    """Show how a single file is split into chunks (no OpenAI calls, no database)."""
    doc = load_document(path)
    console.print(f"[bold]{doc.doc_id}[/bold] {doc.title}  ({doc.doc_type})\n")
    for chunk in _chunker().split(doc):
        page = f", page {chunk.page}" if chunk.page else ""
        console.rule(f"chunk {chunk.index} | {chunk.token_count} tokens{page}")
        console.print(chunk.embedding_text, markup=False, highlight=False)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to look for, in plain words.")],
    k: Annotated[int, typer.Option("--k", "-k", min=1, max=20, help="Number of results.")] = 5,
    mode: Annotated[SearchMode, typer.Option(help="vector, keyword or hybrid.")] = "hybrid",
    category: Annotated[str | None, typer.Option(help="e.g. how-to, policy.")] = None,
    doc_type: Annotated[str | None, typer.Option(help="markdown, html or pdf.")] = None,
    service: Annotated[str | None, typer.Option(help="e.g. VPN, Email.")] = None,
) -> None:
    """Search the knowledge base and show the best-matching chunks."""
    _check_database()
    try:
        embedder = None if mode == "keyword" else get_embedder()
    except ConfigurationError as exc:
        console.print(f"[red]{exc}[/red] Tip: --mode keyword works without an API key.")
        raise typer.Exit(1) from exc

    filters = SearchFilters(category=category, doc_type=doc_type, service=service)
    hits = Retriever(get_sessionmaker(), embedder).search(query, k=k, mode=mode, filters=filters)
    if not hits:
        console.print("No results.")
        return

    table = Table(title=f"{mode} search: {query}")
    for column in ["#", "score", "vec", "kw", "doc_id", "section", "content"]:
        table.add_column(column)
    for rank, hit in enumerate(hits, start=1):
        table.add_row(
            str(rank),
            f"{hit.score:.3f}",
            str(hit.vector_rank or "-"),  # where vector search ranked it (- = not in its list)
            str(hit.keyword_rank or "-"),
            hit.doc_id,
            hit.section.split(" > ")[-1],
            hit.content[:80].replace("\n", " ") + "...",
        )
    console.print(table)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your IT question, in plain words.")],
    category: Annotated[str | None, typer.Option(help="e.g. how-to, policy.")] = None,
    service: Annotated[str | None, typer.Option(help="e.g. VPN, Email.")] = None,
) -> None:
    """Answer a question from the knowledge base, with citations (or refuse)."""
    _check_database()
    settings = get_settings()
    try:
        answerer = Answerer(
            Retriever(get_sessionmaker(), get_embedder()),
            get_llm(),
            top_k=settings.rag_top_k,
            min_similarity=settings.rag_min_similarity,
        )
    except ConfigurationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    with console.status("Thinking ..."):
        result = answerer.ask(question, filters=SearchFilters(category=category, service=service))

    if result.refused:
        console.print(f"[yellow]{result.answer}[/yellow]")
        console.print(f"[dim]reason: {result.refusal_reason}[/dim]")
    else:
        console.print(result.answer, markup=False, highlight=False)
        table = Table(title="Sources")
        for column in ["#", "doc_id", "title", "section"]:
            table.add_column(column)
        for c in result.citations:
            page = f" (p. {c.page})" if c.page else ""
            table.add_row(str(c.number), c.doc_id, c.title, c.section + page)
        console.print(table)
    similarity = f"{result.top_similarity:.2f}" if result.top_similarity is not None else "-"
    console.print(
        f"[dim]best similarity {similarity} | model {result.model or '-'} | "
        f"tokens in {result.input_tokens} out {result.output_tokens}[/dim]"
    )


@app.command()
def seed() -> None:
    """Load the service catalogue and sample tickets (data/seed/). Safe to re-run."""
    _check_database("services")
    report = seed_database(get_sessionmaker())
    console.print(f"services: {report.services}  sample tickets added: {report.tickets_added}")


@app.command()
def status(
    service: Annotated[str | None, typer.Argument(help="One service, e.g. VPN.")] = None,
) -> None:
    """Show the health of IT services."""
    _check_database("services")
    try:
        result = ServiceDesk(get_sessionmaker()).get_service_status(
            GetServiceStatusInput(service=service)
        )
    except DeskError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    colours = {"operational": "green", "degraded": "yellow", "maintenance": "blue"}
    table = Table(title="Service status")
    for column in ["service", "status", "message"]:
        table.add_column(column)
    for s in result.services:
        colour = colours.get(s.status, "red")
        table.add_row(s.name, f"[{colour}]{s.status}[/{colour}]", s.message or "")
    console.print(table)


@app.command("set-status")
def set_status(
    service: Annotated[str, typer.Argument(help="e.g. VPN")],
    health: Annotated[ServiceHealth, typer.Argument(help="New status.")],
    message: Annotated[str | None, typer.Option(help="What users should know.")] = None,
) -> None:
    """Admin: change a service's status (not available to the assistant)."""
    _check_database("services")
    try:
        result = ServiceDesk(get_sessionmaker()).set_service_status(service, health, message)
    except DeskError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"{result.name}: {result.status}")


@app.command()
def tools() -> None:
    """List the service desk tools the assistant can use."""
    table = Table(title="Tools")
    for column in ["name", "writes", "arguments", "description"]:
        table.add_column(column)
    for t in TOOLS:
        args = ", ".join(t.input_schema()["properties"])
        table.add_row(t.name, "yes" if t.writes else "", args, t.description)
    console.print(table)


@app.command()
def tool(
    name: Annotated[str, typer.Argument(help="Tool name (see `servicedesk tools`).")],
    arguments: Annotated[str, typer.Argument(help="Arguments as JSON.")] = "{}",
) -> None:
    """Run one tool exactly as the assistant would, and print the JSON result."""
    _check_database("tickets")
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Arguments must be JSON: {exc}[/red]")
        raise typer.Exit(1) from exc
    result = run_tool(ServiceDesk(get_sessionmaker()), name, parsed)
    console.print_json(data=result)
    if not result["ok"]:
        raise typer.Exit(1)


@app.command()
def chat(
    message: Annotated[
        str | None, typer.Option("--message", "-m", help="Send one message and exit.")
    ] = None,
    thread: Annotated[
        str | None, typer.Option("--thread", "-t", help="Continue an earlier conversation.")
    ] = None,
) -> None:
    """Talk to the assistant (the LangGraph agent). Conversations are saved in Postgres:
    pass --thread <id> to pick one up again, even after a restart.

    Before the assistant changes anything (creates or updates a ticket, adds a comment) it
    shows you the change and asks you to approve or reject it."""
    _check_database("tickets")
    settings = get_settings()
    try:
        model = get_chat_model(settings)
        answerer = Answerer(
            Retriever(get_sessionmaker(), get_embedder()),
            get_llm(),
            top_k=settings.rag_top_k,
            min_similarity=settings.rag_min_similarity,
        )
    except ConfigurationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    with postgres_checkpointer(settings.database_url) as checkpointer:
        copilot = build_copilot(
            model, ServiceDesk(get_sessionmaker()), answerer, checkpointer, settings
        )
        thread = thread or new_thread_id()
        if message is not None:
            _settle(copilot, thread)
            _chat_turn(copilot, thread, lambda: copilot.chat(message, thread))
            return
        console.print("[dim]Type your message. Empty line or 'exit' to quit.[/dim]")
        while True:
            _settle(copilot, thread)
            text = console.input("[bold cyan]you>[/bold cyan] ").strip()
            if text.lower() in {"", "exit", "quit"}:
                break
            _chat_turn(copilot, thread, partial(copilot.chat, text, thread))
        if copilot.status(thread) != "new":
            console.print(f"[dim]Continue later with: servicedesk chat --thread {thread}[/dim]")


def _settle(copilot: Copilot, thread: str) -> None:
    """A conversation left waiting for approval or half-finished (e.g. the model API failed
    or the process was stopped) must be dealt with before the next message."""
    status = copilot.status(thread)
    if status == "awaiting_approval":
        console.print("[yellow]This conversation is waiting for your approval.[/yellow]")
        _chat_turn(copilot, thread, lambda: _ask_approval(copilot, thread))
    elif status == "interrupted":
        if typer.confirm("The last turn didn't finish. Resume it now?", default=True):
            _chat_turn(copilot, thread, lambda: copilot.resume(thread))


def _chat_turn(copilot: Copilot, thread: str, run: Callable[[], ChatTurn]) -> None:
    try:
        with console.status("Thinking ..."):
            turn = run()
        _show_turn(turn)
        while turn.status == "awaiting_approval":
            turn = _ask_approval(copilot, thread)
            _show_turn(turn)
    except openai.APIError as exc:
        console.print(
            f"[red]Language model error ({type(exc).__name__}). The conversation was saved up "
            f"to the failure: run `servicedesk chat -t {thread}` to resume it.[/red]"
        )
    except ChatError as exc:
        console.print(f"[red]{exc}[/red]")


def _ask_approval(copilot: Copilot, thread: str) -> ChatTurn:
    actions = copilot.pending_actions(thread)
    for action in actions:
        _print_action(action)
    question = "Approve this change?" if len(actions) == 1 else "Approve these changes?"
    if typer.confirm(question, default=False):
        with console.status("Working ..."):
            return copilot.decide(thread, approved=True)
    reason = console.input("[dim]Reason, or what to do instead (Enter to skip):[/dim] ").strip()
    with console.status("Thinking ..."):
        return copilot.decide(thread, approved=False, reason=reason or None)


def _print_action(action: dict[str, Any]) -> None:
    table = Table(title=f"Pending change: {action['tool']}", title_justify="left")
    table.add_column("field", style="dim")
    table.add_column("value")
    for name, value in action["arguments"].items():
        shown = value if isinstance(value, str) else json.dumps(value)
        table.add_row(name, escape(shown))
    console.print(table)


def _show_turn(turn: ChatTurn) -> None:
    for use in turn.tools_used:
        mark = "[green]ok[/green]" if use.ok else f"[red]{use.error_code}[/red]"
        console.print(f"[dim]  tool {use.name}: [/dim]{mark}")
    console.print("[bold green]assistant>[/bold green] ", end="")
    console.print(turn.reply, markup=False, highlight=False)
    for c in turn.citations:
        source = escape(f"[{c['number']}] {c['doc_id']} {c['title']} > {c['section']}")
        console.print(f"[dim]  {source}[/dim]")
    console.print(
        f"[dim]  thread {turn.thread_id} | tokens in {turn.input_tokens} "
        f"out {turn.output_tokens}[/dim]"
    )


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to listen on.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Restart when code changes (dev).")] = False,
) -> None:
    """Run the HTTP API (docs at http://127.0.0.1:8000/docs)."""
    import uvicorn  # imported here so other commands don't pay for loading the web stack

    uvicorn.run("servicedesk.api.app:app", host=host, port=port, reload=reload)


class McpTransport(StrEnum):
    stdio = "stdio"
    http = "http"


@app.command()
def mcp(
    transport: Annotated[
        McpTransport,
        typer.Option(help="stdio: started by an MCP client. http: a URL to connect to."),
    ] = McpTransport.stdio,
    host: Annotated[str, typer.Option(help="Interface to listen on (http only).")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on (http only).")] = 8001,
) -> None:
    """Run the MCP server, so MCP clients (Claude Desktop, VS Code, MCP Inspector) can use
    the service desk tools and read the knowledge base.

    stdio (default): the client starts this command itself and talks to it over
    stdin/stdout. http: open http://127.0.0.1:8001/mcp in the client."""
    # With stdio, stdout belongs to the protocol: anything else printed there would corrupt
    # it. So every message from this command goes to stderr, which clients show as logs.
    _check_database("tickets", out=err_console)
    from servicedesk.mcp_server.server import build_mcp_server  # loads the MCP SDK only here

    settings = get_settings()
    try:
        answerer = Answerer(
            Retriever(get_sessionmaker(), get_embedder()),
            get_llm(),
            top_k=settings.rag_top_k,
            min_similarity=settings.rag_min_similarity,
        )
        ask_knowledge_base = answerer.ask
    except ConfigurationError as exc:
        err_console.print(
            f"[yellow]{exc} search_knowledge_base will answer 'unavailable'.[/yellow]"
        )
        ask_knowledge_base = None

    server = build_mcp_server(ServiceDesk(get_sessionmaker()), ask_knowledge_base)
    if transport == McpTransport.stdio:
        server.run("stdio")
    else:
        err_console.print(f"MCP server at http://{host}:{port}/mcp (Ctrl+C to stop)")
        server.run("streamable-http", host=host, port=port)


class EvalSuite(StrEnum):
    all = "all"
    retrieval = "retrieval"
    answers = "answers"
    agent = "agent"


@app.command("eval")
def evaluate(
    suite: Annotated[EvalSuite, typer.Option(help="Which suite to run.")] = EvalSuite.all,
    limit: Annotated[
        int | None, typer.Option(help="Only the first N questions and N agent tasks (quick check).")
    ] = None,
    judge: Annotated[bool, typer.Option(help="Grade answers with the LLM judge.")] = True,
    judge_model: Annotated[
        str | None, typer.Option(help="Model for the judge (default: the chat model).")
    ] = None,
    workers: Annotated[int, typer.Option(help="Cases run at the same time.")] = 4,
    dataset: Annotated[Path, typer.Option(help="The dataset (JSON Lines).")] = Path(
        "evals/dataset.jsonl"
    ),
    results_path: Annotated[Path, typer.Option("--results", help="Per-case results.")] = Path(
        "evals/results.json"
    ),
    report_path: Annotated[Path, typer.Option("--report", help="The Markdown report.")] = Path(
        "evals/report.md"
    ),
) -> None:
    """Evaluate retrieval (hit@k, MRR), answers (refusals, citations, grounding and
    correctness by an LLM judge) and agent task completion; write evals/report.md.

    Uses the real models (costs a little: about 100 model calls for the full run). Agent
    tasks run against a scratch database, servicedesk_eval, reset to the sample data."""
    _check_database()
    from datetime import UTC, datetime

    from langgraph.checkpoint.memory import InMemorySaver

    from servicedesk.evals.database import fresh_desk_database
    from servicedesk.evals.dataset import agent_tasks, load_dataset, questions
    from servicedesk.evals.report import render_report
    from servicedesk.evals.runner import evaluate_agent, evaluate_answers, evaluate_retrieval
    from servicedesk.openai_client import model_name
    from servicedesk.rag.llm import OpenAIChat

    cases = load_dataset(dataset)
    question_cases, tasks = questions(cases)[:limit], agent_tasks(cases)[:limit]
    settings = get_settings()
    try:
        retriever = Retriever(get_sessionmaker(), get_embedder())
        llm = get_llm()
        chat_model = get_chat_model(settings) if suite in (EvalSuite.all, EvalSuite.agent) else None
    except ConfigurationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    answerer = Answerer(
        retriever, llm, top_k=settings.rag_top_k, min_similarity=settings.rag_min_similarity
    )
    judge_llm = (
        OpenAIChat(llm.client, judge_model or llm.model, settings.llm_reasoning_effort)
        if judge
        else None
    )
    # Running one suite keeps the others' last results, so the report stays complete.
    results = json.loads(results_path.read_text("utf-8")) if results_path.exists() else {}
    progress = partial(console.print, style="dim")

    if suite in (EvalSuite.all, EvalSuite.retrieval):
        console.print(f"[bold]Retrieval[/bold]: {len(question_cases)} questions, 3 modes")
        results["retrieval"] = evaluate_retrieval(retriever, question_cases, k=settings.rag_top_k)
    if suite in (EvalSuite.all, EvalSuite.answers):
        console.print(f"[bold]Answers[/bold]: {len(question_cases)} questions")
        results["answers"] = evaluate_answers(
            answerer, question_cases, judge=judge_llm, workers=workers, progress=progress
        )
    if suite in (EvalSuite.all, EvalSuite.agent):
        console.print(f"[bold]Agent[/bold]: {len(tasks)} tasks (scratch database reset)")
        desk = ServiceDesk(fresh_desk_database(settings.database_url))
        copilot = build_copilot(chat_model, desk, answerer, InMemorySaver(), settings)
        results["agent"] = evaluate_agent(copilot, tasks, workers=workers, progress=progress)

    results["run"] = {
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "chat_model": llm.model,
        "judge_model": judge_llm.model if judge_llm else results.get("run", {}).get("judge_model"),
        "embedding_model": model_name(settings, "embedding"),
        "questions": len(question_cases),
        "answerable": sum(c.answerable for c in question_cases),
        "out_of_scope": sum(not c.answerable for c in question_cases),
        "agent_tasks": len(tasks),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    report_path.write_text(render_report(results), encoding="utf-8")
    _print_eval_summary(results)
    console.print(f"Report: {report_path}  ·  details: {results_path}")


def _print_eval_summary(results: dict[str, Any]) -> None:
    from servicedesk.evals.report import num, pct

    table = Table(title="Evaluation")
    table.add_column("Metric")
    table.add_column("Result", justify="right")
    if "retrieval" in results:
        k = results["retrieval"]["summary"]["k"]
        hybrid = results["retrieval"]["summary"]["modes"]["hybrid"]
        table.add_row(f"Retrieval hit@{k} (hybrid)", pct(hybrid[f"hit@{k}"]))
        table.add_row("Retrieval MRR (hybrid)", num(hybrid["mrr"]))
    if "answers" in results:
        s = results["answers"]["summary"]
        table.add_row("Answered when answerable", pct(s["answered_when_answerable"]))
        table.add_row("Refused when out of scope", pct(s["refused_when_out_of_scope"]))
        table.add_row("Cites an expected document", pct(s["citation_hit_rate"]))
        if s["judged"]:
            table.add_row("Grounded (judge)", pct(s["grounded_rate"]))
            table.add_row("Correct (judge)", pct(s["correct_rate"]))
    if "agent" in results:
        table.add_row("Agent task completion", pct(results["agent"]["summary"]["task_completion"]))
    console.print(table)


def _print_preview(path: Path, chunker: Chunker) -> None:
    rows, failed = preview(path, chunker)
    table = Table(title=f"Dry run: {path}")
    for column in ["doc_id", "type", "sections", "chunks", "tokens", "source"]:
        table.add_column(column)
    for r in rows:
        table.add_row(
            r.doc_id, r.doc_type, str(r.sections), str(r.chunks), str(r.tokens), r.source_path
        )
    console.print(table)
    console.print(
        f"{len(rows)} documents, {sum(r.chunks for r in rows)} chunks, "
        f"{sum(r.tokens for r in rows)} tokens would be embedded."
    )
    for source, error in failed:
        console.print(f"[red]FAILED[/red] {source}: {error}")
    if failed:
        raise typer.Exit(1)  # same exit code as a real ingest, so CI can use --dry-run as a check


def _check_database(table: str = "chunks", out: Console = console) -> None:
    """Fail early with a helpful message instead of one error per file."""
    try:
        engine = get_engine()
        if not inspect(engine).has_table(table):
            out.print("[red]Tables missing. Run: uv run alembic upgrade head[/red]")
            raise typer.Exit(1)
    except OperationalError as exc:
        out.print(
            "[red]Cannot connect to Postgres. Is Docker running? Try: docker compose up -d[/red]"
        )
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
