"""Command-line interface: `uv run servicedesk --help`."""

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError

from servicedesk.config import get_settings
from servicedesk.db.session import get_engine, get_sessionmaker
from servicedesk.ingestion.chunker import Chunker
from servicedesk.ingestion.embeddings import ConfigurationError, get_embedder
from servicedesk.ingestion.loaders import load_document
from servicedesk.ingestion.pipeline import IngestionPipeline, preview
from servicedesk.rag.answerer import Answerer
from servicedesk.rag.llm import get_llm
from servicedesk.search.retriever import Retriever, SearchFilters, SearchMode

app = typer.Typer(help="Enterprise AI Service Desk Copilot tools.", no_args_is_help=True)
console = Console()


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
def serve(
    host: Annotated[str, typer.Option(help="Interface to listen on.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Restart when code changes (dev).")] = False,
) -> None:
    """Run the HTTP API (docs at http://127.0.0.1:8000/docs)."""
    import uvicorn  # imported here so other commands don't pay for loading the web stack

    uvicorn.run("servicedesk.api.app:app", host=host, port=port, reload=reload)


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


def _check_database() -> None:
    """Fail early with a helpful message instead of one error per file."""
    try:
        engine = get_engine()
        if not inspect(engine).has_table("chunks"):
            console.print("[red]Tables missing. Run: uv run alembic upgrade head[/red]")
            raise typer.Exit(1)
    except OperationalError as exc:
        console.print(
            "[red]Cannot connect to Postgres. Is Docker running? Try: docker compose up -d[/red]"
        )
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
