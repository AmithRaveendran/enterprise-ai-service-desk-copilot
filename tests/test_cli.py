from pathlib import Path

from typer.testing import CliRunner

from servicedesk.cli import app

KB = Path(__file__).resolve().parent.parent / "data" / "kb"
runner = CliRunner()


def test_dry_run_previews_the_knowledge_base():
    result = runner.invoke(app, ["ingest", str(KB), "--dry-run"])
    assert result.exit_code == 0
    assert "19 documents" in result.output


def test_dry_run_fails_when_a_file_cannot_be_loaded(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("plain text")
    result = runner.invoke(app, ["ingest", str(bad), "--dry-run"])
    assert result.exit_code == 1
    assert "UnsupportedFileTypeError" in result.output


def test_mcp_keeps_stdout_clean_when_it_cannot_start(monkeypatch):
    """Under stdio, stdout is the MCP protocol: even an error message must go to stderr."""
    from sqlalchemy.exc import OperationalError

    def database_down():
        raise OperationalError("connect", {}, Exception("connection refused"))

    monkeypatch.setattr("servicedesk.cli.get_engine", database_down)
    result = runner.invoke(app, ["mcp"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Cannot connect to Postgres" in result.stderr
