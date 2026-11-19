import json
from pathlib import Path

import pytest
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


# --- commands against the test database ---------------------------------------------------------
# The commands normally use the dev database (get_sessionmaker / get_engine); here they are
# pointed at the test database, so these tests can change things freely. Rich tables wrap
# to the terminal width, so a wide COLUMNS keeps ids like INC-1002 on one line.

WIDE = {"COLUMNS": "200"}


@pytest.fixture
def cli_database(monkeypatch, test_database_url, desk_session_factory):
    from sqlalchemy import create_engine

    engine = create_engine(test_database_url)
    monkeypatch.setattr("servicedesk.cli.get_engine", lambda: engine)
    monkeypatch.setattr("servicedesk.cli.get_sessionmaker", lambda: desk_session_factory)
    yield desk_session_factory
    engine.dispose()


def test_tools_lists_every_tool_and_marks_the_writes():
    result = runner.invoke(app, ["tools"], env=WIDE)
    assert result.exit_code == 0
    for name in ["get_service_status", "search_tickets", "create_ticket", "add_comment"]:
        assert name in result.output


def test_chunks_shows_how_a_file_is_split():
    result = runner.invoke(app, ["chunks", str(KB / "how-to" / "vpn-setup.md")], env=WIDE)
    assert result.exit_code == 0
    assert "KB-0001" in result.output and "chunk 0" in result.output


@pytest.mark.integration
def test_status_shows_every_service(cli_database):
    result = runner.invoke(app, ["status"], env=WIDE)
    assert result.exit_code == 0
    assert "VPN" in result.output and "degraded" in result.output
    assert "HR Portal" in result.output and "maintenance" in result.output


@pytest.mark.integration
def test_status_of_an_unknown_service_fails(cli_database):
    result = runner.invoke(app, ["status", "Fax machine"], env=WIDE)
    assert result.exit_code == 1


@pytest.mark.integration
def test_set_status_changes_what_status_shows(cli_database):
    changed = runner.invoke(
        app, ["set-status", "VPN", "major_outage", "--message", "Gateway down"], env=WIDE
    )
    assert changed.exit_code == 0 and "major_outage" in changed.output
    shown = runner.invoke(app, ["status", "vpn"], env=WIDE)
    assert "major_outage" in shown.output and "Gateway down" in shown.output


@pytest.mark.integration
def test_seed_is_safe_to_run_again(cli_database):
    result = runner.invoke(app, ["seed"], env=WIDE)
    assert result.exit_code == 0
    assert "sample tickets added: 0" in result.output  # the fixture already seeded them


@pytest.mark.integration
def test_tool_runs_a_tool_like_the_assistant_would(cli_database):
    result = runner.invoke(app, ["tool", "get_ticket", '{"ticket_id": "INC-1002"}'], env=WIDE)
    assert result.exit_code == 0
    assert json.loads(result.output)["result"]["number"] == "INC-1002"


@pytest.mark.integration
def test_tool_with_invalid_arguments_prints_the_errors_and_fails(cli_database):
    result = runner.invoke(app, ["tool", "create_ticket", '{"title": "x"}'], env=WIDE)
    assert result.exit_code == 1
    assert json.loads(result.output)["error"]["code"] == "invalid_input"


@pytest.mark.integration
def test_tool_arguments_must_be_json(cli_database):
    result = runner.invoke(app, ["tool", "get_ticket", "ticket 1002"], env=WIDE)
    assert result.exit_code == 1 and "must be JSON" in result.output


@pytest.mark.integration
def test_keyword_search_works_without_an_api_key(
    monkeypatch, test_database_url, kb_session_factory
):
    from sqlalchemy import create_engine

    engine = create_engine(test_database_url)
    monkeypatch.setattr("servicedesk.cli.get_engine", lambda: engine)
    monkeypatch.setattr("servicedesk.cli.get_sessionmaker", lambda: kb_session_factory)
    result = runner.invoke(
        app, ["search", "SecureLink disconnects macOS", "--mode", "keyword", "-k", "3"], env=WIDE
    )
    engine.dispose()
    assert result.exit_code == 0
    assert "KI-0007" in result.output
