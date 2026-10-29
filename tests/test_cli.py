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
