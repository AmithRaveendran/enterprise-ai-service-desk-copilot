"""Render the policy sources in scripts/pdf_sources/ into PDFs under data/kb/policies/.

Real knowledge bases contain PDFs, so we generate a few to make ingestion realistic.
Each page gets a running header and footer, which the ingestion pipeline must strip.

Source format: YAML front matter, then lines starting with "# " (title), "## " (section),
"- " (bullet), or plain paragraph text.

Usage:
    uv run python scripts/build_pdfs.py
"""

from pathlib import Path

import yaml
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "scripts" / "pdf_sources"
OUTPUT_DIR = ROOT / "data" / "kb" / "policies"


class PolicyPDF(FPDF):
    def __init__(self, meta: dict):
        super().__init__(format="A4")
        self.meta = meta
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 8, f"Halcyon Dynamics | {self.meta['id']} | INTERNAL", align="R")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(
            0,
            10,
            f"Last updated {self.meta['last_updated']} | Page {self.page_no()}/{{nb}}",
            align="C",
        )


def parse_source(path: Path) -> tuple[dict, list[str]]:
    _, front_matter, body = path.read_text(encoding="utf-8").split("---", 2)
    return yaml.safe_load(front_matter), body.strip().splitlines()


def render(meta: dict, lines: list[str], out_path: Path) -> None:
    pdf = PolicyPDF(meta)
    # Embedded metadata, which the PDF loader can read in Phase 1.
    pdf.set_title(meta["title"])
    pdf.set_subject(meta["id"])
    pdf.set_author(meta["owner"])
    pdf.set_keywords(f"category={meta['category']}; service={meta['service']}")
    pdf.add_page()

    for line in lines:
        line = line.strip()
        pdf.set_text_color(0)
        if not line:
            continue
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 18)
            pdf.multi_cell(0, 10, line[2:])
            pdf.ln(4)
        elif line.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 8, line[3:])
            pdf.ln(1)
        elif line.startswith("- "):
            pdf.set_font("Helvetica", "", 11)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(0, 6, "- " + line[2:])
            pdf.ln(1)
        else:
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, line)
            pdf.ln(2)

    pdf.output(str(out_path))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for source in sorted(SOURCE_DIR.glob("*.md")):
        meta, lines = parse_source(source)
        out_path = OUTPUT_DIR / f"{source.stem}.pdf"
        render(meta, lines, out_path)
        print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
