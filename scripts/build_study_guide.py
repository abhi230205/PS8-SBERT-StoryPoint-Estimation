"""Convert report/Study_Guide.md into a printable PDF (report/PS8_Study_Guide.pdf).

Usage:  python scripts/build_study_guide.py      (needs the `markdown` package and Edge or Chrome)
"""
from __future__ import annotations

import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_report import to_pdf  # noqa: E402

CSS = """
@page { size: A4; margin: 16mm 16mm 18mm 16mm;
        @bottom-center { content: counter(page); font-family: Cambria, Georgia, serif; font-size: 9pt; color: #555; } }
html, body { background: #fff; }
body { font-family: Cambria, Georgia, "Times New Roman", serif; color: #111; font-size: 10.5pt; line-height: 1.45; margin: 0; }
h1 { font-size: 20pt; margin: 0 0 6pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; padding-bottom: 3pt; border-bottom: 1px solid #ccc; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; break-after: avoid; }
p { margin: 0 0 7pt; }
ul, ol { margin: 0 0 7pt; padding-left: 18pt; }
li { margin-bottom: 2pt; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 10pt; font-size: 9.5pt; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { border: 1px solid #bdbdbd; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #f0f0f0; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt; background: #f4f4f4; padding: 0 2px; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 7pt 9pt; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { margin: 6pt 0 10pt; padding: 6pt 10pt; border-left: 3px solid #1f4e8c; background: #f5f8fc; }
blockquote p { margin: 0; }
hr { border: 0; border-top: 1px solid #ddd; margin: 14pt 0; }
a { color: #1f4e8c; text-decoration: none; }
"""


def main():
    md_path = ROOT / "report" / "Study_Guide.md"
    text = md_path.read_text(encoding="utf-8").replace("- [ ]", "- ☐")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>PS-8 Study Guide – Group 1</title><style>{CSS}</style></head><body>{body}</body></html>")
    out_html = ROOT / "report" / "study_guide.html"
    out_html.write_text(html, encoding="utf-8")
    pdf = to_pdf(out_html, "PS8_Study_Guide.pdf")
    out_html.unlink()  # the Markdown file is the source; keep only the PDF
    print("wrote", pdf)


if __name__ == "__main__":
    main()
