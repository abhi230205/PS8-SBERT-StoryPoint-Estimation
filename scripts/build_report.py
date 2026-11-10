"""Build the final report (HTML -> PDF) from report/report_template.html and the results.

Placeholders in the template
  {{TABLE:<path relative to results/>}}        CSV rendered as an HTML table
  {{FIG:<path relative to results/>|caption}}  figure embedded as base64 PNG
  {{NUM:<run>:<dotted.key>:<fmt>}}             value from results/<run>/summary.json
  {{CSVNUM:<path>:<row>:<col>}}                single cell of a CSV

Usage:  python scripts/build_report.py [--pdf]
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium",
]


def table(path: str) -> str:
    df = pd.read_csv(RES / path, dtype=str, keep_default_na=False)  # keep numbers exactly as written (e.g. 1.40)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            s = "" if (isinstance(v, float) and pd.isna(v)) else str(v)
            cls = ' class="num"' if re.fullmatch(r"[-+]?[\d.,]+( ± [\d.]+)?( / \d+ / \d+)?", s.strip()) or re.fullmatch(r"\d+ / \d+ / \d+", s) else ""
            cells.append(f"<td{cls}>{html.escape(s)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def figure(spec: str) -> str:
    """``path|caption`` or ``path|caption|width`` (CSS width of the image, e.g. 70%)."""
    parts = spec.split("|")
    path, caption = parts[0], parts[1]
    width = parts[2] if len(parts) > 2 else "100%"
    b64 = base64.b64encode((RES / path).read_bytes()).decode()
    return (f'<figure><img style="width:{width}" src="data:image/png;base64,{b64}" alt="{html.escape(caption)}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


def num(spec: str) -> str:
    run, key, fmt = spec.split(":")
    obj = json.loads((RES / run / "summary.json").read_text())
    for k in key.split("."):
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    return format(obj, fmt) if fmt else str(obj)


def csvnum(spec: str) -> str:
    path, row, col = spec.split(":")
    df = pd.read_csv(RES / path)
    return str(df.iloc[int(row)][col])


def build() -> Path:
    tpl = (ROOT / "report" / "report_template.html").read_text(encoding="utf-8")
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    tpl = tpl.replace("{{GITHUB_URL}}", cfg["project"]["github_url"])
    # numbers first: they may appear inside figure captions
    tpl = re.sub(r"\{\{NUM:([^}]+)\}\}", lambda m: num(m.group(1)), tpl)
    tpl = re.sub(r"\{\{CSVNUM:([^}]+)\}\}", lambda m: csvnum(m.group(1)), tpl)
    tpl = re.sub(r"\{\{TABLE:([^}]+)\}\}", lambda m: table(m.group(1)), tpl)
    tpl = re.sub(r"\{\{FIG:([^}]+)\}\}", lambda m: figure(m.group(1)), tpl)
    left = re.findall(r"\{\{[^}]*\}\}", tpl)
    if left:
        raise SystemExit(f"unresolved placeholders: {left[:5]}")
    out = ROOT / "report" / "report.html"
    out.write_text(tpl, encoding="utf-8")
    return out


def to_pdf(html_path: Path, pdf_name: str = "PS8_Group1_Report.pdf") -> Path:
    pdf = html_path.with_name(pdf_name)
    exe = next((b for b in BROWSERS if Path(b).exists()), None)
    if exe is None:
        raise SystemExit("no Chromium-based browser found; open report/report.html and print to PDF manually")
    import tempfile
    import time

    before = pdf.stat().st_mtime if pdf.exists() else 0
    # a throw-away profile so the job is not handed to an already-running browser window
    with tempfile.TemporaryDirectory(prefix="ps8-pdf-") as profile:
        subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-first-run", "--no-pdf-header-footer",
                        f"--user-data-dir={profile}", f"--print-to-pdf={pdf}", html_path.resolve().as_uri()],
                       check=True, timeout=180)
        time.sleep(1)
    if not pdf.exists() or pdf.stat().st_mtime <= before:
        raise SystemExit("PDF was not (re)written -- close the browser or print report/report.html manually")
    return pdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()
    h = build()
    print("wrote", h)
    if args.pdf:
        print("wrote", to_pdf(h))
