"""Stream the TAWOS MySQL dump (inside TAWOS.sql.zip) and extract the tables we need.

No MySQL server is required: the dump is read line by line straight out of the zip
archive, and only the ``Issue``, ``Project`` and ``Repository`` tables are parsed.
Issues without a story point are discarded at this stage to save memory.

Outputs (data/interim/):
    tawos_issues_with_sp.parquet   all issues that carry a story point (all 39 projects)
    tawos_projects.csv             Project table joined with Repository names

Usage:  python -m src.extract_tawos
"""
from __future__ import annotations

import io
import re
import time
import zipfile

import pandas as pd

from .utils import ROOT, get_logger, load_config, resolve

WANTED = {"Issue", "Project", "Repository"}
ISSUE_KEEP = [
    "ID", "Issue_Key", "Title", "Description", "Description_Text", "Description_Code", "Type",
    "Priority", "Status", "Resolution", "Creation_Date", "Estimation_Date", "Resolution_Date",
    "Story_Point", "Title_Changed_After_Estimation", "Description_Changed_After_Estimation",
    "Story_Point_Changed_After_Estimation", "Project_ID",
]

_TOKEN = re.compile(r"'((?:[^'\\]|\\.)*)'|(NULL)|(\),\()|([^,()']+)", re.S)
_ESC = re.compile(r"\\(.)", re.S)
_ESC_MAP = {"0": "\x00", "b": "\b", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a"}


def _unescape(s: str) -> str:
    if "\\" not in s:
        return s
    return _ESC.sub(lambda m: _ESC_MAP.get(m.group(1), m.group(1)), s)


def parse_values(payload: str):
    """Parse the ``(...),(...)`` part of an extended INSERT into a list of row lists."""
    rows, row = [], []
    for m in _TOKEN.finditer(payload):
        s, null, sep, bare = m.groups()
        if s is not None:
            row.append(_unescape(s))
        elif null is not None:
            row.append(None)
        elif sep is not None:
            rows.append(row)
            row = []
        elif bare is not None:
            b = bare.strip()
            if b:
                row.append(b)
    if row:
        rows.append(row)
    return rows


def main():
    cfg = load_config()
    log = get_logger("extract", ROOT / "results" / "logs" / "extract.log")
    zpath = resolve(cfg["paths"]["raw_zip"])
    out_dir = ROOT / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)

    columns: dict[str, list[str]] = {}
    data: dict[str, list[list]] = {t: [] for t in WANTED}
    current_table, current_cols = None, []
    create_re = re.compile(r"CREATE TABLE `(\w+)`")
    col_re = re.compile(r"\s+`(\w+)`\s")
    t0 = time.time()
    n_lines = 0

    with zipfile.ZipFile(zpath) as zf:
        name = [n for n in zf.namelist() if n.endswith(".sql")][0]
        with zf.open(name) as raw:
            stream = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="\n")
            for line in stream:
                n_lines += 1
                if line.startswith("CREATE TABLE"):
                    current_table = create_re.match(line).group(1)
                    current_cols = []
                    continue
                if current_table and line.startswith("  `"):
                    current_cols.append(col_re.match(line).group(1))
                    continue
                if current_table and line.startswith(")"):
                    columns[current_table] = current_cols
                    current_table = None
                    continue
                if line.startswith("INSERT INTO `"):
                    table = line[13 : line.index("`", 13)]
                    if table not in WANTED:
                        continue
                    payload = line[line.index("VALUES (") + 8 : line.rstrip().rstrip(";").rstrip().rfind(")")]
                    rows = parse_values(payload)
                    cols = columns[table]
                    for r in rows:
                        if len(r) != len(cols):
                            raise ValueError(f"{table}: parsed {len(r)} fields, expected {len(cols)}")
                    if table == "Issue":
                        sp = cols.index("Story_Point")
                        rows = [r for r in rows if r[sp] is not None]
                    data[table].extend(rows)
                    log.info(f"{table}: +{len(rows)} rows (total {len(data[table])}) [{time.time()-t0:.0f}s]")

    log.info(f"read {n_lines} lines in {time.time()-t0:.0f}s; tables: { {t: len(v) for t, v in data.items()} }")

    issues = pd.DataFrame(data["Issue"], columns=columns["Issue"])
    issues = issues[[c for c in ISSUE_KEEP if c in issues.columns]]
    issues["Story_Point"] = pd.to_numeric(issues["Story_Point"], errors="coerce")
    for c in ["ID", "Project_ID"]:
        issues[c] = pd.to_numeric(issues[c], errors="coerce").astype("Int64")
    for c in ["Creation_Date", "Estimation_Date", "Resolution_Date"]:
        issues[c] = pd.to_datetime(issues[c], errors="coerce")
    issues.to_parquet(out_dir / "tawos_issues_with_sp.parquet", index=False)

    projects = pd.DataFrame(data["Project"], columns=columns["Project"])
    repos = pd.DataFrame(data["Repository"], columns=columns["Repository"])
    log.info(f"Project columns: {list(projects.columns)}")
    log.info(f"Repository columns: {list(repos.columns)}")
    projects.to_csv(out_dir / "tawos_projects_raw.csv", index=False)
    repos.to_csv(out_dir / "tawos_repositories_raw.csv", index=False)
    log.info(f"saved {len(issues)} issues with story points, {len(projects)} projects, {len(repos)} repositories")


if __name__ == "__main__":
    main()
