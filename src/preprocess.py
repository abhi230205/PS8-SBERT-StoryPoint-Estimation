"""Clean the extracted TAWOS issues and split each issue text into sentences.

Steps
  1. keep the 39 projects of the assignment (configs/project_map.csv)
  2. keep issues whose resolution means the work was completed, or -- when no resolution
     was recorded -- whose status is Closed/Done/Resolved/Complete (story points of
     Won't-Fix / Duplicate / unresolved issues do not reflect effort that was spent)
  3. keep 0 < story point <= data.max_story_point (removes zero and data-entry outliers
     such as 26,710,114 or 6,765 points)
  4. text = title + natural-language description (TAWOS ``Description_Text``, i.e. the
     description with code blocks removed); light Jira-markup / URL cleaning
  5. sentence segmentation (line breaks + end-of-sentence punctuation)
  6. drop exact duplicates (same project, text and story point)

Output: data/processed/issues.parquet  and  results/data_summary.csv

Usage:  python -m src.preprocess
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .utils import ROOT, get_logger, load_config, resolve

COMPLETED = {"Fixed", "Done", "Complete", "Completed", "Implemented", "Resolved", "Deployed"}
DONE_STATUS = {"Closed", "Done", "Resolved", "Complete"}  # used when no resolution was recorded (e.g. MXNet)
_QUOTES = re.compile(r'"{2,}')

_URL = re.compile(r"(https?://|www\.)\S+", re.I)
_USER = re.compile(r"\[~[^\]]*\]?|<USER>")
_JIRA_LINK = re.compile(r"\[([^\]|]*)\|[^\]]*\]")  # [text|url] -> text
_IMAGE = re.compile(r"![^!\s]+!")
_MARKUP = re.compile(r"\{(?:code|noformat|quote|panel|color)[^}]*\}|\bh[1-6]\.\s*|\{\{|\}\}|bq\.\s*")
_LIST = re.compile(r"(^|\s)[#*-]+\s+")
_TABLE = re.compile(r"\|\|?")
_WS = re.compile(r"[ \t\x0b\x0c]+")
# line breaks were stored as two (or more) spaces in TAWOS
_LINEBREAK = re.compile(r"\s{2,}|\r|\n")
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def strip_quotes(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip()
    while len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    return _QUOTES.sub('"', s)  # TAWOS doubles embedded quotes (sometimes twice)


def clean(s: str) -> str:
    s = _URL.sub(" URL ", s)
    s = _USER.sub(" USER ", s)
    s = _JIRA_LINK.sub(r"\1", s)
    s = _IMAGE.sub(" ", s)
    s = _MARKUP.sub(" ", s)
    s = _TABLE.sub(" ", s)
    s = s.replace("*", "").replace(" ", " ")
    return s


def split_sentences(text: str, min_chars: int, max_chars: int) -> list[str]:
    out = []
    for line in _LINEBREAK.split(text):
        line = _LIST.sub(" ", line)
        line = _WS.sub(" ", line).strip()
        if not line:
            continue
        for sent in _SENT_END.split(line):
            sent = sent.strip()
            if len(sent) >= min_chars and re.search(r"[A-Za-z]", sent):
                out.append(sent[:max_chars])
    return out


def main():
    cfg = load_config()
    log = get_logger("preprocess", ROOT / "results" / "logs" / "preprocess.log")
    raw = pd.read_parquet(ROOT / "data" / "interim" / "tawos_issues_with_sp.parquet")
    pmap = pd.read_csv(ROOT / "configs" / "project_map.csv")
    tcfg, dcfg = cfg["text"], cfg["data"]

    steps = [("TAWOS issues with a story point", len(raw))]
    df = raw[raw["Project_ID"].isin(pmap["tawos_project_id"])].copy()
    steps.append(("in the 39 assigned projects", len(df)))
    done = df["Resolution"].isin(COMPLETED) | (df["Resolution"].isna() & df["Status"].isin(DONE_STATUS))
    df = df[done]
    steps.append(("completed work (resolution or done-status)", len(df)))
    df = df[(df["Story_Point"] > 0) & (df["Story_Point"] <= dcfg["max_story_point"])]
    steps.append((f"0 < story point <= {dcfg['max_story_point']}", len(df)))
    df = df[df["Creation_Date"].notna()]

    df["title"] = df["Title"].map(strip_quotes).map(clean).map(lambda s: _WS.sub(" ", s).strip())
    df["description"] = df["Description_Text"].map(strip_quotes).map(clean)
    df = df[df["title"].str.len() > 0]
    steps.append(("non-empty title", len(df)))

    df["desc_sentences"] = df["description"].map(lambda s: split_sentences(s, tcfg["min_sentence_chars"], tcfg["max_sentence_chars"]))
    df["sentences"] = [[t[: tcfg["max_sentence_chars"]]] + d for t, d in zip(df["title"], df["desc_sentences"])]
    df["full_text"] = [t + ". " + " ".join(d) for t, d in zip(df["title"], df["desc_sentences"])]

    if dcfg["dedupe"]:
        df = df.drop_duplicates(subset=["Project_ID", "full_text", "Story_Point"], keep="first")
        steps.append(("after removing exact duplicates", len(df)))

    df = df.merge(pmap[["tawos_project_id", "project_key", "guideline_name"]], left_on="Project_ID", right_on="tawos_project_id")
    out = pd.DataFrame({
        "issue_id": df["ID"].astype("int64"),
        "issue_key": df["Issue_Key"],
        "project_slug": df["project_key"],
        "project_name": df["guideline_name"],
        "created": df["Creation_Date"],
        "story_point": df["Story_Point"].astype("float32"),
        "type": df["Type"],
        "title": df["title"],
        "full_text": df["full_text"],
        "sentences": df["sentences"],
        "n_sentences": df["sentences"].map(len),
    }).sort_values(["project_slug", "created", "issue_id"]).reset_index(drop=True)

    path = resolve(cfg["paths"]["issues"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)

    for name, n in steps:
        log.info(f"{name:45s} {n:7d}")
    ns = out["n_sentences"]
    log.info(f"sentences/issue: mean {ns.mean():.1f}, median {ns.median():.0f}, p90 {ns.quantile(.9):.0f}, "
             f"p95 {ns.quantile(.95):.0f}, share <= {tcfg['max_sentences']}: {(ns <= tcfg['max_sentences']).mean():.3f}")

    summ = out.groupby(["project_slug", "project_name"]).agg(
        issues=("issue_id", "size"), first_created=("created", "min"), last_created=("created", "max"),
        sp_mean=("story_point", "mean"), sp_median=("story_point", "median"), sp_std=("story_point", "std"),
        sp_max=("story_point", "max"), sentences_mean=("n_sentences", "mean"),
    ).reset_index().sort_values("project_name")
    summ["first_created"] = summ["first_created"].dt.date
    summ["last_created"] = summ["last_created"].dt.date
    summ.columns = ["Key", "Project (guideline name)", "Issues", "First created", "Last created", "SP mean",
                    "SP median", "SP std", "SP max", "Sentences / issue"]
    (ROOT / "results").mkdir(exist_ok=True)
    summ.to_csv(ROOT / "results" / "data_summary.csv", index=False, float_format="%.2f")
    pd.DataFrame(steps, columns=["Filtering step", "Issues remaining"]).to_csv(ROOT / "results" / "data_filtering.csv", index=False)
    log.info(f"saved {len(out)} issues of {out['project_slug'].nunique()} projects -> {path}")


if __name__ == "__main__":
    main()
