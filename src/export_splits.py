"""Export the exact data partitions of every pair so other representations can reuse them.

Guideline 3.2 asks to use the same partitions when comparing word, sentence and combined
embeddings.  This writes results/splits/pair_XX.csv with one row per issue:

    issue_id, issue_key, project, created, story_point, role (dev|test), blocked_fold (1-5, dev only)

and results/splits/README.md describing the protocol.

Usage:  python -m src.export_splits [--strict]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .splits import make_folds, source_development_set, target_dev_test_split
from .utils import ROOT, load_config, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    issues = pd.read_parquet(resolve(cfg["paths"]["issues"]))
    pairs = pd.read_csv(resolve(cfg["paths"]["pairs"]))
    out = ROOT / "results" / ("splits_strict" if args.strict else "splits")
    out.mkdir(parents=True, exist_ok=True)
    cols = ["issue_id", "issue_key", "project_slug", "created", "story_point"]
    for p in pairs.itertuples():
        _, test = target_dev_test_split(issues[issues.project_slug == p.target_slug], cfg["split"]["test_frac"])
        dev = source_development_set(issues[issues.project_slug == p.source_slug], test["created"].min(), strict=args.strict)
        fold = np.zeros(len(dev), int)
        for k, _, va in make_folds(len(dev), cfg["split"]["n_folds"], "blocked"):
            fold[va] = k + 1
        d = dev[cols].assign(role="dev", blocked_fold=fold)
        t = test[cols].assign(role="test", blocked_fold=0)
        pd.concat([d, t]).rename(columns={"project_slug": "project"}).to_csv(out / f"pair_{p.pair_id:02d}.csv", index=False)
    (out / "README.md").write_text(
        "# Data partitions per pair\n\n"
        "`role=test`: latest 20 % of the target project (chronological). `role=dev`: the source project, chronologically "
        "ordered; `blocked_fold` is the 5-fold chronological validation fold (1 = oldest).  Issues are identified by the "
        "TAWOS `Issue.ID` (`issue_id`) and `Issue_Key`.\n", encoding="utf-8")
    print(f"wrote {len(pairs)} split files to {out}")


if __name__ == "__main__":
    main()
