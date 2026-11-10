"""Chronological data partitioning for the cross-project temporal setting (PS-8).

All functions operate on data frames that have a ``created`` timestamp column and an
``issue_id`` column (used only to break ties deterministically).

Protocol (see README, section "Data splitting"):

* Target project  -> sorted by creation date; the most recent ``test_frac`` (20 %) of its
  issues form the held-out TEST set ("later issues in a target project").  The earlier
  80 % of the target is never used, which keeps the setting strictly cross-project.
* Source project  -> the DEVELOPMENT data, sorted by creation date.  Optionally restricted
  to issues created before the first target test issue (``strict_chronology``).
* Five-fold validation on the development data uses contiguous chronological blocks
  (no shuffling).  Two schemes are provided:
    - ``blocked``          : 5 blocks; each block is the validation fold once and the
                             other four blocks are used for training (guideline 3.2).
    - ``forward_chaining`` : 6 blocks; run k trains on blocks 1..k and validates on block
                             k+1, so every validation fold is strictly later than its
                             training data (guideline 1.4 / 3.2 temporal rule).
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def chronological(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` sorted oldest -> newest (ties broken by issue id)."""
    return df.sort_values(["created", "issue_id"], kind="mergesort").reset_index(drop=True)


def target_dev_test_split(target: pd.DataFrame, test_frac: float = 0.2):
    """Chronological 80:20 split of the target project. Returns (dev, test)."""
    t = chronological(target)
    n_test = max(1, int(round(len(t) * test_frac)))
    return t.iloc[: len(t) - n_test].reset_index(drop=True), t.iloc[len(t) - n_test :].reset_index(drop=True)


def source_development_set(source: pd.DataFrame, test_start=None, strict: bool = False) -> pd.DataFrame:
    """Chronologically ordered source data; if ``strict`` keep only issues created before ``test_start``."""
    s = chronological(source)
    if strict and test_start is not None:
        s = s[s["created"] < test_start].reset_index(drop=True)
    return s


def blocked_folds(n: int, k: int = 5) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """k contiguous chronological blocks; each is the validation fold exactly once."""
    blocks = np.array_split(np.arange(n), k)
    for i in range(k):
        val = blocks[i]
        train = np.concatenate([b for j, b in enumerate(blocks) if j != i])
        yield i, train, val


def forward_chaining_folds(n: int, k: int = 5) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Expanding-window validation: train on everything before the validation block."""
    blocks = np.array_split(np.arange(n), k + 1)
    for i in range(k):
        train = np.concatenate(blocks[: i + 1])
        val = blocks[i + 1]
        yield i, train, val


def make_folds(n: int, k: int, scheme: str):
    if scheme == "blocked":
        return list(blocked_folds(n, k))
    if scheme == "forward_chaining":
        return list(forward_chaining_folds(n, k))
    raise ValueError(f"unknown CV scheme: {scheme}")
