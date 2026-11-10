"""Compute and cache SBERT sentence embeddings for every issue of the 39 projects.

For each issue we build the sentence sequence  [title, s_1, s_2, ..., s_{L-1}]  (description
sentences in reading order, truncated to ``text.max_sentences``) and encode every sentence
with a pre-trained, frozen SBERT model (sentence-transformers).  Identical sentences are
encoded only once.  We also store one SBERT vector for the whole issue text
("title. description", truncated at the model's max_seq_length) for the reference model.

Output: data/embeddings/<project_slug>.npz with
    issue_id  (N,)        int64
    X         (N, L, d)   float16   sentence-embedding sequence, zero padded
    mask      (N, L)      bool      True where a real sentence is present
    doc       (N, d)      float16   whole-issue SBERT embedding
plus data/embeddings/meta.json (model name, dimensions, timings, library versions).

Usage:  python -m src.embed_sbert
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from .utils import ROOT, dump_json, get_logger, load_config, resolve, set_seed


def main():
    cfg = load_config()
    log = get_logger("embed", ROOT / "results" / "logs" / "embed.log")
    set_seed(cfg["train"]["seeds"][0])
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    import transformers

    scfg, L = cfg["sbert"], cfg["text"]["max_sentences"]
    issues = pd.read_parquet(resolve(cfg["paths"]["issues"]))
    out_dir = resolve(cfg["paths"]["embeddings_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(scfg["model_name"], device=device)
    model.max_seq_length = scfg["max_seq_length"]
    d = model.get_sentence_embedding_dimension()
    log.info(f"SBERT {scfg['model_name']} on {device}; dim={d}; max_seq_length={model.max_seq_length}")

    # ---- sentence-level embeddings (deduplicated) ----
    seqs = [list(s)[:L] for s in issues["sentences"]]
    uniq = sorted({s for seq in seqs for s in seq})
    log.info(f"{len(issues)} issues, {sum(map(len, seqs))} sentences, {len(uniq)} unique")
    t0 = time.time()
    with torch.inference_mode():
        E = model.encode(
            uniq, batch_size=scfg["batch_size"], convert_to_numpy=True, show_progress_bar=True,
            normalize_embeddings=scfg["normalize_embeddings"],
        )
    t_sent = time.time() - t0
    index = {s: i for i, s in enumerate(uniq)}
    log.info(f"sentence encoding took {t_sent:.0f}s")

    # ---- whole-issue embeddings ----
    t0 = time.time()
    with torch.inference_mode():
        D = model.encode(
            issues["full_text"].tolist(), batch_size=max(16, scfg["batch_size"] // 4), convert_to_numpy=True,
            show_progress_bar=True, normalize_embeddings=scfg["normalize_embeddings"],
        )
    t_doc = time.time() - t0
    log.info(f"document encoding took {t_doc:.0f}s")

    dtype = np.float16 if scfg["dtype"] == "float16" else np.float32
    issues = issues.reset_index(drop=True)
    for slug, g in issues.groupby("project_slug", sort=True):
        rows = g.index.to_numpy()
        X = np.zeros((len(rows), L, d), dtype=dtype)
        M = np.zeros((len(rows), L), dtype=bool)
        for j, r in enumerate(rows):
            ids = [index[s] for s in seqs[r]]
            X[j, : len(ids)] = E[ids]
            M[j, : len(ids)] = True
        np.savez(
            out_dir / f"{slug}.npz",
            issue_id=g["issue_id"].to_numpy(np.int64), X=X, mask=M, doc=D[rows].astype(dtype),
        )
        log.info(f"saved {slug}: {len(rows)} issues")

    dump_json(
        {
            "model_name": scfg["model_name"], "dim": d, "max_sentences": L,
            "max_seq_length": model.max_seq_length, "device": device,
            "n_issues": len(issues), "n_unique_sentences": len(uniq),
            "seconds_sentences": round(t_sent, 1), "seconds_documents": round(t_doc, 1),
            "sentence_transformers": sentence_transformers.__version__,
            "transformers": transformers.__version__, "torch": torch.__version__,
        },
        out_dir / "meta.json",
    )


if __name__ == "__main__":
    main()
