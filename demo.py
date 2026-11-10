"""Demo: estimate story points for an issue with a trained SBERT + CNN model.

Examples
  # predict a new issue with the model trained on source project XD (pair 1, XD -> MESOS)
  python demo.py --pair 1 --title "Fix login failure" --description "The login fails when a user enters an incorrect password."

  # show predictions for a few real held-out test issues of the target project
  python demo.py --pair 1 --show-test 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocess import clean, split_sentences  # noqa: E402
from src.splits import target_dev_test_split  # noqa: E402
from src.train import build_model  # noqa: E402
from src.utils import load_config, resolve  # noqa: E402


def load_pair_model(pair_id: int, tag: str, device):
    files = sorted((ROOT / "models" / tag).glob(f"pair{pair_id:02d}_*.pt"))
    if not files:
        raise SystemExit(f"no model for pair {pair_id} in models/{tag} -- run `python -m src.run_experiments` first")
    ckpt = torch.load(files[0], map_location=device, weights_only=False)
    model = build_model(ckpt["hp"], ckpt["emb_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def encode_issue(sbert, title: str, description: str, max_sentences: int, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    tc = cfg["text"]
    sents = [clean(title).strip()] + split_sentences(clean(description), tc["min_sentence_chars"], tc["max_sentence_chars"])
    sents = sents[:max_sentences]
    E = sbert.encode(sents, convert_to_numpy=True)
    X = np.zeros((1, max_sentences, E.shape[1]), np.float32)
    M = np.zeros((1, max_sentences), bool)
    X[0, : len(sents)] = E
    M[0, : len(sents)] = True
    return torch.from_numpy(X), torch.from_numpy(M), sents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=1, help="pair id from configs/pairs.csv (model trained on its source project)")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--title", default="Fix the login failure when a user enters an incorrect password")
    ap.add_argument("--description", default="When a user types a wrong password the login page throws an exception instead of showing an error message. Add validation and a friendly error.")
    ap.add_argument("--show-test", type=int, default=0, help="also predict N random held-out test issues of the target project")
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_pair_model(args.pair, args.tag, device)
    pair = ckpt["pair"]
    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer(ckpt["sbert_model"], device=str(device))
    sbert.max_seq_length = cfg["sbert"]["max_seq_length"]
    print(f"Model of pair {pair['pair_id']}: trained on '{pair['source']}', evaluated on '{pair['target']}'  (hp={ckpt['hp']})")

    X, M, sents = encode_issue(sbert, args.title, args.description, ckpt["max_sentences"], cfg)
    with torch.no_grad():
        z = model(X.to(device), M.to(device))
    sp = float(torch.expm1(z) if ckpt["target_transform"] == "log1p" else z)
    print("\nInput sentences:")
    for s in sents:
        print("  -", s)
    print(f"\nPredicted story points: {max(sp, 0):.2f}")

    if args.show_test:
        issues = pd.read_parquet(resolve(cfg["paths"]["issues"]))
        _, test = target_dev_test_split(issues[issues["project_slug"] == pair["target_slug"]], cfg["split"]["test_frac"])
        sample = test.sample(min(args.show_test, len(test)), random_state=0)
        print(f"\n{len(sample)} random held-out test issues of {pair['target']} (latest 20 %):")
        for r in sample.itertuples():
            Xi, Mi, _ = encode_issue(sbert, r.title, " ".join(list(r.sentences)[1:]), ckpt["max_sentences"], cfg)
            with torch.no_grad():
                p = float(model(Xi.to(device), Mi.to(device)))
            print(f"  {r.issue_key:<14} true={r.story_point:5.1f}  predicted={max(p, 0):5.2f}  | {r.title[:70]}")


if __name__ == "__main__":
    main()
