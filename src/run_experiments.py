"""Run the PS-8 experiments for all 39 assigned cross-project pairs.

For every (source -> target) pair:
  1. Target project: chronological 80:20 split; the latest 20 % is the held-out TEST set.
  2. Source project: the DEVELOPMENT set (chronologically ordered).
  3. Model selection: every grid configuration is evaluated with chronological five-fold
     validation on the development set (primary scheme: blocked, 4 folds train / 1 fold
     validation).  The configuration with the lowest mean validation MAE is selected.
  4. The selected configuration is additionally validated with forward-chaining folds
     (validation always later than training) -- reported only, never used for selection.
  5. Final model: refit on the whole development set for the median best epoch found in
     CV, once per seed, and evaluated ONCE on the target test set.
  6. Reference predictors fitted on the same development data and scored on the same test
     set: Mean, Median and a linear model (ridge) on the whole-issue SBERT vector.

Usage:
    python -m src.run_experiments                     # all 39 pairs, settings from config
    python -m src.run_experiments --pairs 1 2 3       # subset (pair ids)
    python -m src.run_experiments --strict --tag strict   # strict cross-project chronology
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from .splits import make_folds, source_development_set, target_dev_test_split
from .train import fit, predict
from .utils import ROOT, dump_json, get_logger, load_config, mae, mdae, resolve, set_seed

_EMB_CACHE: dict[str, dict] = {}


def load_embeddings(slug: str, emb_dir) -> dict:
    if slug not in _EMB_CACHE:
        z = np.load(emb_dir / f"{slug}.npz")
        ids = z["issue_id"]
        _EMB_CACHE[slug] = {
            "pos": pd.Series(np.arange(len(ids)), index=ids),
            "X": z["X"], "mask": z["mask"], "doc": z["doc"],
        }
    return _EMB_CACHE[slug]


def gather(df: pd.DataFrame, emb: dict, device):
    pos = emb["pos"].loc[df["issue_id"].to_numpy()].to_numpy()
    X = torch.from_numpy(emb["X"][pos]).to(device)  # float16 on device, cast per batch
    M = torch.from_numpy(emb["mask"][pos]).to(device)
    y = torch.tensor(df["story_point"].to_numpy(np.float32), device=device)
    doc = emb["doc"][pos].astype(np.float32)
    return X, M, y, doc


def grid_configs(cfg) -> list[dict]:
    g = cfg["grid"]
    keys = list(g.keys())
    out = []
    for vals in itertools.product(*[g[k] for k in keys]):
        hp = dict(zip(keys, vals))
        hp.update(kernel_size=cfg["model"]["kernel_size"], hidden=cfg["model"]["hidden"])
        out.append(hp)
    return out


def cross_validate(hp, X, M, y, folds, cfg, seed):
    rows = []
    for k, tr, va in folds:
        tr_t, va_t = torch.as_tensor(tr, device=X.device), torch.as_tensor(va, device=X.device)
        model, info = fit(X[tr_t], M[tr_t], y[tr_t], hp, cfg["train"], seed, X[va_t], M[va_t], y[va_t])
        p = predict(model, X[va_t], M[va_t], cfg["train"]["target_transform"])
        yt = y[va_t].cpu().numpy()
        rows.append({"fold": k + 1, "n_train": len(tr), "n_val": len(va), "val_mae": mae(yt, p),
                     "val_mdae": mdae(yt, p), "best_epoch": info["best_epoch"]})
        del model
    return rows


def ridge_reference(doc_dev, y_dev, doc_test, folds, alphas=(1.0, 10.0, 100.0, 1000.0)):
    """Linear reference model on the whole-issue SBERT vector; alpha chosen by the same CV."""
    best = None
    for a in alphas:
        errs = []
        for _, tr, va in folds:
            r = Ridge(alpha=a).fit(doc_dev[tr], y_dev[tr])
            errs.append(mae(y_dev[va], np.clip(r.predict(doc_dev[va]), 0, None)))
        if best is None or np.mean(errs) < best[1]:
            best = (a, float(np.mean(errs)), float(np.std(errs)))
    r = Ridge(alpha=best[0]).fit(doc_dev, y_dev)
    return np.clip(r.predict(doc_test), 0, None), {"alpha": best[0], "cv_mae_mean": best[1], "cv_mae_std": best[2]}


def run_pair(pair, issues, cfg, emb_dir, device, log, strict, models_dir):
    t_start = time.time()
    src = issues[issues["project_slug"] == pair["source_slug"]]
    tgt = issues[issues["project_slug"] == pair["target_slug"]]
    _, test = target_dev_test_split(tgt, cfg["split"]["test_frac"])
    test_start = test["created"].min()
    dev = source_development_set(src, test_start, strict=strict)
    if len(dev) < 5 * cfg["split"]["n_folds"]:
        log.warning(f"pair {pair['pair_id']}: only {len(dev)} development issues -> skipped")
        return None

    X_dev, M_dev, y_dev, doc_dev = gather(dev, load_embeddings(pair["source_slug"], emb_dir), device)
    X_te, M_te, y_te, doc_te = gather(test, load_embeddings(pair["target_slug"], emb_dir), device)
    y_dev_np, y_te_np = y_dev.cpu().numpy(), y_te.cpu().numpy()
    seeds = cfg["train"]["seeds"]
    k = cfg["split"]["n_folds"]

    # ---- model selection with chronological 5-fold validation on the source data ----
    folds = make_folds(len(dev), k, cfg["split"]["cv_scheme"])
    cv_rows, summary = [], []
    for ci, hp in enumerate(grid_configs(cfg)):
        rows = cross_validate(hp, X_dev, M_dev, y_dev, folds, cfg, seeds[0])
        for r in rows:
            cv_rows.append({"pair_id": pair["pair_id"], "scheme": cfg["split"]["cv_scheme"], "config": ci, **hp, **r})
        maes = [r["val_mae"] for r in rows]
        summary.append({"config": ci, "hp": hp, "val_mae_mean": float(np.mean(maes)), "val_mae_std": float(np.std(maes)),
                        "val_mdae_mean": float(np.mean([r["val_mdae"] for r in rows])),
                        "val_mdae_std": float(np.std([r["val_mdae"] for r in rows])),
                        "best_epochs": [r["best_epoch"] for r in rows]})
    best = min(summary, key=lambda s: (s["val_mae_mean"], s["val_mdae_mean"]))
    hp = best["hp"]
    final_epochs = int(max(1, np.median(best["best_epochs"])))

    # ---- secondary (forward-chaining) validation of the selected configuration ----
    sec_scheme = cfg["split"].get("secondary_cv_scheme")
    sec = None
    if sec_scheme:
        rows = cross_validate(hp, X_dev, M_dev, y_dev, make_folds(len(dev), k, sec_scheme), cfg, seeds[0])
        for r in rows:
            cv_rows.append({"pair_id": pair["pair_id"], "scheme": sec_scheme, "config": best["config"], **hp, **r})
        sec = {"scheme": sec_scheme, "val_mae_mean": float(np.mean([r["val_mae"] for r in rows])),
               "val_mae_std": float(np.std([r["val_mae"] for r in rows])),
               "val_mdae_mean": float(np.mean([r["val_mdae"] for r in rows])),
               "val_mdae_std": float(np.std([r["val_mdae"] for r in rows]))}

    # ---- final refit on all development data, one model per seed, single test evaluation ----
    preds, seed_rows = [], []
    for s in seeds:
        model, _ = fit(X_dev, M_dev, y_dev, hp, cfg["train"], s, epochs=final_epochs)
        p = predict(model, X_te, M_te, cfg["train"]["target_transform"])
        preds.append(p)
        seed_rows.append({"seed": s, "test_mae": mae(y_te_np, p), "test_mdae": mdae(y_te_np, p)})
        if s == seeds[0] and models_dir is not None:
            torch.save({"state_dict": model.state_dict(), "hp": hp, "emb_dim": X_dev.shape[-1],
                        "target_transform": cfg["train"]["target_transform"], "pair": pair,
                        "sbert_model": cfg["sbert"]["model_name"], "max_sentences": cfg["text"]["max_sentences"]},
                       models_dir / f"pair{pair['pair_id']:02d}_{pair['source_slug']}__{pair['target_slug']}.pt")
        del model

    # ---- reference predictors (same development data, same test set) ----
    mean_p = np.full_like(y_te_np, y_dev_np.mean())
    med_p = np.full_like(y_te_np, np.median(y_dev_np))
    ridge_p, ridge_info = ridge_reference(doc_dev, y_dev_np, doc_te, folds)

    pred_df = pd.DataFrame({
        "pair_id": pair["pair_id"], "issue_id": test["issue_id"].to_numpy(), "created": test["created"].to_numpy(),
        "y_true": y_te_np, **{f"cnn_seed{s}": p for s, p in zip(seeds, preds)},
        "cnn_mean_of_seeds": np.mean(preds, axis=0), "baseline_mean": mean_p, "baseline_median": med_p, "sbert_ridge": ridge_p,
    })

    test_mae = [r["test_mae"] for r in seed_rows]
    test_mdae = [r["test_mdae"] for r in seed_rows]
    result = {
        **pair,
        "strict_chronology": strict,
        "n_source_total": int(len(src)), "n_dev": int(len(dev)), "n_target_total": int(len(tgt)), "n_test": int(len(test)),
        "dev_period": [str(dev["created"].min()), str(dev["created"].max())],
        "test_period": [str(test["created"].min()), str(test["created"].max())],
        "dev_sp_median": float(np.median(y_dev_np)), "dev_sp_mean": float(y_dev_np.mean()),
        "test_sp_median": float(np.median(y_te_np)), "test_sp_mean": float(y_te_np.mean()),
        "selected_config": best["config"], "selected_hp": hp, "final_epochs": final_epochs,
        "cv_primary": {k_: best[k_] for k_ in ["val_mae_mean", "val_mae_std", "val_mdae_mean", "val_mdae_std"]},
        "cv_secondary": sec,
        "grid_summary": [{k_: v for k_, v in s_.items() if k_ != "hp"} for s_ in summary],
        "test_per_seed": seed_rows,
        "cnn_test_mae_mean": float(np.mean(test_mae)), "cnn_test_mae_std": float(np.std(test_mae)),
        "cnn_test_mdae_mean": float(np.mean(test_mdae)), "cnn_test_mdae_std": float(np.std(test_mdae)),
        "baseline_mean_mae": mae(y_te_np, mean_p), "baseline_mean_mdae": mdae(y_te_np, mean_p),
        "baseline_median_mae": mae(y_te_np, med_p), "baseline_median_mdae": mdae(y_te_np, med_p),
        "ridge_mae": mae(y_te_np, ridge_p), "ridge_mdae": mdae(y_te_np, ridge_p), "ridge_info": ridge_info,
        "seconds": round(time.time() - t_start, 1),
    }
    log.info(
        f"pair {pair['pair_id']:2d} {pair['source']} -> {pair['target']}: dev={len(dev)} test={len(test)} | "
        f"CV MAE {best['val_mae_mean']:.2f}±{best['val_mae_std']:.2f} | TEST CNN MAE {result['cnn_test_mae_mean']:.2f} "
        f"MdAE {result['cnn_test_mdae_mean']:.2f} | median-baseline MAE {result['baseline_median_mae']:.2f} | "
        f"{result['seconds']:.0f}s"
    )
    return result, pd.DataFrame(cv_rows), pred_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--pairs", type=int, nargs="*", default=None)
    ap.add_argument("--strict", action="store_true", help="drop source issues created after the target test start")
    ap.add_argument("--tag", default="main", help="results sub-folder name")
    args = ap.parse_args()

    cfg = load_config(args.config)
    strict = args.strict or cfg["split"]["strict_chronology"]
    out_dir = resolve(cfg["paths"]["results_dir"]) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = resolve(cfg["paths"]["models_dir"]) / args.tag
    models_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("run", out_dir / "run.log")
    set_seed(cfg["train"]["seeds"][0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}), strict={strict}")
    issues = pd.read_parquet(resolve(cfg["paths"]["issues"]))
    pairs = pd.read_csv(resolve(cfg["paths"]["pairs"]))
    if args.pairs:
        pairs = pairs[pairs["pair_id"].isin(args.pairs)]
    emb_dir = resolve(cfg["paths"]["embeddings_dir"])

    results, cv_all, pred_all = [], [], []
    for pair in pairs.to_dict("records"):
        out = run_pair(pair, issues, cfg, emb_dir, device, log, strict, models_dir)
        if out is None:
            continue
        r, cv_df, pred_df = out
        results.append(r)
        cv_all.append(cv_df)
        pred_all.append(pred_df)
        dump_json(results, out_dir / "pair_results.json")  # checkpoint after every pair
        pd.concat(cv_all).to_csv(out_dir / "cv_folds.csv", index=False)
        pd.concat(pred_all).to_csv(out_dir / "test_predictions.csv", index=False)
        torch.cuda.empty_cache()
    dump_json({"config": cfg, "strict": strict, "n_pairs": len(results)}, out_dir / "run_config.json")
    log.info(f"done: {len(results)} pairs -> {out_dir}")


if __name__ == "__main__":
    main()
