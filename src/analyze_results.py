"""Aggregate the per-pair results into the tables, statistics and figures of the report.

Reads   results/<tag>/pair_results.json, cv_folds.csv, test_predictions.csv
Writes  results/<tag>/tables/*.csv|*.md, results/<tag>/summary.json, results/<tag>/figures/*.png

Usage:  python -m src.analyze_results [--tag main] [--compare strict]
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

from .splits import make_folds, source_development_set, target_dev_test_split  # noqa: E402
from .utils import ROOT, dump_json, load_config, mae, mdae, resolve  # noqa: E402

# Reference palette (light mode) -- see README "Figures".
# (surface is pure white instead of #fcfcfb so the figures sit seamlessly on the white report page)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_CNN, C_MED, C_RIDGE = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def a12(x, y) -> float:
    """Vargha-Delaney A12: probability that a value from x is larger than one from y."""
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def a12_label(a: float) -> str:
    d = abs(a - 0.5)
    return "negligible" if d < 0.06 else "small" if d < 0.14 else "medium" if d < 0.21 else "large"


def fmt_pm(m, s, nd=2):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def hp_str(hp: dict) -> str:
    return f"lr={hp['lr']:g}, dropout={hp['dropout']:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--compare", default=None, help="tag of a second run (e.g. strict) to compare on common pairs")
    args = ap.parse_args()
    cfg = load_config()
    base = resolve(cfg["paths"]["results_dir"]) / args.tag
    tdir, fdir = base / "tables", base / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    res = json.loads((base / "pair_results.json").read_text())
    cv = pd.read_csv(base / "cv_folds.csv")
    pred = pd.read_csv(base / "test_predictions.csv")
    R = pd.DataFrame(res)
    seeds = cfg["train"]["seeds"]

    # ------------------------------------------------------------------ per-pair table
    per_pair = pd.DataFrame({
        "Pair": R["pair_id"],
        "Source (train)": R["source"],
        "Target (test)": R["target"],
        "n dev": R["n_dev"],
        "n test": R["n_test"],
        "Selected config": R["selected_hp"].map(hp_str),
        "Epochs": R["final_epochs"],
        "5-fold val MAE": [fmt_pm(c["val_mae_mean"], c["val_mae_std"]) for c in R["cv_primary"]],
        "5-fold val MdAE": [fmt_pm(c["val_mdae_mean"], c["val_mdae_std"]) for c in R["cv_primary"]],
        "Fwd-chain val MAE": [fmt_pm(c["val_mae_mean"], c["val_mae_std"]) if c else "-" for c in R["cv_secondary"]],
        "Test MAE": [fmt_pm(m, s) for m, s in zip(R["cnn_test_mae_mean"], R["cnn_test_mae_std"])],
        "Test MdAE": [fmt_pm(m, s) for m, s in zip(R["cnn_test_mdae_mean"], R["cnn_test_mdae_std"])],
        "Median-baseline MAE": R["baseline_median_mae"].map("{:.2f}".format),
        "Median-baseline MdAE": R["baseline_median_mdae"].map("{:.2f}".format),
        "Mean-baseline MAE": R["baseline_mean_mae"].map("{:.2f}".format),
        "SBERT-Ridge MAE": R["ridge_mae"].map("{:.2f}".format),
        "SBERT-Ridge MdAE": R["ridge_mdae"].map("{:.2f}".format),
    })
    per_pair.to_csv(tdir / "per_pair_results.csv", index=False)
    (tdir / "per_pair_results.md").write_text(to_markdown(per_pair), encoding="utf-8")
    compact = per_pair[["Pair", "Source (train)", "Target (test)", "n dev", "n test", "Selected config", "Epochs",
                        "5-fold val MAE", "5-fold val MdAE", "Fwd-chain val MAE", "Test MAE", "Test MdAE",
                        "Median-baseline MAE", "Median-baseline MdAE", "SBERT-Ridge MAE"]].copy()
    compact.columns = ["#", "Source", "Target", "n dev", "n test", "Config", "Ep.", "Val MAE", "Val MdAE",
                       "Fwd-chain val MAE", "Test MAE", "Test MdAE", "Median MAE", "Median MdAE", "Ridge MAE"]
    compact.to_csv(tdir / "per_pair_compact.csv", index=False)
    simple = pd.DataFrame({
        "#": R["pair_id"],
        "Train project → Test project": R["source"] + " → " + R["target"],
        "Train issues": R["n_dev"],
        "Test issues": R["n_test"],
        "5-fold val. MAE": [fmt_pm(c["val_mae_mean"], c["val_mae_std"]) for c in R["cv_primary"]],
        "Test MAE": R["cnn_test_mae_mean"].map("{:.2f}".format),
        "Test MdAE": R["cnn_test_mdae_mean"].map("{:.2f}".format),
        "Median baseline MAE": R["baseline_median_mae"].map("{:.2f}".format),
    })
    simple.to_csv(tdir / "per_pair_simple.csv", index=False)

    # ------------------------------------------------------------------ fold-level validation table
    prim = cv[cv["scheme"] == cfg["split"]["cv_scheme"]]
    sel = R.set_index("pair_id")["selected_config"]
    prim_sel = prim[prim.apply(lambda r: sel.get(r["pair_id"]) == r["config"], axis=1)]
    fold_tab = prim_sel.pivot_table(index="pair_id", columns="fold", values="val_mae").round(2)
    fold_tab.columns = [f"Fold {c} MAE" for c in fold_tab.columns]
    fold_tab["Mean"] = fold_tab.mean(axis=1).round(2)
    fold_tab["Std"] = prim_sel.groupby("pair_id")["val_mae"].std(ddof=0).round(2)
    fold_tab = fold_tab.reset_index().rename(columns={"pair_id": "Pair"})
    fold_fmt = fold_tab.copy()
    for c in fold_fmt.columns[1:]:
        fold_fmt[c] = fold_fmt[c].map("{:.2f}".format)
    fold_fmt.to_csv(tdir / "validation_folds_selected_config.csv", index=False)
    (tdir / "validation_folds_selected_config.md").write_text(to_markdown(fold_fmt), encoding="utf-8")

    # ------------------------------------------------------------------ headline table (guideline 3.4)
    pooled_true = pred["y_true"].to_numpy()

    def pooled(col):
        return mae(pooled_true, pred[col]), mdae(pooled_true, pred[col])

    seed_pooled = np.array([pooled(f"cnn_seed{s}") for s in seeds])
    val_mae_all = np.array([c["val_mae_mean"] for c in R["cv_primary"]])
    val_mdae_all = np.array([c["val_mdae_mean"] for c in R["cv_primary"]])
    headline = pd.DataFrame([
        {
            "Representation": "SBERT sentence embeddings",
            "Model configuration": f"{cfg['sbert']['model_name'].split('/')[-1]} (frozen, 768-d) per sentence, title + ≤{cfg['text']['max_sentences']-1} description sentences -> 3-layer 1-D CNN; L1 loss, AdamW; lr/dropout tuned per pair by 5-fold chronological CV",
            "5-fold validation MAE (mean ± std over pairs)": fmt_pm(val_mae_all.mean(), val_mae_all.std()),
            "5-fold validation MdAE (mean ± std over pairs)": fmt_pm(val_mdae_all.mean(), val_mdae_all.std()),
            "Test MAE (mean over 39 pairs)": f"{R['cnn_test_mae_mean'].mean():.2f}",
            "Test MdAE (mean over 39 pairs)": f"{R['cnn_test_mdae_mean'].mean():.2f}",
            "Test MAE (pooled issues)": fmt_pm(seed_pooled[:, 0].mean(), seed_pooled[:, 0].std()),
            "Test MdAE (pooled issues)": fmt_pm(seed_pooled[:, 1].mean(), seed_pooled[:, 1].std()),
        },
        {
            "Representation": "SBERT whole-issue vector (reference)",
            "Model configuration": "same SBERT model, one vector per issue -> ridge regression (alpha by the same CV)",
            "5-fold validation MAE (mean ± std over pairs)": fmt_pm(np.mean([d["cv_mae_mean"] for d in R["ridge_info"]]), np.std([d["cv_mae_mean"] for d in R["ridge_info"]])),
            "5-fold validation MdAE (mean ± std over pairs)": "-",
            "Test MAE (mean over 39 pairs)": f"{R['ridge_mae'].mean():.2f}",
            "Test MdAE (mean over 39 pairs)": f"{R['ridge_mdae'].mean():.2f}",
            "Test MAE (pooled issues)": f"{pooled('sbert_ridge')[0]:.2f}",
            "Test MdAE (pooled issues)": f"{pooled('sbert_ridge')[1]:.2f}",
        },
        {
            "Representation": "No text (baseline)",
            "Model configuration": "Median story point of the source project",
            "5-fold validation MAE (mean ± std over pairs)": "-", "5-fold validation MdAE (mean ± std over pairs)": "-",
            "Test MAE (mean over 39 pairs)": f"{R['baseline_median_mae'].mean():.2f}",
            "Test MdAE (mean over 39 pairs)": f"{R['baseline_median_mdae'].mean():.2f}",
            "Test MAE (pooled issues)": f"{pooled('baseline_median')[0]:.2f}",
            "Test MdAE (pooled issues)": f"{pooled('baseline_median')[1]:.2f}",
        },
        {
            "Representation": "No text (baseline)",
            "Model configuration": "Mean story point of the source project",
            "5-fold validation MAE (mean ± std over pairs)": "-", "5-fold validation MdAE (mean ± std over pairs)": "-",
            "Test MAE (mean over 39 pairs)": f"{R['baseline_mean_mae'].mean():.2f}",
            "Test MdAE (mean over 39 pairs)": f"{R['baseline_mean_mdae'].mean():.2f}",
            "Test MAE (pooled issues)": f"{pooled('baseline_mean')[0]:.2f}",
            "Test MdAE (pooled issues)": f"{pooled('baseline_mean')[1]:.2f}",
        },
    ])
    headline.to_csv(tdir / "headline_table.csv", index=False)
    (tdir / "headline_table.md").write_text(to_markdown(headline), encoding="utf-8")

    # ------------------------------------------------------------------ statistics across pairs
    def compare(other_mae, other_mdae, name):
        out = {"vs": name}
        for metric, a, b in [("MAE", R["cnn_test_mae_mean"], other_mae), ("MdAE", R["cnn_test_mdae_mean"], other_mdae)]:
            a, b = np.asarray(a), np.asarray(b)
            diff = a - b
            try:
                p = float(wilcoxon(a, b, zero_method="zsplit").pvalue) if np.any(diff != 0) else 1.0
            except ValueError:
                p = 1.0
            A = a12(a, b)
            out[metric] = {"cnn_wins": int((a < b).sum()), "ties": int((a == b).sum()), "cnn_losses": int((a > b).sum()),
                           "mean_diff": float(diff.mean()), "wilcoxon_p": p, "A12": A, "A12_magnitude": a12_label(A)}
        return out

    stats = [
        compare(R["baseline_median_mae"], R["baseline_median_mdae"], "Median baseline"),
        compare(R["baseline_mean_mae"], R["baseline_mean_mdae"], "Mean baseline"),
        compare(R["ridge_mae"], R["ridge_mdae"], "SBERT whole-issue vector + ridge"),
    ]
    stat_rows = []
    for s in stats:
        for metric in ["MAE", "MdAE"]:
            d = s[metric]
            stat_rows.append({"SBERT-CNN vs": s["vs"], "Metric": metric, "CNN better / tie / worse (pairs)": f"{d['cnn_wins']} / {d['ties']} / {d['cnn_losses']}",
                              "Mean difference (CNN - other)": f"{d['mean_diff']:+.2f}", "Wilcoxon p": f"{d['wilcoxon_p']:.3g}",
                              "A12 (CNN error > other)": f"{d['A12']:.2f} ({d['A12_magnitude']})"})
    stat_tab = pd.DataFrame(stat_rows)
    stat_tab.to_csv(tdir / "statistical_tests.csv", index=False)
    (tdir / "statistical_tests.md").write_text(to_markdown(stat_tab), encoding="utf-8")

    # median baseline on the SAME source validation folds -> does the CNN beat it inside the source project?
    issues = pd.read_parquet(resolve(cfg["paths"]["issues"]))
    med_val = []
    for r in R.itertuples():
        _, test = target_dev_test_split(issues[issues.project_slug == r.target_slug], cfg["split"]["test_frac"])
        dev = source_development_set(issues[issues.project_slug == r.source_slug], test["created"].min(), strict=r.strict_chronology)
        y = dev["story_point"].to_numpy()
        med_val.append(float(np.mean([mae(y[va], np.full(len(va), np.median(y[tr])))
                                      for _, tr, va in make_folds(len(y), cfg["split"]["n_folds"], cfg["split"]["cv_scheme"])])))
    R["median_baseline_val_mae"] = med_val
    ok = (R["median_baseline_val_mae"] > 0) & (R["baseline_median_mae"] > 0)  # avoid 0/0 on constant tiny sets
    val_gain_rel = ((R["median_baseline_val_mae"] - val_mae_all) / R["median_baseline_val_mae"])[ok]
    test_gain_rel = ((R["baseline_median_mae"] - R["cnn_test_mae_mean"]) / R["baseline_median_mae"])[ok]

    # semantic closeness of source and target: cosine similarity of the mean SBERT whole-issue vectors
    emb_dir = resolve(cfg["paths"]["embeddings_dir"])
    centroid = {}
    for slug in set(R["source_slug"]) | set(R["target_slug"]):
        v = np.load(emb_dir / f"{slug}.npz")["doc"].astype(np.float32)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        c = v.mean(0)
        centroid[slug] = c / np.linalg.norm(c)
    R["semantic_similarity"] = [float(centroid[s] @ centroid[t]) for s, t in zip(R["source_slug"], R["target_slug"])]
    R["gain_vs_median"] = R["baseline_median_mae"] - R["cnn_test_mae_mean"]
    pmap = pd.read_csv(ROOT / "configs" / "project_map.csv").set_index("project_key")["repository"]
    R["same_repository"] = [pmap[s] == pmap[t] for s, t in zip(R["source_slug"], R["target_slug"])]
    corr_sim_gain = float(np.corrcoef(R["semantic_similarity"], R["gain_vs_median"])[0, 1])

    # ranking ability: does the predicted value order the target issues by effort? (scale-free)
    rho, rho_p, rho_r = [], [], []
    for pid in R["pair_id"]:
        g = pred[pred["pair_id"] == pid]
        r_, p_ = spearmanr(g["y_true"], g["cnn_mean_of_seeds"])
        rho.append(float(r_) if np.isfinite(r_) else 0.0)
        rho_p.append(float(p_) if np.isfinite(p_) else 1.0)
        rr_, _ = spearmanr(g["y_true"], g["sbert_ridge"])
        rho_r.append(float(rr_) if np.isfinite(rr_) else 0.0)
    R["spearman_cnn"], R["spearman_cnn_p"], R["spearman_ridge"] = rho, rho_p, rho_r
    # how much do predictions vary within a target test set? (std ~ 0 -> the CNN collapsed to a constant)
    R["cnn_pred_std"] = [float(pred.loc[pred["pair_id"] == pid, "cnn_mean_of_seeds"].std()) for pid in R["pair_id"]]
    R["true_sp_std"] = [float(pred.loc[pred["pair_id"] == pid, "y_true"].std()) for pid in R["pair_id"]]
    sim_tab = R[["pair_id", "source", "target", "same_repository", "semantic_similarity", "cnn_test_mae_mean",
                 "baseline_median_mae", "gain_vs_median", "dev_sp_median", "test_sp_median",
                 "spearman_cnn", "spearman_cnn_p", "spearman_ridge", "cnn_pred_std", "true_sp_std", "final_epochs",
                 "median_baseline_val_mae"]].round(3)
    sim_tab.to_csv(tdir / "transfer_analysis.csv", index=False)

    # generalisation gap: validation (source) vs test (target)
    gap = R["cnn_test_mae_mean"] - val_mae_all
    corr = float(np.corrcoef(val_mae_all, R["cnn_test_mae_mean"])[0, 1])
    sp_shift = (R["test_sp_median"] - R["dev_sp_median"]).abs()
    corr_shift = float(np.corrcoef(sp_shift, R["cnn_test_mae_mean"])[0, 1])

    best_row = R.loc[R["cnn_test_mae_mean"].idxmin()]
    worst_row = R.loc[R["cnn_test_mae_mean"].idxmax()]
    summary = {
        "n_pairs": int(len(R)), "n_test_issues": int(len(pred)),
        "cnn": {"val_mae_mean_over_pairs": float(val_mae_all.mean()), "val_mae_std_over_pairs": float(val_mae_all.std()),
                "val_mdae_mean_over_pairs": float(val_mdae_all.mean()), "val_mdae_std_over_pairs": float(val_mdae_all.std()),
                "test_mae_mean_over_pairs": float(R["cnn_test_mae_mean"].mean()), "test_mdae_mean_over_pairs": float(R["cnn_test_mdae_mean"].mean()),
                "test_mae_median_over_pairs": float(R["cnn_test_mae_mean"].median()),
                "pooled_test_mae": float(seed_pooled[:, 0].mean()), "pooled_test_mdae": float(seed_pooled[:, 1].mean()),
                "seed_std_mean_over_pairs": float(R["cnn_test_mae_std"].mean())},
        "median_baseline": {"test_mae_mean_over_pairs": float(R["baseline_median_mae"].mean()), "test_mdae_mean_over_pairs": float(R["baseline_median_mdae"].mean()),
                            "pooled": pooled("baseline_median")},
        "mean_baseline": {"test_mae_mean_over_pairs": float(R["baseline_mean_mae"].mean()), "test_mdae_mean_over_pairs": float(R["baseline_mean_mdae"].mean()),
                          "pooled": pooled("baseline_mean")},
        "ridge": {"test_mae_mean_over_pairs": float(R["ridge_mae"].mean()), "test_mdae_mean_over_pairs": float(R["ridge_mdae"].mean()),
                  "pooled": pooled("sbert_ridge")},
        "statistics": stats,
        "generalisation_gap_mean": float(gap.mean()), "corr_val_test_mae": corr,
        "corr_abs_median_sp_shift_vs_test_mae": corr_shift,
        "corr_semantic_similarity_vs_gain": corr_sim_gain,
        "same_repository_pairs": int(R["same_repository"].sum()),
        "gain_same_repository_mean": float(R.loc[R["same_repository"], "gain_vs_median"].mean()),
        "gain_cross_repository_mean": float(R.loc[~R["same_repository"], "gain_vs_median"].mean()),
        "pairs_cnn_beats_median_mae": int((R["gain_vs_median"] > 0).sum()),
        "median_baseline_val_mae_mean_over_pairs": float(R["median_baseline_val_mae"].mean()),
        "val_rel_gain_vs_median_mean": float(val_gain_rel.mean()),
        "val_rel_gain_vs_median_median": float(val_gain_rel.median()),
        "val_pairs_cnn_beats_median": int((val_gain_rel > 0).sum()),
        "test_rel_gain_vs_median_mean": float(test_gain_rel.mean()),
        "test_rel_gain_vs_median_median": float(test_gain_rel.median()),
        "val_test_wilcoxon_p_rel_gain": float(wilcoxon(val_gain_rel, test_gain_rel).pvalue),
        "spearman_cnn_median_over_pairs": float(np.median(rho)),
        "spearman_cnn_mean_over_pairs": float(np.mean(rho)),
        "spearman_cnn_positive_significant_pairs": int(sum(r_ > 0 and p_ < 0.05 for r_, p_ in zip(rho, rho_p))),
        "spearman_ridge_median_over_pairs": float(np.median(rho_r)),
        "spearman_cnn_pooled": float(spearmanr(pred["y_true"], pred["cnn_mean_of_seeds"])[0]),
        "pairs_near_constant_cnn": int((R["cnn_pred_std"] < 0.25).sum()),
        "pairs_abs_gain_below_0_02": int((R["gain_vs_median"].abs() < 0.02).sum()),
        "median_final_epochs": float(R["final_epochs"].median()),
        "pairs_final_epochs_le_2": int((R["final_epochs"] <= 2).sum()),
        "cnn_pred_std_median": float(R["cnn_pred_std"].median()),
        "true_sp_std_median": float(R["true_sp_std"].median()),
        "best_pair": {k: best_row[k] for k in ["pair_id", "source", "target", "cnn_test_mae_mean", "cnn_test_mdae_mean"]},
        "worst_pair": {k: worst_row[k] for k in ["pair_id", "source", "target", "cnn_test_mae_mean", "cnn_test_mdae_mean"]},
        "selected_configs": R["selected_hp"].map(hp_str).value_counts().to_dict(),
        "secondary_cv_mae_mean_over_pairs": float(np.mean([c["val_mae_mean"] for c in R["cv_secondary"] if c])),
    }

    # ------------------------------------------------------------------ figures
    order = R.sort_values("cnn_test_mae_mean").reset_index(drop=True)
    labels = [f"{r.pair_id:>2}  {r.source} → {r.target}" for r in order.itertuples()]
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.6, 9.2))
    for x0, x1, yy in zip(np.minimum.reduce([order.cnn_test_mae_mean, order.baseline_median_mae, order.ridge_mae]),
                          np.maximum.reduce([order.cnn_test_mae_mean, order.baseline_median_mae, order.ridge_mae]), y):
        ax.plot([x0, x1], [yy, yy], color=GRID, lw=1.5, zorder=1)
    ax.scatter(order.baseline_median_mae, y, s=26, color=C_MED, label="Median baseline", zorder=3, edgecolor=SURFACE, linewidth=1)
    ax.scatter(order.ridge_mae, y, s=26, color=C_RIDGE, label="SBERT whole-issue + ridge", zorder=3, edgecolor=SURFACE, linewidth=1)
    ax.scatter(order.cnn_test_mae_mean, y, s=34, color=C_CNN, label="SBERT sentences + CNN (mean of 3 seeds)", zorder=4, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Test MAE on the target project's latest 20 % of issues (story points)")
    ax.set_title("Held-out test MAE per cross-project pair", pad=24)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=7.5, handletextpad=0.3, columnspacing=1.2,
              borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(fdir / "test_mae_per_pair.png", dpi=200)
    plt.close(fig)

    # simple version for the report: CNN vs median baseline only
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    for x0, x1, yy in zip(np.minimum(order.cnn_test_mae_mean, order.baseline_median_mae),
                          np.maximum(order.cnn_test_mae_mean, order.baseline_median_mae), y):
        ax.plot([x0, x1], [yy, yy], color=GRID, lw=1.5, zorder=1)
    ax.scatter(order.baseline_median_mae, y, s=26, color=C_MED, label="Median baseline (no text)", zorder=3,
               edgecolor=SURFACE, linewidth=1)
    ax.scatter(order.cnn_test_mae_mean, y, s=34, color=C_CNN, label="SBERT + CNN (our model)", zorder=4,
               edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Test MAE in story points (lower is better)")
    ax.set_title("Test MAE for each of the 39 train → test project pairs", pad=24)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(fdir / "test_mae_per_pair_simple.png", dpi=200)
    plt.close(fig)

    # source (validation) vs target (test): CNN vs median baseline
    med_val_mean = float(R["median_baseline_val_mae"].mean())
    groups = ["Validation\n(same project as training)", "Test\n(a different, later project)"]
    cnn_v = [float(val_mae_all.mean()), float(R["cnn_test_mae_mean"].mean())]
    med_v = [med_val_mean, float(R["baseline_median_mae"].mean())]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    xg = np.arange(2)
    w = 0.34
    b1 = ax.bar(xg - w / 2 - 0.01, cnn_v, w, color=C_CNN, label="SBERT + CNN")
    b2 = ax.bar(xg + w / 2 + 0.01, med_v, w, color=C_MED, label="Median baseline")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(xg, groups)
    ax.set_ylabel("MAE (story points), mean of 39 pairs")
    ax.set_ylim(0, max(cnn_v + med_v) * 1.2)
    ax.set_title("The model beats the baseline on its own project, but not on a new one")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fdir / "source_vs_target.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    lim = max(val_mae_all.max(), R["cnn_test_mae_mean"].max()) * 1.08
    ax.plot([0, lim], [0, lim], color=AXIS, lw=1, ls="--", zorder=1)
    ax.scatter(val_mae_all, R["cnn_test_mae_mean"], s=30, color=C_CNN, zorder=3, edgecolor=SURFACE, linewidth=1)
    for r, vx in zip(R.itertuples(), val_mae_all):
        if r.cnn_test_mae_mean > 4.5 or vx > 4.5:
            ax.annotate(f"{r.source}→{r.target}", (vx, r.cnn_test_mae_mean), fontsize=6.5, color=INK2, xytext=(4, 2), textcoords="offset points")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("5-fold validation MAE on the source project")
    ax.set_ylabel("Test MAE on the target project")
    ax.set_title(f"Source validation vs target test error (r = {corr:.2f})")
    ax.text(lim * 0.97, lim * 0.06, "below the line: target easier than source", ha="right", fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(fdir / "validation_vs_test.png", dpi=200)
    plt.close(fig)

    # absolute error distribution, pooled over all test issues
    errs = {
        "SBERT + CNN": np.abs(pred["y_true"] - pred["cnn_mean_of_seeds"]),
        "SBERT + ridge": np.abs(pred["y_true"] - pred["sbert_ridge"]),
        "Median baseline": np.abs(pred["y_true"] - pred["baseline_median"]),
    }
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    bp = ax.boxplot(list(errs.values()), orientation="horizontal", showfliers=False, widths=0.5, patch_artist=True,
                    medianprops={"color": INK, "lw": 1.5}, whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    for patch, c in zip(bp["boxes"], [C_CNN, C_RIDGE, C_MED]):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
        patch.set_edgecolor(SURFACE)
    ax.set_yticks([1, 2, 3], [f"{k}\nMAE {e.mean():.2f} · MdAE {np.median(e):.2f}" for k, e in errs.items()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Absolute error (story points), all test issues pooled; whiskers = 1.5 IQR")
    ax.set_title("Distribution of absolute errors")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(fdir / "abs_error_distribution.png", dpi=200)
    plt.close(fig)

    # fold-wise validation MAE of the selected configuration
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    fold_cols = [c for c in fold_tab.columns if c.startswith("Fold")]
    for j, row in fold_tab.iterrows():
        vals = row[fold_cols].to_numpy(float)
        ax.plot([j] * len(vals), vals, color=GRID, lw=1.2, zorder=1)
        ax.scatter([j] * len(vals), vals, s=10, color="#86b6ef", zorder=2)
        ax.scatter([j], [vals.mean()], s=26, color=C_CNN, zorder=3, edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(range(len(fold_tab)), fold_tab["Pair"].astype(str), fontsize=7)
    ax.set_xlabel("Pair id (source project = development data)")
    ax.set_ylabel("Validation MAE")
    ax.set_title("Five-fold chronological validation: each fold (light) and the mean (dark)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(fdir / "validation_folds.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.axhline(0, color=AXIS, lw=1, ls="--", zorder=1)
    same = R["same_repository"].to_numpy()
    ax.scatter(R.loc[~same, "semantic_similarity"], R.loc[~same, "gain_vs_median"], s=30, color=C_CNN, zorder=3,
               edgecolor=SURFACE, linewidth=1, label="different repository")
    ax.scatter(R.loc[same, "semantic_similarity"], R.loc[same, "gain_vs_median"], s=34, color=C_MED, marker="D", zorder=3,
               edgecolor=SURFACE, linewidth=1, label="same repository")
    ax.set_xlabel("Source–target semantic similarity (cosine of mean SBERT issue vectors)")
    ax.set_ylabel("Median-baseline MAE − CNN MAE  (> 0: CNN better)")
    ax.set_title(f"Does SBERT transfer better between similar projects? (r = {corr_sim_gain:.2f})")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fdir / "similarity_vs_gain.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------ optional comparison with a second run
    if args.compare:
        other = pd.DataFrame(json.loads((resolve(cfg["paths"]["results_dir"]) / args.compare / "pair_results.json").read_text()))
        m = R.merge(other, on="pair_id", suffixes=("", "_cmp"))
        comp = pd.DataFrame({
            "Pair": m["pair_id"], "Source": m["source"], "Target": m["target"],
            "n dev (main)": m["n_dev"], f"n dev ({args.compare})": m["n_dev_cmp"],
            "Test MAE (main)": m["cnn_test_mae_mean"].round(2), f"Test MAE ({args.compare})": m["cnn_test_mae_mean_cmp"].round(2),
            "Test MdAE (main)": m["cnn_test_mdae_mean"].round(2), f"Test MdAE ({args.compare})": m["cnn_test_mdae_mean_cmp"].round(2),
            f"Median-baseline MAE ({args.compare})": m["baseline_median_mae_cmp"].round(2),
        })
        comp.to_csv(tdir / f"compare_{args.compare}.csv", index=False)
        (tdir / f"compare_{args.compare}.md").write_text(to_markdown(comp), encoding="utf-8")
        a, b = m["cnn_test_mae_mean"].to_numpy(), m["cnn_test_mae_mean_cmp"].to_numpy()
        summary[f"compare_{args.compare}"] = {
            "n_common_pairs": int(len(m)),
            "main_test_mae_mean": float(a.mean()), f"{args.compare}_test_mae_mean": float(b.mean()),
            "main_test_mdae_mean": float(m["cnn_test_mdae_mean"].mean()), f"{args.compare}_test_mdae_mean": float(m["cnn_test_mdae_mean_cmp"].mean()),
            f"{args.compare}_median_baseline_mae_mean": float(m["baseline_median_mae_cmp"].mean()),
            "pairs_where_compare_better": int((b < a).sum()),
            "wilcoxon_p": float(wilcoxon(a, b, zero_method="zsplit").pvalue) if np.any(a != b) else 1.0,
        }
        fig, ax = plt.subplots(figsize=(4.6, 4.3))
        lim = max(a.max(), b.max()) * 1.08
        ax.plot([0, lim], [0, lim], color=AXIS, lw=1, ls="--")
        ax.scatter(a, b, s=30, color=C_CNN, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Test MAE, within-project chronology (main)")
        ax.set_ylabel(f"Test MAE, {args.compare} chronology")
        ax.set_title(f"Effect of strict cross-project chronology ({len(m)} pairs)")
        fig.tight_layout()
        fig.savefig(fdir / f"compare_{args.compare}.png", dpi=200)
        plt.close(fig)

    dump_json(summary, base / "summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "statistics"}, indent=2, default=str))
    print(stat_tab.to_string(index=False))


if __name__ == "__main__":
    main()
