"""Stage 5 - do the clock genes actually earn their place?

Four controls, because the stage 4 table on its own cannot distinguish "the
clock is special" from "111 reasonable genes classify kidney tumors".

  E1  size-matched random panels
      Null: random gene sets of the same size. Answers "does this panel beat an
      arbitrary panel of the same size?" Only the label-blind comparisons here
      are fair tests -- see the note printed with the results.

  E2  expression-matched random panels
      As E1, but each random gene is drawn from the same mean-expression and
      variance bin as the clock gene it replaces. Around 270 genes in the
      matrix are flat zero everywhere, so an unmatched null mostly proves the
      clock beats noise. This one asks whether the clock beats genes that
      *look like* the clock.

  E3  random-anchor ablation -- THE THESIS TEST
      Runs the identical selection procedure with random anchor genes instead
      of clock genes. Same size, same steps, only the anchors change. If clock
      anchors do not beat random anchors, the method is not about the clock.

  E4  accuracy-vs-panel-size curve
      Dissolves the feature-count confound instead of controlling for it: plots
      accuracy against k for supervised top-k, random-k, and anchor-k panels.

Outputs:
    stage5_validation.json, stage5_e1_e2_nulls.csv, stage5_e3_ablation.csv,
    stage5_e4_curve.csv, plots/E1_E2_null_distributions.png,
    plots/E3_anchor_ablation.png, plots/E4_accuracy_vs_panel_size.png
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from saygex import config, evaluation, io
from saygex.selectors import (
    AnchorPanelSelector,
    FixedGeneSet,
    WelchSelectKBest,
    build_anchor_panel,
    expression_profile,
    gene_indices,
    matched_random_indices,
    rank_by_pvalue,
)
from saygex.stats import empirical_p, welch_ttest


def precompute_screens(X, y, fold_list, top_n):
    """One genome-wide Welch screen per fold, reused across every draw.

    The screen is by far the most expensive step and does not depend on the
    anchor set, so caching it turns the ablation from hours into minutes.
    """
    screens = []
    for train, test in fold_list:
        _, p_vals = welch_ttest(X[train][y[train] == 0], X[train][y[train] == 1])
        screens.append({
            "train": train, "test": test, "p_vals": p_vals,
            "screened": rank_by_pvalue(p_vals)[:top_n],
        })
    return screens


def score_anchor_set(X, y, screens, anchor_idx, model_name,
                     n_final=config.FINAL_N_CANDIDATES, n_drop=config.N_ANCHORS_TO_DROP):
    """Full anchor method for one anchor set, refit per fold."""
    per_fold = []
    for screen in screens:
        train, test = screen["train"], screen["test"]
        panel = build_anchor_panel(
            X[train], screen["p_vals"], screen["screened"], anchor_idx,
            n_final=n_final, n_drop=n_drop,
        )
        per_fold.append(evaluation.score_panel(X, y, panel, train, test, model_name))
    return float(np.mean([f["balanced_accuracy"] for f in per_fold]))


def _progress(label, i, total, started):
    if i % max(total // 10, 1) == 0 or i == total:
        rate = i / max(time.time() - started, 1e-9)
        remaining = (total - i) / rate if rate else 0
        print(f"    {label}: {i}/{total} draws  (~{remaining:.0f}s left)", flush=True)


# ------------------------------------------------------------------ E1/E2 ---

def run_e1_e2(X, y, genes, anchor_idx, fold_list, rng, n_draws):
    print("\n" + "=" * 72)
    print("E1/E2 - random panels of matched size (and matched expression)")
    print("=" * 72, flush=True)

    means, sds = expression_profile(X)
    n_genes = len(genes)
    anchor_set = set(anchor_idx)

    observed = {}
    for model in config.MODELS:
        observed[("clock only", model)] = evaluation.cross_validate_selector(
            X, y, lambda: FixedGeneSet(anchor_idx), model, fold_list
        )["balanced_accuracy"]
        observed[("anchor panel", model)] = evaluation.cross_validate_selector(
            X, y, lambda: AnchorPanelSelector(anchor_idx), model, fold_list
        )["balanced_accuracy"]
        observed[("top-200", model)] = evaluation.cross_validate_selector(
            X, y, lambda: WelchSelectKBest(config.TOP_N_GENES), model, fold_list
        )["balanced_accuracy"]

    panel_sizes = {
        "clock only": len(anchor_idx),
        "anchor panel": config.FINAL_N_CANDIDATES
                        + max(len(anchor_idx) - config.N_ANCHORS_TO_DROP, 1),
        "top-200": config.TOP_N_GENES,
    }
    # Only the clock-only row is a like-for-like test: both it and its null are
    # label-blind. The other two panels were built using the labels, so beating
    # a random panel is expected and proves only that selection works.
    fair = {"clock only": True, "anchor panel": False, "top-200": False}

    rows = []
    nulls = {}
    for panel, k in panel_sizes.items():
        for model in config.MODELS:
            started = time.time()
            draws = []
            for i in range(1, n_draws + 1):
                idx = rng.choice(n_genes, size=k, replace=False)
                draws.append(evaluation.cross_validate_panel(X, y, idx, model, fold_list)
                             ["balanced_accuracy"])
                _progress(f"E1 {panel}/{model} k={k}", i, n_draws, started)
            draws = np.array(draws)
            nulls[f"E1|{panel}|{model}"] = draws
            obs = observed[(panel, model)]
            rows.append({
                "experiment": "E1_random", "panel": panel, "model": model,
                "panel_size": k, "observed": obs,
                "null_mean": draws.mean(), "null_sd": draws.std(),
                "null_p95": np.percentile(draws, 95), "null_max": draws.max(),
                "empirical_p": empirical_p(obs, draws), "fair_test": fair[panel],
            })
            print(f"  E1 {panel:14s} {model:10s} k={k:4d}  observed {obs:.4f}  "
                  f"null {draws.mean():.4f}+/-{draws.std():.4f}  p={rows[-1]['empirical_p']:.4f}"
                  f"{'' if fair[panel] else '  (not a fair test)'}", flush=True)

    # E2: expression-matched null for the clock panel only -- it is the panel
    # whose composition the matching is defined against.
    k = len(anchor_idx)
    for model in config.MODELS:
        started = time.time()
        draws = []
        for i in range(1, n_draws + 1):
            idx = matched_random_indices(rng, means, sds, anchor_idx, exclude=anchor_set)
            draws.append(evaluation.cross_validate_panel(X, y, idx, model, fold_list)
                         ["balanced_accuracy"])
            _progress(f"E2 clock/{model}", i, n_draws, started)
        draws = np.array(draws)
        nulls[f"E2|clock only|{model}"] = draws
        obs = observed[("clock only", model)]
        rows.append({
            "experiment": "E2_expression_matched", "panel": "clock only", "model": model,
            "panel_size": k, "observed": obs,
            "null_mean": draws.mean(), "null_sd": draws.std(),
            "null_p95": np.percentile(draws, 95), "null_max": draws.max(),
            "empirical_p": empirical_p(obs, draws), "fair_test": True,
        })
        print(f"  E2 {'clock only':14s} {model:10s} k={k:4d}  observed {obs:.4f}  "
              f"null {draws.mean():.4f}+/-{draws.std():.4f}  p={rows[-1]['empirical_p']:.4f}",
              flush=True)

    return pd.DataFrame(rows), nulls, observed


# --------------------------------------------------------------------- E3 ---

def run_e3(X, y, genes, anchor_idx, fold_list, rng, n_draws):
    print("\n" + "=" * 72)
    print("E3 - random-anchor ablation (the thesis test)")
    print("=" * 72, flush=True)

    screens = precompute_screens(X, y, fold_list, config.TOP_N_GENES)
    means, sds = expression_profile(X)
    n_genes = len(genes)
    anchor_set = set(anchor_idx)
    k = len(anchor_idx)

    rows = []
    nulls = {}
    for model in config.MODELS:
        observed = score_anchor_set(X, y, screens, np.asarray(anchor_idx), model)

        for null_kind in ("random", "expression_matched"):
            started = time.time()
            draws = []
            for i in range(1, n_draws + 1):
                if null_kind == "random":
                    idx = rng.choice(n_genes, size=k, replace=False)
                else:
                    idx = matched_random_indices(rng, means, sds, anchor_idx,
                                                 exclude=anchor_set)
                draws.append(score_anchor_set(X, y, screens, idx, model))
                _progress(f"E3 {null_kind}/{model}", i, n_draws, started)
            draws = np.array(draws)
            nulls[f"E3|{null_kind}|{model}"] = draws
            rows.append({
                "experiment": f"E3_{null_kind}_anchors", "model": model,
                "n_anchors": k, "observed_clock": observed,
                "null_mean": draws.mean(), "null_sd": draws.std(),
                "null_p95": np.percentile(draws, 95), "null_max": draws.max(),
                "empirical_p": empirical_p(observed, draws),
            })
            print(f"  E3 {null_kind:20s} {model:10s}  clock anchors {observed:.4f}  "
                  f"random anchors {draws.mean():.4f}+/-{draws.std():.4f}  "
                  f"p={rows[-1]['empirical_p']:.4f}", flush=True)

    return pd.DataFrame(rows), nulls


# --------------------------------------------------------------------- E4 ---

def run_e4(X, y, genes, anchor_idx, fold_list, rng, k_values, n_draws):
    print("\n" + "=" * 72)
    print("E4 - accuracy vs panel size")
    print("=" * 72, flush=True)

    n_genes = len(genes)
    n_kept = max(len(anchor_idx) - config.N_ANCHORS_TO_DROP, 1)
    rows = []

    for model in config.MODELS:
        for k in k_values:
            if k > n_genes:
                continue

            top = evaluation.cross_validate_selector(
                X, y, lambda kk=k: WelchSelectKBest(kk), model, fold_list
            )["balanced_accuracy"]

            draws = np.array([
                evaluation.cross_validate_panel(
                    X, y, rng.choice(n_genes, size=k, replace=False), model, fold_list
                )["balanced_accuracy"]
                for _ in range(n_draws)
            ])

            # The anchor method generalises as "screen 2*n_final, keep n_final,
            # add the kept anchors" -- exactly what it does at k=111.
            anchor_score = np.nan
            if k > n_kept:
                n_final = k - n_kept
                anchor_score = evaluation.cross_validate_selector(
                    X, y,
                    lambda nf=n_final: AnchorPanelSelector(
                        anchor_idx, top_n=min(2 * nf, n_genes), n_final=nf
                    ),
                    model, fold_list,
                )["balanced_accuracy"]

            rows.append({
                "model": model, "k": k, "top_k_welch": top,
                "random_mean": draws.mean(), "random_sd": draws.std(),
                "random_p05": np.percentile(draws, 5),
                "random_p95": np.percentile(draws, 95),
                "anchor_panel": anchor_score,
            })
            anchor_txt = f"{anchor_score:.4f}" if np.isfinite(anchor_score) else "n/a"
            print(f"  E4 {model:10s} k={k:6d}  top-k {top:.4f}  "
                  f"random {draws.mean():.4f}+/-{draws.std():.4f}  "
                  f"anchor {anchor_txt}", flush=True)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plots ---

def plot_nulls(nulls, observed, path):
    keys = [k for k in nulls if k.startswith(("E1|clock only", "E2|clock only"))]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        exp, panel, model = key.split("|")
        draws = nulls[key]
        obs = observed[(panel, model)]
        ax.hist(draws, bins=30, color="lightsteelblue", edgecolor="white")
        ax.axvline(obs, color="indianred", lw=2,
                   label=f"clock genes ({obs:.3f})")
        ax.set_title(f"{exp}: {model}", fontweight="bold", fontsize=10)
        ax.set_xlabel("balanced accuracy")
        ax.set_ylabel("random panels")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_ablation(nulls, e3, path):
    keys = sorted(nulls)
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        _, kind, model = key.split("|")
        draws = nulls[key]
        row = e3[(e3["model"] == model) & (e3["experiment"] == f"E3_{kind}_anchors")]
        obs = float(row["observed_clock"].iloc[0])
        ax.hist(draws, bins=30, color="lightsteelblue", edgecolor="white")
        ax.axvline(obs, color="indianred", lw=2, label=f"clock anchors ({obs:.3f})")
        ax.set_title(f"E3 {kind}: {model}", fontweight="bold", fontsize=10)
        ax.set_xlabel("balanced accuracy")
        ax.set_ylabel("random anchor sets")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_curve(e4, anchor_size, path):
    models = e4["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 4.5), squeeze=False)
    for ax, model in zip(axes[0], models):
        sub = e4[e4["model"] == model].sort_values("k")
        ax.fill_between(sub["k"], sub["random_p05"], sub["random_p95"],
                        color="lightgrey", label="random panels (5-95%)")
        ax.plot(sub["k"], sub["random_mean"], color="grey", lw=1.2, ls=":", label="random mean")
        ax.plot(sub["k"], sub["top_k_welch"], marker="o", ms=4, color="steelblue",
                label="top-k by Welch (supervised)")
        anchors = sub.dropna(subset=["anchor_panel"])
        ax.plot(anchors["k"], anchors["anchor_panel"], marker="s", ms=4,
                color="indianred", label="anchor panel")
        ax.axvline(anchor_size, color="black", ls="--", lw=0.8,
                   label=f"published panel (k={anchor_size})")
        ax.set_xscale("log")
        ax.set_xlabel("panel size (genes, log scale)")
        ax.set_ylabel("balanced accuracy")
        ax.set_title(model, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ------------------------------------------------------------------- main ---

def main():
    config.ensure_dirs()
    X_df, y_str = io.load_labelled(config.pair_matrix_path())
    genes = X_df.columns.tolist()
    X = X_df.to_numpy()
    classes = sorted(y_str.unique())
    y = (y_str == classes[1]).astype(int).to_numpy()

    anchor_idx = gene_indices(genes, config.CLOCK_GENES)
    fold_list = evaluation.folds(X, y)
    rng = np.random.default_rng(config.RANDOM_STATE)
    n_draws = config.N_NULL_DRAWS

    print(f"Validation on {X.shape[0]} samples x {X.shape[1]} genes, "
          f"{len(anchor_idx)} clock anchors, {n_draws} draws per null.")
    print(f"One minority sample is worth {evaluation.one_sample_worth(y):.4f} "
          "balanced accuracy.", flush=True)

    started = time.time()
    e1e2, nulls_12, observed = run_e1_e2(X, y, genes, anchor_idx, fold_list, rng, n_draws)
    e3, nulls_3 = run_e3(X, y, genes, anchor_idx, fold_list, rng, n_draws)
    e4 = run_e4(X, y, genes, anchor_idx, fold_list, rng,
                config.CURVE_K_VALUES, config.N_CURVE_DRAWS)

    e1e2.to_csv(config.csv_path("stage5_e1_e2_nulls.csv"), index=False)
    e3.to_csv(config.csv_path("stage5_e3_ablation.csv"), index=False)
    e4.to_csv(config.csv_path("stage5_e4_curve.csv"), index=False)

    anchor_size = config.FINAL_N_CANDIDATES + max(
        len(anchor_idx) - config.N_ANCHORS_TO_DROP, 1)
    plot_nulls(nulls_12, observed, config.plot_path("E1_E2_null_distributions.png"))
    plot_ablation(nulls_3, e3, config.plot_path("E3_anchor_ablation.png"))
    plot_curve(e4, anchor_size, config.plot_path("E4_accuracy_vs_panel_size.png"))

    io.write_json({
        "n_draws": n_draws,
        "n_curve_draws": config.N_CURVE_DRAWS,
        "anchor_genes": [genes[i] for i in anchor_idx],
        "one_sample_worth": evaluation.one_sample_worth(y),
        "e1_e2": e1e2.to_dict(orient="records"),
        "e3": e3.to_dict(orient="records"),
        "e4": e4.to_dict(orient="records"),
        "runtime_seconds": time.time() - started,
    }, "stage5_validation.json")

    print(f"\nStage 5 complete in {time.time() - started:.0f}s.")
    _verdict(e1e2, e3)


def _verdict(e1e2, e3):
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    for _, row in e1e2[e1e2["fair_test"]].iterrows():
        better = "beats" if row["empirical_p"] < 0.05 else "does NOT beat"
        print(f"  {row['experiment']:22s} {row['model']:10s} clock panel {better} "
              f"its null (p={row['empirical_p']:.4f})")
    for _, row in e3.iterrows():
        better = "beats" if row["empirical_p"] < 0.05 else "does NOT beat"
        print(f"  {row['experiment']:22s} {row['model']:10s} clock anchors {better} "
              f"random anchors (p={row['empirical_p']:.4f})")


if __name__ == "__main__":
    main()
