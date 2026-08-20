"""Stage 2 - which cohort pair is most separable, and which genes separate it.

Outputs (results/csvs, results/plots, results/json):
    PCA_<A>_<B>.png, Volcano_<A>_<B>.png    one per pair
    Correlation_Clock_vs_Tumor.png          clock genes vs top differential genes
    pair_separability.csv                   silhouette score per pair
    <N>MostDiffGenes_<A>_<B>.csv            the differential-expression screen
    Clock_Genes_Analysis.csv                per-clock-gene statistics
    stage2_summary.json
"""

import os

import matplotlib
matplotlib.use("Agg")  # No display when driven by pipeline.py.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from saygex import config, io
from saygex.stats import benjamini_hochberg, neg_log10, welch_ttest


def differential_expression(X, y, t1, t2):
    """Welch t-test per gene, BH q-values, and log2 fold change."""
    group1, group2 = X[y == t1], X[y == t2]
    _, p_vals = welch_ttest(group1, group2)

    # Xena HiSeqV2 values are already log2(norm_count + 1), so the difference
    # of the means *is* the log2 fold change.
    log2_fc = group1.mean() - group2.mean()

    return pd.DataFrame(
        {"p_val": p_vals, "q_val": benjamini_hochberg(p_vals), "log2_fc": log2_fc.values},
        index=X.columns,
    )


def analyse_pair(path):
    X, y = io.load_labelled(path)
    types = y.unique()
    if len(types) != 2:
        print(f"  ! {os.path.basename(path)} has {len(types)} labels, skipping.")
        return None

    t1, t2 = types
    pair_name = f"{t1} vs {t2}"

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    components = pca.fit_transform(X_scaled)
    sil = silhouette_score(components, y)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=y.to_numpy(),
                    s=80, alpha=0.8, palette="Set1")
    plt.title(f"PCA: {pair_name}\nSilhouette Score: {sil:.3f}", fontweight="bold")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.tight_layout()
    plt.savefig(config.plot_path(f"PCA_{t1}_{t2}.png"), dpi=300)
    plt.close()

    de = differential_expression(X, y, t1, t2)
    significant = (de["q_val"] < config.FDR_CUTOFF) & (de["log2_fc"].abs() > config.LOG2FC_CUTOFF)
    colors = np.where(significant & (de["log2_fc"] > 0), "indianred",
                      np.where(significant & (de["log2_fc"] < 0), "steelblue", "lightgrey"))

    passing = de.loc[de["q_val"] < config.FDR_CUTOFF, "p_val"]

    plt.figure(figsize=(7, 6))
    plt.scatter(de["log2_fc"], neg_log10(de["p_val"].values), c=colors, s=10, alpha=0.6)
    if len(passing):
        # Where BH significance actually starts -- a far stricter bar than raw
        # p < 0.05 across ~20,500 genes.
        plt.axhline(-np.log10(passing.max()), color="k", linestyle="--", lw=1,
                    label=f"FDR {config.FDR_CUTOFF} (p<={passing.max():.1e})")
        plt.legend(loc="lower right", fontsize=8)
    plt.axvline(config.LOG2FC_CUTOFF, color="k", linestyle="--", lw=1)
    plt.axvline(-config.LOG2FC_CUTOFF, color="k", linestyle="--", lw=1)
    plt.title(f"Volcano: {pair_name} ({int(significant.sum())} significant)", fontweight="bold")
    plt.xlabel(f"Log2 Fold Change ({t1} - {t2})")
    plt.ylabel("-Log10 P-Value")
    plt.tight_layout()
    plt.savefig(config.plot_path(f"Volcano_{t1}_{t2}.png"), dpi=300)
    plt.close()

    print(f"  {pair_name}: silhouette {sil:.3f}, {int(significant.sum())} genes "
          f"at FDR<{config.FDR_CUTOFF} and |log2FC|>{config.LOG2FC_CUTOFF}")
    return {"pair": pair_name, "silhouette": float(sil),
            "n_significant": int(significant.sum())}


def main():
    config.ensure_dirs()
    print("Comparing cohort pairs...", flush=True)

    results = []
    for a, b in config.PAIRS:
        path = config.pair_matrix_path((a, b))
        if not os.path.exists(path):
            print(f"  ! missing {path}; run stage 1 first.")
            continue
        outcome = analyse_pair(path)
        if outcome:
            results.append(outcome)

    if not results:
        print("No cohort pairs could be analysed.")
        return

    ranked = pd.DataFrame(results).sort_values("silhouette", ascending=False)
    ranked.to_csv(config.csv_path("pair_separability.csv"), index=False)

    target_name = " vs ".join(config.TARGET_PAIR)
    best = ranked.iloc[0]["pair"]
    if best != target_name:
        print(f"  ! most separable pair is {best}, but TARGET_PAIR is {target_name}")

    # Focused screen on the pair the modelling stages use.
    X, y = io.load_labelled(config.pair_matrix_path())
    t1, t2 = y.unique()
    de = differential_expression(X, y, t1, t2)

    top = de.sort_values("p_val").head(config.TOP_N_GENES)
    top.rename_axis("Gene").reset_index().to_csv(
        config.csv_path(f"{config.TOP_N_GENES}MostDiffGenes_{t1}_{t2}.csv"), index=False)
    print(f"Saved top {config.TOP_N_GENES} genes for {t1} vs {t2}.")

    valid_clock = [g for g in config.CLOCK_GENES if g in X.columns]
    clock_de = differential_expression(X[valid_clock], y, t1, t2)
    clock_de.rename_axis("Gene").reset_index().sort_values("p_val").to_csv(
        config.csv_path("Clock_Genes_Analysis.csv"), index=False)

    correlations = pd.concat([X[valid_clock], X[top.index]], axis=1).corr()
    plt.figure(figsize=(14, 8))
    sns.heatmap(correlations.loc[valid_clock, top.index], cmap="coolwarm",
                center=0, vmin=-1, vmax=1)
    plt.title(f"Correlation: Clock Genes vs Top {config.TOP_N_GENES} Diff Genes "
              f"({t1} vs {t2})", fontweight="bold")
    plt.xlabel(f"Top {config.TOP_N_GENES} Differentially Expressed Genes")
    plt.ylabel("Clock Genes")
    plt.tight_layout()
    plt.savefig(config.plot_path("Correlation_Clock_vs_Tumor.png"), dpi=300)
    plt.close()

    io.write_json({"pairs": results, "target_pair": target_name,
                   "most_separable": best,
                   "clock_genes_tested": valid_clock}, "stage2_summary.json")
    print("Stage 2 complete.")


if __name__ == "__main__":
    main()
