"""Stage 3 - build the feature panels the classifiers are benchmarked on.

Applies the anchor method (see saygex/selectors.py) once to the whole target
pair, producing the descriptive panels and gene lists.

IMPORTANT: these panels are fit on all samples, so they are for description,
plots, and gene-list inspection -- not for measuring accuracy. Stage 4 refits
the same selection inside each cross-validation fold for that.

Outputs (results/csvs, results/json):
    top<N>_raw_expression_TumorLabels.csv / ..._zscored_...
    clock_expression_raw_TumorLabels.csv  / ..._zscored_...
    anchor_panel_<K>genes_raw.csv / ..._zscored.csv     the 100+11 panel
    WeightedAnchorGenes.csv         per-clock-gene weight and rank
    AnchorCorrelationScores.csv     weighted |corr| score per candidate gene
    stage3_summary.json
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from saygex import config, io
from saygex.selectors import build_anchor_panel, gene_indices, rank_by_pvalue
from saygex.stats import benjamini_hochberg, neg_log10, standardized_softmax, welch_ttest


def main():
    config.ensure_dirs()
    path = config.pair_matrix_path()
    print(f"Building panels from {os.path.basename(path)}", flush=True)

    df_raw, y_str = io.load_labelled(path)
    genes = df_raw.columns.tolist()

    types = y_str.unique()
    if len(types) != 2:
        print(f"Expected 2 tumor types, found {list(types)}")
        return
    t1, t2 = types
    y = (y_str == t2).astype(int).to_numpy()  # label 1 = second class

    X = df_raw.to_numpy()

    # Z-scored view, fit on all samples. Used for the descriptive panels only.
    df_z = pd.DataFrame(StandardScaler().fit_transform(X),
                        index=df_raw.index, columns=genes)

    _, p_vals = welch_ttest(X[y == 0], X[y == 1])
    q_vals = benjamini_hochberg(p_vals)
    scores = neg_log10(p_vals)

    screened = rank_by_pvalue(p_vals)[: config.TOP_N_GENES]
    top_genes = [genes[i] for i in screened]
    print(f"  screened {len(genes)} genes -> top {len(top_genes)} by Welch p-value")

    anchor_idx = np.array(gene_indices(genes, config.CLOCK_GENES))
    anchor_names = [genes[i] for i in anchor_idx]
    missing = [g for g in config.CLOCK_GENES if g not in set(anchor_names)]
    if missing:
        print(f"  ! clock genes not in matrix: {missing}")
    if len(anchor_idx) == 0:
        print("No anchor genes found; cannot continue.")
        return

    # Per-anchor statistics and softmax weights.
    weights = standardized_softmax(scores[anchor_idx])
    anchor_table = pd.DataFrame({
        "Gene": anchor_names,
        "p_val": p_vals[anchor_idx],
        "q_val": q_vals[anchor_idx],
        "Score": scores[anchor_idx],
        "Weight": weights,
    }).sort_values("p_val")
    anchor_table.to_csv(config.csv_path("WeightedAnchorGenes.csv"), index=False)

    panel_idx = build_anchor_panel(
        X, p_vals, screened, anchor_idx,
        n_final=config.FINAL_N_CANDIDATES, n_drop=config.N_ANCHORS_TO_DROP,
    )
    panel_genes = [genes[i] for i in panel_idx]

    ranked_anchors = anchor_idx[rank_by_pvalue(p_vals[anchor_idx])]
    kept_anchors = [genes[i] for i in ranked_anchors[: len(ranked_anchors) - config.N_ANCHORS_TO_DROP]]
    dropped_anchors = [genes[i] for i in ranked_anchors[len(kept_anchors):]]
    print(f"  dropped {len(dropped_anchors)} weakest anchors {dropped_anchors}; "
          f"kept {len(kept_anchors)}")
    print(f"  final panel: {len(panel_genes)} genes "
          f"({config.FINAL_N_CANDIDATES} correlated + {len(kept_anchors)} anchors)")

    # Descriptive panels.
    io.save_labelled(df_raw[top_genes], y_str,
                     config.csv_path(f"top{config.TOP_N_GENES}_raw_expression_TumorLabels.csv"))
    io.save_labelled(df_z[top_genes], y_str,
                     config.csv_path(f"top{config.TOP_N_GENES}_zscored_TumorLabels.csv"))
    io.save_labelled(df_raw[anchor_names], y_str,
                     config.csv_path("clock_expression_raw_TumorLabels.csv"))
    io.save_labelled(df_z[anchor_names], y_str,
                     config.csv_path("clock_expression_zscored_TumorLabels.csv"))
    io.save_labelled(df_raw[panel_genes], y_str,
                     config.csv_path(f"anchor_panel_{len(panel_genes)}genes_raw.csv"))
    io.save_labelled(df_z[panel_genes], y_str,
                     config.csv_path(f"anchor_panel_{len(panel_genes)}genes_zscored.csv"))

    # Candidate scores, so the ranking is inspectable rather than implicit.
    kept_idx = np.array(gene_indices(genes, kept_anchors))
    candidates = np.array([i for i in screened if i not in set(kept_idx.tolist())])
    block = np.corrcoef(X[:, np.concatenate([candidates, kept_idx])], rowvar=False)
    corr = np.nan_to_num(block[: candidates.size, candidates.size:], nan=0.0)
    w_map = dict(zip(anchor_idx.tolist(), weights))
    w_vec = np.array([w_map[i] for i in kept_idx.tolist()])
    pd.DataFrame({
        "Gene": [genes[i] for i in candidates],
        "WeightedAnchorCorrelation": np.abs(corr) @ w_vec,
        "p_val": p_vals[candidates],
        "InFinalPanel": [genes[i] in set(panel_genes) for i in candidates],
    }).sort_values("WeightedAnchorCorrelation", ascending=False).to_csv(
        config.csv_path("AnchorCorrelationScores.csv"), index=False)

    io.write_json({
        "target_pair": f"{t1} vs {t2}",
        "n_genes_screened": len(genes),
        "top_genes": top_genes,
        "anchors_found": anchor_names,
        "anchors_missing": missing,
        "anchors_kept": kept_anchors,
        "anchors_dropped": dropped_anchors,
        "final_panel": panel_genes,
        "final_panel_size": len(panel_genes),
    }, "stage3_summary.json")
    print("Stage 3 complete.")


if __name__ == "__main__":
    main()
