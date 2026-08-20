"""Stage 4 - benchmark the panels, leak-free.

Every panel is rebuilt inside each cross-validation fold. The saved panels from
stage 3 were fit on all 600 samples, so scoring those directly would measure a
model that has already seen its own test set; that biased number is reported
alongside for comparison, clearly labelled.

Outputs:
    stage4_results.csv / stage4_results.json
"""

import numpy as np
import pandas as pd

from saygex import config, evaluation, io
from saygex.selectors import AnchorPanelSelector, FixedGeneSet, WelchSelectKBest, gene_indices


def main():
    config.ensure_dirs()
    X_df, y_str = io.load_labelled(config.pair_matrix_path())
    genes = X_df.columns.tolist()
    X = X_df.to_numpy()

    classes = sorted(y_str.unique())
    y = (y_str == classes[1]).astype(int).to_numpy()
    counts = np.bincount(y)

    print(f"Matrix: {X.shape[0]} samples x {X.shape[1]} genes")
    print(f"Classes: {classes[0]}={counts[0]} (label 0), {classes[1]}={counts[1]} (label 1)")
    print(f"Always-predict-{classes[int(counts.argmax())]} floor: "
          f"balanced 0.5000, plain {counts.max() / counts.sum():.4f}\n")

    anchor_idx = gene_indices(genes, config.CLOCK_GENES)
    all_idx = list(range(len(genes)))

    panels = {
        "clock only (a priori)": lambda: FixedGeneSet(anchor_idx),
        f"top-{config.TOP_N_GENES} by Welch": lambda: WelchSelectKBest(config.TOP_N_GENES),
        "anchor panel (100+11)": lambda: AnchorPanelSelector(anchor_idx),
        "all genes": lambda: FixedGeneSet(all_idx),
    }

    fold_list = evaluation.folds(X, y)
    print(evaluation.header())

    rows = []
    for panel_name, factory in panels.items():
        for model_name in config.MODELS:
            scores = evaluation.cross_validate_selector(X, y, factory, model_name, fold_list)
            print(evaluation.format_row(f"{panel_name} / {model_name}", scores))
            rows.append({
                "panel": panel_name,
                "model": model_name,
                "balanced_accuracy": scores["balanced_accuracy"],
                "balanced_accuracy_std": scores["balanced_accuracy_std"],
                "roc_auc": scores["roc_auc"],
                "accuracy": scores["accuracy"],
                "folds": list(scores["balanced_accuracy_folds"]),
            })

    resolution = evaluation.one_sample_worth(y)
    print(f"\nOne {classes[int(counts.argmin())]} sample is worth {resolution:.4f} "
          "balanced accuracy; treat smaller gaps as noise.")

    table = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    table.drop(columns=["folds"]).to_csv(config.csv_path("stage4_results.csv"), index=False)
    io.write_json({
        "classes": classes,
        "class_counts": counts,
        "majority_floor": counts.max() / counts.sum(),
        "one_sample_worth": resolution,
        "results": rows,
    }, "stage4_results.json")

    best = table.iloc[0]
    print(f"\nBest: {best['panel']} / {best['model']} at "
          f"{best['balanced_accuracy']:.4f} balanced accuracy")
    print("Stage 4 complete.")


if __name__ == "__main__":
    main()
