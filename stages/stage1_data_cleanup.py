"""Stage 1 - load the cohorts, label the samples, write the analysis matrices.

Outputs (results/csvs, results/json):
    ExpressionLevels.csv                     genes x tumor samples, all cohorts
    ExpressionLevels_<A>_<B>_Labelled.csv    one per cohort pair
    ClockExpressionLevels_Labelled.csv       clock genes only, all cohorts
    metadata.json                            barcode -> cohort + sample type
    stage1_summary.json                      what was kept and dropped
"""

import pandas as pd

from saygex import config, io


def main():
    config.ensure_dirs()
    print("Loading cohorts...", flush=True)
    frames = io.load_all_cohorts()

    n_genes = len(frames[config.COHORTS[0]].index)
    print(f"  gene axis verified identical across cohorts: {n_genes} genes")

    metadata = io.build_metadata(frames)
    io.write_json(metadata, "metadata.json")

    # Keep tumor samples only, and report exactly what that removed.
    summary = {"n_genes": n_genes, "cohorts": {}}
    tumor_frames = {}
    for cohort, df in frames.items():
        keep = [b for b in df.columns if metadata[b]["is_tumor"]]
        by_type = {}
        for barcode in df.columns:
            label = metadata[barcode]["sample_type_label"]
            by_type[label] = by_type.get(label, 0) + 1
        tumor_frames[cohort] = df[keep]
        summary["cohorts"][cohort] = {
            "total": len(df.columns),
            "kept": len(keep),
            "dropped": len(df.columns) - len(keep),
            "by_sample_type": by_type,
        }
        print(f"  {cohort}: kept {len(keep)}/{len(df.columns)} "
              f"({', '.join(f'{v} {k}' for k, v in sorted(by_type.items()))})")

    all_levels = pd.concat([tumor_frames[c] for c in config.COHORTS], axis=1)
    all_levels.to_csv(config.csv_path("ExpressionLevels.csv"))

    summary["pairs"] = {}
    for a, b in config.PAIRS:
        pair = io.to_labelled([tumor_frames[a], tumor_frames[b]], metadata)
        pair.to_csv(config.pair_matrix_path((a, b)))
        summary["pairs"][f"{a}_{b}"] = {
            "samples": int(pair.shape[0]), "genes": int(pair.shape[1] - 1)
        }
        print(f"  {a} vs {b}: {pair.shape[0]} samples x {pair.shape[1] - 1} genes")

    present = [g for g in config.CLOCK_GENES if g in all_levels.index]
    missing = [g for g in config.CLOCK_GENES if g not in all_levels.index]
    if missing:
        print(f"  ! clock genes absent from the matrix: {missing}")

    clock = io.to_labelled([all_levels.loc[present]], metadata)
    clock.to_csv(config.csv_path("ClockExpressionLevels_Labelled.csv"))
    summary["clock_genes"] = {"found": present, "missing": missing}
    print(f"  clock matrix: {clock.shape[0]} samples x {len(present)} genes")

    io.write_json(summary, "stage1_summary.json")
    print("Stage 1 complete.")


if __name__ == "__main__":
    main()
