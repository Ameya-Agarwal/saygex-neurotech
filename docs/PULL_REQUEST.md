# Fix data leakage and sample contamination; add clock-gene validation

## Summary

The pipeline reported ~0.98 accuracy for a clock-anchored gene panel. Two
defects inflated that number, and a third made the project's central claim
untestable. This PR fixes all three, restructures the codebase, and adds the
controls that determine what the result actually means.

**The headline finding changed.** The clock genes are genuinely differentially
expressed — but they are not better features than random genes, because the
classification task turns out to be saturated.

## Correctness fixes

| Issue | Effect |
|---|---|
| **Adjacent normal tissue labelled as a tumor subtype** | 129 of 1020 samples are TCGA `-11` (solid tissue normal). Contamination was unequal — 27% of KICH vs 12% of KIRC — so it biased the comparison rather than adding noise. Only `-01`/`-05` now enter the analysis. |
| **Feature-selection leakage** | Genes were chosen by t-testing all 600 samples, then cross-validated on those same samples. Every label-aware step is now refit inside each training fold. |
| **`scale_pos_weight` inverted** | Used `max/min`, which exceeds 1 whichever class is larger. XGBoost scales the *positive* class and `LabelEncoder` makes KIRC (the majority) label 1, so this amplified the imbalance ~44× in the wrong direction. Now `negatives/positives`. |
| **Sample identity destroyed** | Barcodes were overwritten by tumor labels, producing duplicate-index frames that pandas concatenates unpredictably. Barcodes stay as the index; the label is a `Tumor_Type` column. |
| **Degenerate softmax weighting** | Raw −log10(p) spans 0–200+, so softmax put ~1.0 on one gene and underflowed the rest to exactly `0.0`. The "weighted" combination collapsed to a single gene, and which anchors got dropped depended on dict ordering. Scores are standardised first; anchors are dropped by p-value. |
| **Equal-variance t-test** | Groups differ 8-fold in size. Switched to Welch throughout — selecting with the ANOVA F-test instead cost ~4 points of balanced accuracy. |
| **No multiple-testing correction** | ~20,500 genes screened at p<0.05. Added Benjamini–Hochberg FDR. |
| **`CalibratedClassifierCV(cv="prefit")`** | Removed in scikit-learn 1.8, was fit on its own training rows, and being monotone could never change the AUC it computed. Removed; ROC-AUC from `decision_function`. |
| Hard-coded `"KICH"` positive class; `"python"` instead of `sys.executable` | Fixed. |

## Structure

```
pipeline.py          entry point, --validate / --only N
saygex/              library: config, stats, io, selectors, evaluation
stages/              stage1…stage5, one concern each
docs/                RESULTS.md, METHODS.md, STRUCTURE.md
results/             all generated output (gitignored)
```

- `saygex/` has no import side effects; `stages/` are thin orchestrators.
- Every path and constant lives in `saygex/config.py`.
- Each stage writes a JSON summary, so nothing downstream re-derives state or
  parses console output.
- **`AnchorPanelSelector` takes the anchor set as a parameter** rather than
  hard-coding the clock genes. That single change is what makes the project's
  claim testable — the old code could not express "run this with different
  anchors."

## New: stage 5 validation

Four controls, because the benchmark alone cannot distinguish "the clock is
special" from "111 reasonable genes classify kidney tumors."

- **E1** — size-matched random panels
- **E2** — expression-matched random panels (harder null; ~270 genes are flat
  zero genome-wide, so an unmatched null only proves the clock beats noise)
- **E3** — random-anchor ablation: the identical procedure with random anchors
- **E4** — accuracy vs panel size

## Results

**What holds up:** 11 of 14 core clock genes are differentially expressed with
large effects — `NPAS2` at −4.31 log2FC (q ≈ 4.6e-29), `PER2` at −2.10
(q ≈ 7.1e-30). Real biology, consistent with BMAL1–HIF1α crosstalk in a
*VHL*-driven tumor.

**What does not:**

| Test | Observed | Null | p |
|---|---:|---:|---:|
| E1 clock vs random-14 (LinearSVC) | 0.9391 | 0.9416 ± 0.0213 | 0.63 |
| E1 clock vs random-14 (XGBoost) | 0.9271 | 0.9298 ± 0.0233 | 0.61 |
| E2 clock vs expression-matched (LinearSVC) | 0.9391 | 0.9530 ± 0.0132 | 0.87 |
| E2 clock vs expression-matched (XGBoost) | 0.9271 | 0.9436 ± 0.0167 | 0.85 |
| E3 clock vs random anchors (LinearSVC) | 0.9484 | 0.9562 ± 0.0117 | 0.78 |
| E3 clock vs random anchors (XGBoost) | 0.9878 | 0.9817 ± 0.0026 | 0.12 |

**Why**, from E4 — supervised top-k balanced accuracy:

| k | 5 | 10 | 14 | 25 | 111 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| LinearSVC | 0.9841 | 0.9850 | 0.9841 | 0.9860 | 0.9475 | 0.9542 |
| XGBoost | 0.9811 | 0.9801 | **0.9897** | 0.9734 | 0.9801 | 0.9801 |

**Five genes chosen by a t-test classify at 0.98.** Fourteen reach 0.9897 —
higher than the 111-gene anchor panel, with 8× fewer genes. The task is
saturated, so no panel can distinguish itself from any other panel. The clock
genes did not fail; there was no headroom in which to succeed.

## Reading the numbers

- **Balanced accuracy, not plain.** 66 KICH vs 534 KIRC: always-predict-KIRC
  scores 0.890 plain / 0.500 balanced.
- **One sample is worth 0.0076.** Gaps below that are one sample, not a finding.
  Results tables print this reminder.

## Verification

`python pipeline.py` runs stages 1–4 in ~4 min; `--validate` adds stage 5
(~40 min). Stages 1–4 reproduce the pre-existing numbers exactly, so the
restructure is behaviour-preserving where it should be.

## Suggested follow-ups

Subtype classification is saturated, so no feature-selection method can
demonstrate value on it.

1. **Change the endpoint** — survival, grade, stage. In TCGA, not saturated.
2. **Use the 129 discarded normals** — tumor-vs-normal clock disruption is the
   question the effect sizes actually speak to. Already labelled in
   `results/json/metadata.json`.
3. **Retest on a harder pair** — KIRC vs KIRP, silhouette 0.450 vs 0.627.

Full detail in [`docs/RESULTS.md`](docs/RESULTS.md).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
