# Results

Target comparison: **66 KICH vs 534 KIRC** primary tumors, 20,530 genes.
All accuracies are **balanced accuracy** under 5-fold stratified CV with every
label-aware step refit inside each fold.

Reference points to read every number against:

| | |
|---|---:|
| Always-predict-KIRC floor (balanced) | 0.5000 |
| Always-predict-KIRC floor (plain accuracy) | 0.8900 |
| Resolution: one KICH sample is worth | **0.0076** |

That last line governs the whole document. With 66 minority samples, a single
misclassification moves balanced accuracy by 1/(2 × 66). **Any gap under 0.0076
is one sample, not a finding.**

---

## 1. Cohort separability

| Pair | Silhouette | Genes at FDR<0.05 and \|log2FC\|>1 |
|---|---:|---:|
| KICH vs KIRC | 0.627 | 7,130 |
| KICH vs KIRP | 0.591 | 6,140 |
| KIRC vs KIRP | 0.450 | 3,836 |

KICH vs KIRC is the most separable pair, which is why the modelling stages use
it. Note the size of those gene counts — **7,130 genes** clear a strict
significance bar. That is the first hint of what the validation later confirms:
this comparison is not subtle.

## 2. Clock genes are genuinely differentially expressed

From `results/csvs/Clock_Genes_Analysis.csv` (log2FC is KICH − KIRC):

| Gene | p | q (BH) | log2FC |
|---|---:|---:|---:|
| PER2 | 5.0e-31 | 7.1e-30 | −2.10 |
| NPAS2 | 6.5e-30 | 4.6e-29 | **−4.31** |
| NR1D2 | 4.7e-28 | 2.2e-27 | +1.55 |
| RORA | 3.4e-25 | 1.2e-24 | −1.45 |
| CRY2 | 3.2e-20 | 9.0e-20 | +1.38 |
| ARNTL | 7.2e-13 | 1.7e-12 | −0.66 |
| ARNTL2 | 1.3e-11 | 2.5e-11 | −1.86 |
| PER1 | 1.4e-04 | 2.4e-04 | −0.49 |
| RORB | 1.7e-03 | 2.6e-03 | +1.00 |
| CLOCK | 4.9e-03 | 6.8e-03 | −0.41 |
| RORC | 6.1e-03 | 7.8e-03 | +0.54 |
| CRY1 | 6.1e-02 | 7.1e-02 | −0.15 |
| NR1D1 | 2.4e-01 | 2.5e-01 | +0.14 |
| PER3 | 3.4e-01 | 3.4e-01 | +0.09 |

Eleven of fourteen core clock genes are significant after FDR correction, and
the effects are large — `NPAS2` differs by 4.3 log2 units, roughly 20-fold.
**This part of the project holds up.** It is a real biological observation and
it is consistent with known BMAL1–HIF1α crosstalk in a *VHL*-driven tumor.

Keep it separate in your head from everything below. "These genes differ
between subtypes" and "these genes are especially useful for telling subtypes
apart" are different claims, and only the first one survives.

## 3. Classification benchmark (leak-free)

| Panel | Model | Balanced acc | sd across folds | ROC-AUC | Plain acc |
|---|---|---:|---:|---:|---:|
| anchor panel (100+11) | XGBoost | **0.9878** | ± 0.0038 | 0.9951 | 0.9783 |
| top-200 by Welch | XGBoost | 0.9801 | ± 0.0167 | 0.9956 | 0.9767 |
| all 20,530 genes | XGBoost | 0.9743 | ± 0.0305 | 0.9976 | 0.9783 |
| all 20,530 genes | LinearSVC | 0.9738 | ± 0.0120 | 0.9941 | 0.9533 |
| top-200 by Welch | LinearSVC | 0.9542 | ± 0.0311 | 0.9806 | 0.9667 |
| anchor panel (100+11) | LinearSVC | 0.9484 | ± 0.0405 | 0.9782 | 0.9683 |
| clock only (14) | LinearSVC | 0.9391 | ± 0.0512 | 0.9815 | 0.9517 |
| clock only (14) | XGBoost | 0.9271 | ± 0.0499 | 0.9831 | 0.9533 |

Read alone, this table looks like a win: the clock-anchored panel tops it, and
14 clock genes alone reach 0.9391 against a 0.5000 floor.

**The sd column is why that reading is already shaky before stage 5.** Fold-to-fold
spread reaches ± 0.0512 — nearly seven times the 0.0076 one-sample resolution,
and larger than every gap in the table. The bottom six rows all overlap inside
one standard deviation of each other. The ranking is mostly fold-assignment
noise, and only the top row has a spread (± 0.0038) tight enough to be taken at
face value.

So the honest reading of this table on its own is "every panel lands somewhere
around 0.93–0.99 and they are hard to separate" — not "the anchor panel wins".
Stage 5 then explains why they are hard to separate.

## 4. Validation

### E1 — clock vs size-matched random panels

| Panel | Model | Observed | Null (mean ± sd) | p | Fair test? |
|---|---|---:|---:|---:|:--:|
| clock only | LinearSVC | 0.9391 | 0.9416 ± 0.0213 | 0.628 | yes |
| clock only | XGBoost | 0.9271 | 0.9298 ± 0.0233 | 0.611 | yes |
| anchor panel | LinearSVC | 0.9484 | 0.9475 ± 0.0170 | 0.535 | no |
| anchor panel | XGBoost | 0.9878 | 0.9577 ± 0.0106 | 0.007 | no |
| top-200 | LinearSVC | 0.9542 | 0.9547 ± 0.0138 | 0.532 | no |
| top-200 | XGBoost | 0.9801 | 0.9590 ± 0.0103 | 0.027 | no |

**Random 14-gene panels match the clock panel** (p ≈ 0.62). The two rows marked
*fair* are the only like-for-like comparisons — clock genes and random genes are
both label-blind. The clock loses both.

The other four rows compare label-selected panels against label-blind nulls, so
their small p-values confirm only that supervised selection works.

### E2 — clock vs expression-matched random panels

| Panel | Model | Observed | Null (mean ± sd) | p |
|---|---|---:|---:|---:|
| clock only | LinearSVC | 0.9391 | 0.9530 ± 0.0132 | 0.870 |
| clock only | XGBoost | 0.9271 | 0.9436 ± 0.0167 | 0.847 |

Worse. When random genes are drawn from the same mean-expression and variance
bins as the clock genes, the null **rises** (0.9530 vs E1's 0.9416) and the
clock panel sits about 1 SD *below* it.

The null rising is itself informative: it shows E1's plain-random null was
partly diluted by low-expression noise, which is exactly the weakness E2 was
built to remove.

### E3 — random-anchor ablation (the thesis test)

Identical selection procedure, anchors swapped:

| Null type | Model | Clock anchors | Random anchors | p |
|---|---|---:|---:|---:|
| random | LinearSVC | 0.9484 | 0.9562 ± 0.0117 | 0.777 |
| expression-matched | LinearSVC | 0.9484 | 0.9538 ± 0.0113 | 0.704 |
| random | XGBoost | 0.9878 | 0.9817 ± 0.0026 | 0.123 |
| expression-matched | XGBoost | 0.9878 | 0.9816 ± 0.0030 | 0.120 |

**Clock anchors never significantly beat random anchors.** Best case p = 0.12.
With LinearSVC, random anchors are *better*.

The XGBoost rows are the closest the thesis comes to support: 0.9878 against a
tight null of 0.9817 ± 0.0026. But the null distribution of balanced accuracy is
quantised in steps of 0.0076 and bounded above, so the standard-deviation
distance is misleading — the empirical p-value is the honest measure, and it is
0.12.

The method works. It is not working *because of the clock*.

### E4 — accuracy vs panel size

Supervised top-k selection, balanced accuracy:

| k | 5 | 10 | 14 | 25 | 50 | 111 | 200 | 500 | 1000 | 5000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LinearSVC | 0.9841 | 0.9850 | 0.9841 | 0.9860 | 0.9601 | 0.9475 | 0.9542 | 0.9589 | 0.9657 | 0.9689 |
| XGBoost | 0.9811 | 0.9801 | **0.9897** | 0.9734 | 0.9743 | 0.9801 | 0.9801 | 0.9647 | 0.9734 | 0.9743 |

**This is the finding that reframes the whole project.**

Five genes chosen by a t-test score 0.981–0.984. Fourteen reach 0.9897 — the
highest number anywhere in this study, beating the 111-gene anchor panel's
0.9878 with eight times fewer genes.

The curve is flat from k = 5 onward. KICH vs KIRC saturates at a handful of
genes; everything past that is noise-fitting. Individual cells wobble by more
than 0.0076, so do not read single points — read the shape.

Random panels for contrast: 0.8901 ± 0.0676 at k=5, rising to 0.9687 ± 0.0045 at
k=5000. Even five random genes clear 0.89 balanced accuracy.

### Verdict

Printed verbatim by `stages/stage5_validation.py` at the end of the run:

```
E1_random                     LinearSVC  clock panel does NOT beat its null (p=0.6279)
E1_random                     XGBoost    clock panel does NOT beat its null (p=0.6113)
E2_expression_matched         LinearSVC  clock panel does NOT beat its null (p=0.8704)
E2_expression_matched         XGBoost    clock panel does NOT beat its null (p=0.8472)
E3_random_anchors             LinearSVC  clock anchors does NOT beat random anchors (p=0.7774)
E3_expression_matched_anchors LinearSVC  clock anchors does NOT beat random anchors (p=0.7043)
E3_random_anchors             XGBoost    clock anchors does NOT beat random anchors (p=0.1229)
E3_expression_matched_anchors XGBoost    clock anchors does NOT beat random anchors (p=0.1196)
```

Eight tests, eight negatives. No configuration of model, null type, or panel
reaches significance.

## 5. Interpretation

**The classification task is saturated.** With 7,130 genes clearing a strict
significance bar and 5 genes sufficing for ~0.98, there is no headroom in which
a panel can prove itself better than another panel. The clock genes did not fail
to distinguish themselves — there was nothing left to distinguish.

This explains every stage-5 result at once. Random 14-gene panels match the
clock (E1) because almost any 14 genes work. Expression-matched panels beat it
(E2) because the constraint that actually matters is "be well-expressed", not
"be circadian". Random anchors match clock anchors (E3) because the anchor step
is doing little beyond re-ranking an already-sufficient screen.

**The 0.9878 headline is real but uninformative.** It is what ~100 genes produce
on an easy problem. It is not evidence about circadian biology.

**What survives is section 2.** Eleven of fourteen clock genes are
differentially expressed with large effects. That is a finding. The inference
from it to "clock genes are special features for classification" is what the
controls reject.

## 6. Limitations

- **Researcher degrees of freedom.** The constants (200 → 100, drop 3) and the
  clock gene list were chosen while looking at this dataset. Cross-validation
  cannot undo that; only a held-out cohort can. KIRP (291 tumors) is loaded,
  pair-compared, and then unused.
- **Small minority class.** 66 KICH samples cap the resolution at 0.0076 and
  make the null distributions lumpy.
- **Single dataset, single platform.** One Xena HiSeqV2 snapshot; no replication
  on an independent cohort or assay.
- **Subtype is a weak endpoint.** It is determined routinely by histopathology,
  so a classifier for it has limited standalone value even when it works.

## 7. Where to take this next

1. **Change the endpoint.** Subtype classification is saturated and no method
   can distinguish itself on it. Survival, tumor grade, and stage are available
   in TCGA and are not saturated. If circadian disruption matters clinically,
   that is where it would show.
2. **Use the 129 discarded normals.** Tumor-vs-normal clock disruption is the
   question the section-2 effect sizes actually speak to, and the samples are
   already labelled in `results/json/metadata.json`.
3. **Retest the method on a harder pair.** KIRC vs KIRP has silhouette 0.450
   against 0.627 for the current pair. A genuinely difficult comparison is the
   only place an anchor method could demonstrate value.
4. **If the clock claim is pursued, test the pathway, not the panel.** A
   coordinated-dysregulation score across the oscillator, compared against
   matched gene sets, asks a biological question that classification accuracy
   on a saturated task cannot answer.

---

## Reproducing

```bash
python pipeline.py --validate
```

Stages 1-4 take ~4 minutes; stage 5 took 2689s (~45 min) on a laptop CPU,
300 draws per null.

Raw outputs: `results/csvs/stage4_results.csv`,
`results/csvs/stage5_e1_e2_nulls.csv`, `results/csvs/stage5_e3_ablation.csv`,
`results/csvs/stage5_e4_curve.csv`, and three figures in `results/plots/`
(`E1_E2_null_distributions.png`, `E3_anchor_ablation.png`,
`E4_accuracy_vs_panel_size.png`).
