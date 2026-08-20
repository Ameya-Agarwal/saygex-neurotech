# Methods

## The question

Do circadian clock genes carry enough signal to distinguish renal cell
carcinoma subtypes — and does using the clock as an *anchor* find better tumor
genes than plain differential expression does?

The second half is the novel claim. The first half is a prerequisite for it.

## Data

Three TCGA kidney cohorts from the UCSC Xena hub, gene-level `HiSeqV2`
matrices, log2(norm_count + 1), 20,530 genes:

| Cohort | Subtype | Samples | Primary tumor | Adjacent normal | Other |
|---|---|---:|---:|---:|---:|
| KICH | Chromophobe RCC | 91 | 66 | 25 | 0 |
| KIRC | Clear cell RCC | 606 | 533 | 72 | 1 |
| KIRP | Papillary RCC | 323 | 290 | 32 | 1 |

### Sample selection

TCGA barcodes end in a sample-type code: `-01` primary tumor, `-11` adjacent
normal, `-05` additional new primary. **Only tumor samples (`01`, `05`) enter
the analysis.**

This matters more than it looks. Adjacent normal kidney is much the same tissue
whichever cohort filed it, so including normals turns "KICH vs KIRC" into a
blend of that question and a tumor-vs-normal question. The contamination is not
even balanced — normals are 27% of KICH but 12% of KIRC — so it biases the
comparison rather than just adding noise.

Set `TUMOR_SAMPLE_TYPES` in `saygex/config.py` to revisit this. The metadata in
`results/json/metadata.json` records every sample's type, so the discarded
normals remain available for a tumor-vs-normal analysis.

After filtering, the target pair is **66 KICH vs 534 KIRC**.

## Statistics

**Welch's t-test** (`equal_var=False`) throughout. The groups differ 8-fold in
size and substantially in variance, which is exactly where Student's
equal-variance test misstates the p-value. This is not cosmetic: selecting
genes with the equal-variance ANOVA F-test instead costs several points of
balanced accuracy.

**Benjamini–Hochberg FDR** for significance calls. Each screen tests ~20,500
genes, where uncorrected p < 0.05 would return roughly a thousand genes by
chance alone.

**Softmax over standardised scores** for anchor weighting. The raw inputs are
−log10(p) values spanning 0 to 200+; a plain softmax over that range puts ~1.0
on the single best gene and underflows every other weight to exactly 0.0. That
collapses the weighted combination to one gene and makes tie-breaks depend on
dictionary ordering. Standardising first preserves the ranking and restores
discrimination.

## The anchor method

Implemented as `AnchorPanelSelector` in `saygex/selectors.py`:

1. Screen every gene by Welch p-value; keep the top **200**.
2. Weight each anchor (clock) gene by how discriminative it is.
3. Drop the **3** weakest anchors, keeping 11.
4. Rank the 200 screened genes by weighted |correlation| with the kept anchors.
5. Final panel = best **100** of those + the 11 anchors = **111 genes**.

The anchor set is a **parameter, not a hard-coded constant**. This is
deliberate: the claim "circadian genes make good anchors" is only testable by
running the identical procedure with other anchors (see E3).

## Evaluation

**Every step that consults the labels is refit inside the training fold.** That
includes gene selection, scaling, and class weighting. 5-fold stratified CV.

This is the single most important methodological point in the project. The
saved panels in `results/csvs/` were built from all 600 samples, so scoring a
model on them measures a model that has already seen its own test set. Stage 4
reports that biased number too, but labelled as such.

**Balanced accuracy is the headline metric**, not plain accuracy. With 66 vs
534 samples, a model that always predicts KIRC scores 0.890 plain accuracy and
0.500 balanced. Plain accuracy is reported alongside for continuity, and
ROC-AUC as an imbalance-robust ranking measure.

**Resolution limit.** With 66 minority samples, one misclassification moves
balanced accuracy by 1/(2 × 66) = **0.0076**. Differences smaller than that are
one sample, not a finding. Every results table prints this figure.

### Class weighting

`scale_pos_weight = sum(negatives) / sum(positives)`, per the XGBoost docs.
XGBoost scales the gradient of the *positive* class (label 1). `LabelEncoder`
sorts alphabetically, so KIRC — the majority — becomes label 1 here. Using
`max/min` instead yields a value above 1 whichever class is larger, which
upweights the majority class: the opposite of rebalancing.

## Validation (stage 5)

Stage 4 alone cannot distinguish "the clock is special" from "111 reasonable
genes classify kidney tumors". Four controls address that.

### E1 — size-matched random panels

Null: random gene sets of the same size, same CV harness. Reports an empirical
p-value with the standard +1 correction.

**Only the clock-only comparison is a fair test.** Both the clock panel and its
null are label-blind. The top-200 and anchor panels were built *using* the
labels, so beating a random panel is expected and demonstrates only that
supervised selection works. Those rows are marked `fair_test = False`.

### E2 — expression-matched random panels

As E1, but each random gene is drawn from the same mean-expression and variance
bin as the clock gene it replaces. Around 270 genes in the matrix are flat zero
everywhere and many more are near-noise, so an unmatched null largely proves
the clock beats junk. E2 asks whether the clock beats genes that *look like*
the clock — the harder and more meaningful question.

### E3 — random-anchor ablation (the thesis test)

Runs the identical selection procedure with random anchors instead of clock
anchors. Same panel size, same steps, only the anchor set differs. Both a plain
random and an expression-matched version.

**This is the experiment that tests the project's actual claim.** If clock
anchors do not beat random anchors, the method is not about the clock.

Made affordable by caching: the genome-wide Welch screen is the expensive step
and does not depend on the anchors, so it is computed once per fold and reused
across every draw.

### E4 — accuracy vs panel size

Plots balanced accuracy against k for supervised top-k, random-k, and anchor-k
panels. This *dissolves* the feature-count confound rather than controlling for
it — you read the whole curve instead of arguing about three points.

The anchor method generalises to arbitrary k as "screen 2 × n_final, keep
n_final, add the kept anchors", which is exactly what it does at k = 111.

## Known limitations

**Researcher degrees of freedom.** The constants (200 → 100, drop 3) and the
clock gene list were chosen while looking at this dataset. Cross-validation
cannot undo that; only a held-out cohort can. **KIRP (290 tumors) is loaded,
pair-compared, and then unused** — it is available as an external validation
set.

**Subtype may be the wrong endpoint.** If the interest is circadian disruption
in cancer, survival, grade, and stage are more meaningful endpoints and are all
available in TCGA. None are used here.

**The discarded normals are interesting.** Tumor-vs-normal clock disruption is
arguably a better question than subtype classification, and 129 normal samples
are sitting labelled in the metadata.
