# saygex-neurotech

Do circadian clock genes carry enough signal to distinguish renal cell
carcinoma subtypes — and does using the clock as an *anchor* find better tumor
genes than plain differential expression does?

Three TCGA kidney cohorts (KICH, KIRC, KIRP), 20,530 genes, gene-level RNA-seq.

## Headline

**Yes to the first question, no to the second — and the second "no" is the
interesting one.**

Eleven of fourteen core clock genes are differentially expressed between KICH
and KIRC with large effects (`NPAS2` at 4.3 log2 units, q ≈ 5e-29). That is a
real biological result.

But the clock-anchored gene panel does **not** outperform the same procedure run
with random anchor genes (p = 0.12 at best), and 14 clock genes do **not**
outperform 14 random genes (p = 0.62), or 14 expression-matched random genes
(p = 0.85, where the clock is actually *worse*).

The reason is visible in one line: **five genes chosen by a t-test classify
these subtypes at 0.98 balanced accuracy.** The task is saturated. There is no
headroom in which any panel can prove itself better than any other panel, so the
clock genes never had a chance to distinguish themselves.

Full numbers and reasoning: **[`docs/RESULTS.md`](docs/RESULTS.md)**.

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python pipeline.py
```

Stages 1–4 take about four minutes. Add `--validate` for the statistical
controls in stage 5, which take around 40 minutes:

```bash
python pipeline.py --validate
```

Everything generated lands in `results/` (`csvs/`, `plots/`, `json/`).

## What the pipeline does

| Stage | Question it answers |
|---|---|
| 1 · data cleanup | Get three cohorts onto one gene axis, label every sample |
| 2 · tumor selection | Which subtype pair is most separable, and which genes separate it |
| 3 · feature panels | Build the clock-anchored gene panel |
| 4 · predictive models | How well does each panel classify, measured honestly |
| 5 · validation | Does the clock actually earn its place, or would any genes do |

Stage 5 is the one that matters. Stages 1–4 produce a number; stage 5 tests
whether that number means anything. It turned out it didn't.

## The method

`AnchorPanelSelector` in [`saygex/selectors.py`](saygex/selectors.py):

1. Screen all genes by Welch p-value → keep the top 200
2. Weight the 14 clock genes by how discriminative each one is
3. Drop the 3 weakest → 11 anchors remain
4. Rank the 200 screened genes by weighted |correlation| with those anchors
5. Final panel = top 100 + the 11 anchors = **111 genes**

The anchor set is a **parameter, not a hard-coded constant**. That single design
choice is what makes the central claim testable: stage 5 reruns this exact
procedure with random anchors and compares.

## Reading any number in this repo

**Balanced accuracy, not plain accuracy.** The target pair is 66 KICH vs 534
KIRC. A model that always predicts KIRC scores **0.890 plain** and **0.500
balanced**.

**One sample is worth 0.0076.** With 66 minority samples, one misclassification
moves balanced accuracy by 1/(2 × 66). Gaps smaller than that are one sample,
not a finding. Every results table prints this reminder.

**Only leak-free numbers are quotable.** Gene selection is refit inside every CV
fold. The panels saved in `results/csvs/` were built from all 600 samples, so
scoring them directly measures a model that has already seen its own test set.

## Sample selection

TCGA barcodes end in a sample-type code: `-01` primary tumor, `-11` adjacent
normal. The cohorts ship with a lot of normal tissue (KICH 25/91, KIRC 72/606,
KIRP 32/323), and adjacent normal kidney is much the same tissue whichever
cohort filed it — so including it turns "KICH vs KIRC" into a blend of that
question and a tumor-vs-normal question, with unequal contamination on each
side (27% of KICH, 12% of KIRC).

**Only tumor samples are analysed**, leaving 66 KICH / 534 KIRC / 291 KIRP. The
discarded normals stay labelled in `results/json/metadata.json` and are
recoverable via `TUMOR_SAMPLE_TYPES` in `saygex/config.py`.

## Project layout

```
pipeline.py          entry point, --validate / --only N
saygex/              library: config, stats, io, selectors, evaluation
stages/              stage1…stage5, one concern each
docs/                RESULTS.md, METHODS.md, STRUCTURE.md
results/             all generated output (gitignored)
```

- [`docs/RESULTS.md`](docs/RESULTS.md) — every number, what it means, what to do next
- [`docs/METHODS.md`](docs/METHODS.md) — statistical choices and why
- [`docs/STRUCTURE.md`](docs/STRUCTURE.md) — layout and design rules

## Data

All three inputs are the **gene-level** `HiSeqV2` matrices from the UCSC Xena
TCGA hub, log2(norm_count + 1). Place the `.gz` files in `gzips/`.

- KICH: https://xenabrowser.net/datapages/?dataset=TCGA.KICH.sampleMap%2FHiSeqV2&host=https%3A%2F%2Ftcga.xenahubs.net
- KIRC: https://xenabrowser.net/datapages/?dataset=TCGA.KIRC.sampleMap%2FHiSeqV2&host=https%3A%2F%2Ftcga.xenahubs.net
- KIRP: https://xenabrowser.net/datapages/?dataset=TCGA.KIRP.sampleMap%2FHiSeqV2&host=https%3A%2F%2Ftcga.xenahubs.net

Stage 1 fails loudly if the three gene axes disagree rather than padding the
mismatch with NaN. (The KIRP link previously pointed at `HiSeqV2_exon`, which is
exon-level with an entirely different row index.)

## Where next

Subtype classification is saturated, so no feature-selection method can
demonstrate value on it. The open directions:

1. **Change the endpoint** — survival, grade, stage. All in TCGA, none saturated.
2. **Use the 129 discarded normals** — tumor-vs-normal clock disruption is the
   question the effect sizes actually speak to.
3. **Retest on a harder pair** — KIRC vs KIRP, silhouette 0.450 vs 0.627.
