# Project structure

```
saygex-neurotech/
├── pipeline.py                 entry point; runs stages in order
├── requirements.txt
├── gzips/                      input data (three TCGA HiSeqV2 matrices)
│
├── saygex/                     reusable library — no side effects on import
│   ├── config.py               every path and constant, in one place
│   ├── stats.py                Welch t-test, BH FDR, softmax, empirical p
│   ├── io.py                   cohort loading, labelled-matrix read/write
│   ├── selectors.py            gene-panel selectors (the method lives here)
│   └── evaluation.py           leak-free cross-validation harness
│
├── stages/                     executable steps — one concern each
│   ├── stage1_data_cleanup.py
│   ├── stage2_tumor_selection.py
│   ├── stage3_feature_panels.py
│   ├── stage4_predictive_models.py
│   └── stage5_validation.py
│
├── docs/
│   ├── METHODS.md              what the pipeline does and why
│   └── STRUCTURE.md            this file
│
└── results/                    all generated output (gitignored)
    ├── csvs/                   tables
    ├── plots/                  figures
    └── json/                   machine-readable summaries
```

## Design rules

**`saygex/` is a library, `stages/` are scripts.** Importing anything from
`saygex` has no side effects — no file writes, no directory creation, no
computation. Every stage script is a thin orchestrator over library functions.

**No path or magic number outside `config.py`.** If you want to change the
panel size, the FDR cutoff, the number of null draws, or where output lands,
there is exactly one place to look.

**Every stage writes a JSON summary.** `results/json/stageN_*.json` records what
each step did in machine-readable form, so a later stage or an external script
never has to re-derive it or parse console output.

**Anything label-aware is a scikit-learn transformer.** That is what makes it
possible to refit selection inside a CV fold, which is the difference between
an honest accuracy number and an inflated one.

## Running

```bash
python pipeline.py                 # stages 1-4, a few minutes
```

```bash
python pipeline.py --validate      # stages 1-5, adds the null distributions
```

```bash
python pipeline.py --only 4 5      # re-run specific stages
```

Individual stages run as modules (they import the `saygex` package, so they
need the project root on the path):

```bash
python -m stages.stage4_predictive_models
```

## Where to look for what

| You want to... | Go to |
|---|---|
| change a constant | `saygex/config.py` |
| understand the method | `saygex/selectors.py` → `AnchorPanelSelector` |
| check the CV is honest | `saygex/evaluation.py` → `cross_validate_selector` |
| see the headline numbers | `results/csvs/stage4_results.csv` |
| see whether the clock earns its place | `results/csvs/stage5_e3_ablation.csv` |
| read the reasoning | `docs/METHODS.md` |
