"""Every path and tunable constant the project uses, in one place.

Nothing else in the codebase should hard-code a path or a magic number.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GZIP_DIR = os.path.join(ROOT, "gzips")
RESULTS_DIR = os.path.join(ROOT, "results")
CSV_DIR = os.path.join(RESULTS_DIR, "csvs")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
JSON_DIR = os.path.join(RESULTS_DIR, "json")

# ---------------------------------------------------------------- cohorts ---

COHORTS = ["KICH", "KIRC", "KIRP"]
PAIRS = [("KICH", "KIRC"), ("KICH", "KIRP"), ("KIRC", "KIRP")]

# The pair every modelling stage works on. Stage 2 checks empirically that this
# really is the most separable pair and warns if it is not.
TARGET_PAIR = ("KICH", "KIRC")

# The two digits at the end of a TCGA barcode identify the sample type.
SAMPLE_TYPE_LABELS = {
    "01": "Primary Solid Tumor",
    "05": "Additional New Primary",
    "06": "Metastatic",
    "11": "Solid Tissue Normal",
}

# Only these take part in the tumor-vs-tumor comparison. Adjacent normal kidney
# (-11) is much the same tissue whichever cohort filed it, so leaving it in
# turns "KICH vs KIRC" into a blend of that question and a tumor-vs-normal
# question -- and the cohorts carry very different normal fractions
# (KICH 25/91, KIRC 72/606), so the contamination is not even balanced.
TUMOR_SAMPLE_TYPES = {"01", "05"}

# ------------------------------------------------------------ clock panel ---

# The complete core circadian oscillator: positive arm (CLOCK/NPAS2 with
# ARNTL/ARNTL2), negative arm (PER1-3, CRY1-2), stabilising loop (NR1D1/2,
# RORA/B/C).
CLOCK_GENES = [
    "CLOCK", "ARNTL", "ARNTL2", "NPAS2", "NR1D1", "NR1D2", "CRY1",
    "CRY2", "RORA", "RORB", "RORC", "PER1", "PER2", "PER3",
]

# ------------------------------------------------------------- selection ---

TOP_N_GENES = 200          # genes kept from the differential-expression screen
FINAL_N_CANDIDATES = 100   # anchor-correlated genes kept for the final panel
N_ANCHORS_TO_DROP = 3      # weakest anchor genes discarded before correlating

FDR_CUTOFF = 0.05
LOG2FC_CUTOFF = 1.0

# ------------------------------------------------------------ evaluation ---

CV_FOLDS = 5
RANDOM_STATE = 42
MODELS = ["LinearSVC", "XGBoost"]

# --------------------------------------------------------- validation (E) ---

# Draws per null distribution. 300 gives empirical p-values down to ~0.003,
# which is finer than anything this dataset can support anyway.
N_NULL_DRAWS = 300

# Panel sizes for the accuracy-vs-k curve (E4).
CURVE_K_VALUES = [5, 10, 14, 25, 50, 111, 200, 500, 1000, 5000]

# Random panels for the E4 band at each k. Lower than N_NULL_DRAWS because the
# curve sweeps many k values; it only needs a band, not a p-value.
N_CURVE_DRAWS = 30


def ensure_dirs():
    for path in (RESULTS_DIR, CSV_DIR, PLOT_DIR, JSON_DIR):
        os.makedirs(path, exist_ok=True)


def csv_path(name):
    return os.path.join(CSV_DIR, name)


def plot_path(name):
    return os.path.join(PLOT_DIR, name)


def json_path(name):
    return os.path.join(JSON_DIR, name)


def pair_matrix_path(pair=None):
    """Path to a labelled sample-by-gene matrix for one cohort pair."""
    a, b = pair or TARGET_PAIR
    return csv_path(f"ExpressionLevels_{a}_{b}_Labelled.csv")
