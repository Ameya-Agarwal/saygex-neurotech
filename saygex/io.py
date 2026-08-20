"""Reading the raw cohorts and the labelled matrices every stage passes around.

Labelled matrices all share one layout: one row per sample, the TCGA barcode
as the index, ``Tumor_Type`` as the first column, genes after that. Keeping the
barcode as the index (rather than overwriting it with the label, as the first
version of this project did) is what makes sample identity traceable and keeps
concatenation safe -- duplicate index labels make pandas align unpredictably.
"""

import gzip
import json
import os

import pandas as pd

from . import config

LABEL_COLUMN = "Tumor_Type"
INDEX_NAME = "Sample"


def load_cohort(cohort):
    """Read one Xena HiSeqV2 matrix: genes down the rows, samples across."""
    path = os.path.join(config.GZIP_DIR, f"TCGA.{cohort}.sampleMap_HiSeqV2.gz")
    with gzip.open(path, "rt") as handle:
        return pd.read_csv(handle, sep="\t", index_col=0)


def load_all_cohorts(cohorts=None):
    """Load every cohort and verify they share one gene axis.

    A column-wise concat of mismatched frames silently pads with NaN, which
    would quietly poison every downstream t-test, so this fails loudly instead.
    """
    cohorts = cohorts or config.COHORTS
    frames = {c: load_cohort(c) for c in cohorts}

    reference = frames[cohorts[0]].index
    for cohort, df in frames.items():
        if not df.index.equals(reference):
            raise ValueError(
                f"{cohort} gene index does not match {cohorts[0]} "
                f"({len(df.index)} vs {len(reference)} genes). All cohorts must "
                "be the gene-level HiSeqV2 matrix, not HiSeqV2_exon."
            )
    return frames


def sample_type_code(barcode):
    """'TCGA-KN-8419-01' -> '01'."""
    parts = barcode.split("-")
    return parts[3][:2] if len(parts) >= 4 else ""


def build_metadata(frames):
    """Map every barcode to its cohort and sample type."""
    metadata = {}
    for cohort, df in frames.items():
        for barcode in df.columns:
            code = sample_type_code(barcode)
            metadata[barcode] = {
                "cohort": cohort,
                "sample_type": code,
                "sample_type_label": config.SAMPLE_TYPE_LABELS.get(code, "Unknown"),
                "is_tumor": code in config.TUMOR_SAMPLE_TYPES,
            }
    return metadata


def to_labelled(frame_list, metadata):
    """Stack gene-by-sample frames into one labelled sample-by-gene frame."""
    combined = pd.concat(frame_list, axis=1).transpose()
    combined.index.name = INDEX_NAME
    combined.insert(0, LABEL_COLUMN, [metadata[s]["cohort"] for s in combined.index])
    return combined


def load_labelled(path):
    """Return (features, labels) from a labelled matrix."""
    df = pd.read_csv(path, index_col=0)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"{path} has no '{LABEL_COLUMN}' column. Re-run stage 1."
        )
    return df.drop(columns=[LABEL_COLUMN]), df[LABEL_COLUMN]


def save_labelled(features, labels, path):
    """Write a feature block back out with its labels attached."""
    out = features.copy()
    out.insert(0, LABEL_COLUMN, labels)
    out.to_csv(path)
    return path


def write_json(obj, name):
    """Write a results document to results/json/<name>."""
    path = config.json_path(name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_jsonable)
    return path


def read_json(name):
    with open(config.json_path(name), encoding="utf-8") as handle:
        return json.load(handle)


def _jsonable(obj):
    """Let numpy scalars and arrays through json.dump."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"{type(obj)} is not JSON serialisable")
