"""Gene-panel selectors.

Each selector is a scikit-learn transformer that picks a subset of gene
columns. Any selector whose ``fit`` looks at ``y`` MUST be refit inside each
cross-validation fold -- selecting genes on the full dataset and then
cross-validating on it scores a model that has already seen the test labels.

The project's method is ``AnchorPanelSelector``. It is deliberately written
with the anchor gene set as a *parameter* rather than hard-coded to the clock
genes, because the central claim -- that circadian genes make good anchors --
can only be tested by running the identical procedure with other anchors.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from . import config
from .stats import neg_log10, standardized_softmax, welch_ttest


def gene_indices(genes, wanted):
    """Positions of `wanted` within the gene list, preserving matrix order."""
    lookup = set(wanted)
    return [i for i, g in enumerate(genes) if g in lookup]


class FixedGeneSet(BaseEstimator, TransformerMixin):
    """A gene list fixed in advance. Touches no labels, so nothing can leak."""

    def __init__(self, idx):
        self.idx = idx

    def fit(self, X, y=None):
        self.idx_ = np.asarray(self.idx)
        return self

    def transform(self, X):
        return X[:, self.idx_]


class WelchSelectKBest(BaseEstimator, TransformerMixin):
    """Top-k genes by Welch p-value, computed on the training fold only.

    Deliberately not ``SelectKBest(f_classif)``: the ANOVA F-test assumes equal
    variances, the assumption 66-vs-534 samples break. Selecting with it costs
    several points of balanced accuracy here and disagrees with how every other
    stage ranks genes.
    """

    def __init__(self, k=config.TOP_N_GENES):
        self.k = k

    def fit(self, X, y):
        _, p_vals = welch_ttest(X[y == 0], X[y == 1])
        self.p_vals_ = p_vals
        self.idx_ = rank_by_pvalue(p_vals)[: self.k]
        return self

    def transform(self, X):
        return X[:, self.idx_]


class AnchorPanelSelector(BaseEstimator, TransformerMixin):
    """The project's method, with the anchor gene set left as a parameter.

    1. screen every gene by Welch p-value, keep the top ``top_n``
    2. weight the anchor genes by how discriminative each one is
    3. drop the ``n_drop`` weakest anchors
    4. rank the screened genes by weighted |correlation| with the kept anchors
    5. final panel = best ``n_final`` of those, plus the kept anchors

    Steps 1-3 consult the labels, so this must be refit inside each fold.
    """

    def __init__(self, anchor_idx, top_n=config.TOP_N_GENES,
                 n_final=config.FINAL_N_CANDIDATES, n_drop=config.N_ANCHORS_TO_DROP):
        self.anchor_idx = anchor_idx
        self.top_n = top_n
        self.n_final = n_final
        self.n_drop = n_drop

    def fit(self, X, y):
        _, p_vals = welch_ttest(X[y == 0], X[y == 1])
        screened = rank_by_pvalue(p_vals)[: self.top_n]
        self.idx_ = build_anchor_panel(
            X, p_vals, screened, np.asarray(self.anchor_idx),
            n_final=self.n_final, n_drop=self.n_drop,
        )
        return self

    def transform(self, X):
        return X[:, self.idx_]


# --------------------------------------------------------------- helpers ---


def rank_by_pvalue(p_vals):
    """Gene positions ordered most to least significant, NaNs last."""
    return np.argsort(np.nan_to_num(p_vals, nan=np.inf))


def build_anchor_panel(X, p_vals, screened, anchor_idx, n_final, n_drop):
    """Steps 2-5 of the method, factored out so it can be reused cheaply.

    The expensive part of the pipeline is the genome-wide Welch screen, and it
    does not depend on the anchors at all. Splitting it out lets the ablation
    reuse one screen per fold across hundreds of anchor draws.
    """
    anchor_idx = np.asarray(anchor_idx)

    # Keep the most discriminative anchors. Rank on p-value rather than on the
    # softmax weight: weak weights can underflow to an identical 0.0, which
    # would make the choice depend on ordering rather than on the data.
    ranked_anchors = anchor_idx[rank_by_pvalue(p_vals[anchor_idx])]
    keep_n = max(len(ranked_anchors) - n_drop, 1)
    keep = ranked_anchors[:keep_n]

    weights = standardized_softmax(neg_log10(p_vals[anchor_idx]))
    weight_map = dict(zip(anchor_idx.tolist(), weights))
    w_vec = np.array([weight_map[g] for g in keep.tolist()])

    kept = set(keep.tolist())
    candidates = np.array([i for i in screened if i not in kept], dtype=int)
    if candidates.size == 0:
        return keep

    block = np.corrcoef(X[:, np.concatenate([candidates, keep])], rowvar=False)
    corr = np.nan_to_num(block[: candidates.size, candidates.size:], nan=0.0)

    ordered = candidates[np.argsort(-(np.abs(corr) @ w_vec))]
    return np.concatenate([ordered[:n_final], keep])


def expression_profile(X):
    """Per-gene mean and standard deviation, used for matched sampling."""
    return X.mean(axis=0), X.std(axis=0)


def matched_random_indices(rng, means, sds, target_idx, n_bins=10, exclude=()):
    """Sample random genes matching `target_idx` on expression and variance.

    A plain random draw from all 20,530 genes is dominated by low-expression
    noise -- around 270 genes are flat zero everywhere. Beating that background
    only shows the anchors are better than junk. Matching each draw to the same
    mean-expression and variance bin as the gene it replaces makes the null
    "other genes that look like these", which is the comparison that matters.
    """
    mean_bin = _bin_ranks(means, n_bins)
    sd_bin = _bin_ranks(sds, n_bins)

    excluded = set(exclude)
    picks = []
    for gene in target_idx:
        pool = np.flatnonzero((mean_bin == mean_bin[gene]) & (sd_bin == sd_bin[gene]))
        pool = np.array([g for g in pool if g not in excluded and g not in picks])
        if pool.size == 0:  # Degenerate bin: fall back to an unmatched draw.
            pool = np.array([g for g in range(len(means))
                             if g not in excluded and g not in picks])
        picks.append(int(rng.choice(pool)))
    return np.array(picks)


def _bin_ranks(values, n_bins):
    """Assign each gene to a bin by rank, so bins are equally populated."""
    order = np.argsort(values)
    bins = np.empty(len(values), dtype=int)
    bins[order] = np.minimum((np.arange(len(values)) * n_bins) // len(values), n_bins - 1)
    return bins
