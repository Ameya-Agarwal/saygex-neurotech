"""Statistics helpers shared by every stage."""

import numpy as np
from scipy import stats as scipy_stats


def welch_ttest(group1, group2):
    """Per-column two-sample t-test that does not assume equal variances.

    The cohorts differ enormously in size (KICH ~66 tumors, KIRC ~534) and in
    spread, which is exactly where Student's equal-variance t-test misstates
    the p-value. Genes constant in both groups come back NaN so callers can
    sort them to the bottom rather than treat them as significant.
    """
    t_stat, p_vals = scipy_stats.ttest_ind(group1, group2, equal_var=False)
    t_stat = np.asarray(t_stat, dtype=float)
    p_vals = np.asarray(p_vals, dtype=float)

    # Underflow to exactly 0.0 would make -log10(p) infinite.
    p_vals = np.where(p_vals == 0.0, np.finfo(float).tiny, p_vals)
    return t_stat, p_vals


def benjamini_hochberg(p_vals):
    """BH FDR-adjusted p-values (q-values). NaN in, NaN out.

    Every screen here tests ~20,500 genes at once, so an uncorrected p < 0.05
    would return roughly a thousand genes by chance alone.
    """
    p_vals = np.asarray(p_vals, dtype=float)
    q_vals = np.full(p_vals.shape, np.nan)

    tested = ~np.isnan(p_vals)
    values = p_vals[tested]
    n = values.size
    if n == 0:
        return q_vals

    order = np.argsort(values)
    ranked = values[order] * n / np.arange(1, n + 1)
    # Step-up: enforce monotonicity from the largest p-value downwards.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]

    adjusted = np.empty(n)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    q_vals[tested] = adjusted
    return q_vals


def neg_log10(p_vals):
    """-log10(p), NaNs preserved."""
    p_vals = np.asarray(p_vals, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log10(p_vals)


def softmax(x, temperature=1.0):
    x = np.asarray(x, dtype=float) / temperature
    x = x - np.max(x)  # Stability shift.
    e = np.exp(x)
    return e / e.sum()


def standardized_softmax(scores, temperature=1.0):
    """Softmax over z-scored inputs, so no single entry absorbs all the weight.

    The inputs are -log10(p) values that routinely span 0 to 200+. A plain
    softmax over that range puts ~1.0 on the single best gene and underflows
    the rest to exactly 0.0, which both collapses the weighted combination to
    one gene and makes any tie-break among the zeros depend on dict ordering.
    Standardising first preserves the ranking but restores discrimination.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    spread = scores.std()
    if spread == 0 or not np.isfinite(spread):
        return np.full(scores.shape, 1.0 / scores.size)
    return softmax((scores - scores.mean()) / spread, temperature)


def empirical_p(observed, null_samples, tail="greater"):
    """Empirical p-value with the standard +1 correction.

    Adding one to numerator and denominator keeps p from ever being exactly 0,
    which would claim more resolution than the number of draws supports.
    """
    null_samples = np.asarray(null_samples, dtype=float)
    null_samples = null_samples[~np.isnan(null_samples)]
    if null_samples.size == 0:
        return float("nan")
    if tail == "greater":
        extreme = int((null_samples >= observed).sum())
    else:
        extreme = int((null_samples <= observed).sum())
    return (extreme + 1) / (null_samples.size + 1)
