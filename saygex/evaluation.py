"""Cross-validation harness.

One rule holds everywhere in here: anything that consults the labels is refit
inside the training fold. That includes gene selection, scaling, and the class
weighting. The saved feature panels on disk were built from all 600 samples, so
scoring a model on them measures a model that has already seen its own test
set; these functions rebuild the selection per fold instead.
"""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from . import config

METRICS = ("balanced_accuracy", "roc_auc", "accuracy")


def scale_pos_weight(y):
    """sum(negatives) / sum(positives), per the XGBoost docs.

    XGBoost scales the gradient of the *positive* class (label 1), so this has
    to be negatives-over-positives. Using max/min instead gives a value above 1
    whichever class is larger, upweighting the majority whenever the majority
    encodes as 1 -- the opposite of rebalancing. LabelEncoder sorts
    alphabetically, so KIRC (the majority) becomes label 1 here.
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return n_neg / n_pos if n_pos else 1.0


def build_model(name, y_train, random_state=config.RANDOM_STATE):
    """A fresh estimator, class-weighted from the training fold only."""
    if name == "LinearSVC":
        return LinearSVC(class_weight="balanced", max_iter=10000,
                         dual="auto", random_state=random_state)
    if name == "XGBoost":
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            n_estimators=100, random_state=random_state, tree_method="hist",
            n_jobs=1, scale_pos_weight=scale_pos_weight(y_train),
        )
    raise ValueError(f"Unknown model: {name}")


def folds(X, y, n_splits=config.CV_FOLDS, random_state=config.RANDOM_STATE):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(X, y))


def score_panel(X, y, panel_idx, train, test, model_name):
    """Fit on the training fold's genes and score the held-out fold."""
    X_train = X[np.ix_(train, panel_idx)]
    X_test = X[np.ix_(test, panel_idx)]
    y_train, y_test = y[train], y[test]

    # Drop genes constant within this training fold; ~270 are flat zero
    # genome-wide and a constant column is information-free anyway.
    varying = X_train.std(axis=0) > 0
    if varying.any():
        X_train, X_test = X_train[:, varying], X_test[:, varying]

    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    model = build_model(model_name, y_train)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        ranking = model.predict_proba(X_test)[:, 1]
    else:
        ranking = model.decision_function(X_test)

    return {
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, ranking),
        "accuracy": accuracy_score(y_test, y_pred),
    }


def cross_validate_selector(X, y, selector_factory, model_name, fold_list=None):
    """Leak-free CV: the selector is rebuilt and refit on every training fold."""
    fold_list = fold_list if fold_list is not None else folds(X, y)
    per_fold = []
    for train, test in fold_list:
        selector = selector_factory()
        selector.fit(X[train], y[train])
        panel = np.asarray(selector.idx_)
        per_fold.append(score_panel(X, y, panel, train, test, model_name))
    return _aggregate(per_fold)


def cross_validate_panel(X, y, panel_idx, model_name, fold_list=None):
    """CV for a label-blind panel, where the gene list is fixed in advance."""
    fold_list = fold_list if fold_list is not None else folds(X, y)
    per_fold = [score_panel(X, y, panel_idx, train, test, model_name)
                for train, test in fold_list]
    return _aggregate(per_fold)


def _aggregate(per_fold):
    out = {}
    for metric in METRICS:
        values = np.array([f[metric] for f in per_fold])
        out[metric] = values.mean()
        out[f"{metric}_std"] = values.std()
        out[f"{metric}_folds"] = values
    return out


def one_sample_worth(y):
    """How much balanced accuracy one minority-class sample is worth.

    With 66 KICH samples a single misclassification moves balanced accuracy by
    1/(2*66) = 0.0076. Gaps smaller than this are one sample, not a finding.
    """
    counts = np.bincount(y)
    return 1.0 / (2 * counts.min())


def format_row(label, scores, width=34):
    return (f"{label:<{width}} "
            f"{scores['balanced_accuracy']:9.4f} "
            f"{scores['roc_auc']:9.4f} "
            f"{scores['accuracy']:8.4f}")


def header(width=34):
    return (f"{'panel':<{width}} {'bal.acc':>9s} {'ROC-AUC':>9s} {'plain':>8s}\n"
            + "-" * (width + 29))
