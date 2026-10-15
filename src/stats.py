"""Classification metrics, agreement, and clustering statistics."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def centroid_features(usage_vecs, anchor_vecs_by_comp, comps):
    """Cosine similarity of each usage to each component centroid -> (n, n_comps)."""
    cents = np.stack([anchor_vecs_by_comp[c].mean(0) for c in comps])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12
    u = usage_vecs / (np.linalg.norm(usage_vecs, axis=1, keepdims=True) + 1e-12)
    return u @ cents.T


def expected_calibration_error(probs, y, n_bins=10):
    conf = probs.max(1)
    pred = probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return ece


def brier_multiclass(probs, y):
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1
    return ((probs - onehot) ** 2).sum(1).mean()


def crossfit_classifier(X, y, n_classes, k, seed):
    """Out-of-fold probabilities from a multinomial logistic model with stratified k-fold cross-fitting."""
    counts = np.bincount(y, minlength=n_classes)
    present = counts[counts > 0]
    k_eff = int(max(2, min(k, present.min())))
    probs = np.zeros((len(y), n_classes))
    skf = StratifiedKFold(n_splits=k_eff, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        probs[np.ix_(te, clf.classes_)] = clf.predict_proba(X[te])
    return probs, k_eff


def classification_report(probs, y, n_classes):
    pred = probs.argmax(1)
    return {
        "macro_f1": f1_score(y, pred, average="macro", labels=list(range(n_classes)), zero_division=0),
        "ece": expected_calibration_error(probs, y),
        "brier": brier_multiclass(probs, y),
        "accuracy": (pred == y).mean(),
        "confusion": confusion_matrix(y, pred, labels=list(range(n_classes))).tolist(),
    }


def krippendorff_alpha_nominal(a, b):
    """Krippendorff's alpha for two coders, nominal data, no missing values."""
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    o = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        o[idx[x], idx[y]] += 1
        o[idx[y], idx[x]] += 1
    n_c = o.sum(1)
    n = n_c.sum()
    if n <= 1:
        return np.nan
    d_o = (n - np.trace(o)) / n
    d_e = (n * n - (n_c ** 2).sum()) / (n * (n - 1))
    return 1.0 - d_o / d_e if d_e > 0 else np.nan


def icc1(values, groups):
    """ICC(1) from one-way ANOVA with unequal group sizes (values: 0/1 or continuous)."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    labs, inv = np.unique(groups, return_inverse=True)
    k, N = len(labs), len(values)
    if k < 2 or N <= k:
        return np.nan
    n_i = np.bincount(inv)
    means = np.bincount(inv, weights=values) / n_i
    grand = values.mean()
    ssb = (n_i * (means - grand) ** 2).sum()
    ssw = ((values - means[inv]) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return (msb - msw) / denom if denom > 0 else np.nan
