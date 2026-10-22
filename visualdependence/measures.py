"""The measures, on score drops d(x, O, f) = s_f(x) - s_f(x_O). No model, no I/O."""
from __future__ import annotations
import numpy as np


def null_scale(K: int) -> float:
    """sqrt(K^2 - 1) / K: gives the null contrast the null variance of the numerator itself."""
    return float(np.sqrt(K * K - 1.0) / K)


def null(d_control: np.ndarray) -> np.ndarray:
    """The scaled null tilde_nu_i = sqrt(K^2 - 1)/K * (d(x, c_i, f) - 1/(K-1) sum_{k != i} d(x, c_k, f)).

    d_control: (..., K) control drops."""
    d = np.asarray(d_control, float)
    K = d.shape[-1]
    if K < 2:
        raise ValueError("the null needs K >= 2 controls")
    others = (d.sum(axis=-1, keepdims=True) - d) / (K - 1)
    return (d - others) * null_scale(K)


def std_null(d_control_all: np.ndarray) -> float:
    """The standard deviation of tilde_nu pooled over all case-finding pairs and controls."""
    return float(null(d_control_all).reshape(-1).std(ddof=1))


def occ(d_organ: np.ndarray, d_control: np.ndarray, std_null: float) -> np.ndarray:
    """occ(x, f) = (d_organ(x, f) - d_control(x, f)) / std_null, with d_control the mean drop over the K controls."""
    if std_null <= 0:
        raise ValueError("std_null must be positive")
    return (np.asarray(d_organ, float) - np.asarray(d_control, float).mean(axis=-1)) / std_null


def occ_null(d_control: np.ndarray, std_null: float) -> np.ndarray:
    """occ^null_i(x, f) = tilde_nu_i(x, f) / std_null: the same quantity with control c_i in place of the organ."""
    return null(d_control) / std_null


def pos_terms(occ_shift: np.ndarray, occ_null_shift: np.ndarray):
    """Numerator and denominator of pos for one case-finding pair:
    std_delta(occ(x^(delta), f)) and sqrt(1/K sum_i std_delta^2(occ^null_i(x^(delta), f))).

    occ_shift: (n_delta,); occ_null_shift: (n_delta, K)."""
    num = np.asarray(occ_shift, float).std(ddof=1)
    den = np.sqrt((np.asarray(occ_null_shift, float).std(axis=0, ddof=1) ** 2).mean())
    return num, den


def R_O_terms(occ_f: np.ndarray, occ_null_f: np.ndarray):
    """Numerator and denominator of R_O for one case: std_{f in F_O}(occ(x, f)) and
    sqrt(1/K sum_i std_{f in F_O}^2(occ^null_i(x, f))).

    occ_f: (|F_O|,); occ_null_f: (|F_O|, K)."""
    occ_f, occ_null_f = np.asarray(occ_f, float), np.asarray(occ_null_f, float)
    tol = 1e-9 * max(np.abs(occ_f).max(), np.abs(occ_null_f).max(), 1e-300)   # identical scores: spread 0, not rounding error
    num = occ_f.std(ddof=1); den = np.sqrt((occ_null_f.std(axis=0, ddof=1) ** 2).mean())
    return (0.0 if num < tol else num), (0.0 if den < tol else den)



def pooled_ratio(terms) -> float:
    """Pool per-case (numerator, denominator) terms into one ratio of pooled standard deviations."""
    num = np.sqrt(np.mean(np.array([n for n, _ in terms], float) ** 2))
    den = np.sqrt(np.mean(np.array([d for _, d in terms], float) ** 2))
    return float(num / den) if den > 0 else float("nan")


def bootstrap_mean_ci(values, n_boot: int = 5000, seed: int = 0):
    """Mean and 95% bootstrap interval over cases."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    boots = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)]
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def bootstrap_diff_ci(a, b, n_boot: int = 5000, seed: int = 0):
    """mean(a) - mean(b) and its 95% bootstrap interval, each group resampled independently."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = [a[rng.integers(0, a.size, a.size)].mean() - b[rng.integers(0, b.size, b.size)].mean() for _ in range(n_boot)]
    return float(a.mean() - b.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def bootstrap_G_f_ci(occ_y1, occ_y0, s_y1, s_y0, std_null, n_boot: int = 5000, seed: int = 0):
    """G_f = (occ_bar(y=1) - occ_bar(y=0)) / Delta_f and its 95% bootstrap interval; numerator and denominator
    come from the same draw, and draws whose Delta_f is not positive are dropped."""
    o1, o0 = np.asarray(occ_y1, float), np.asarray(occ_y0, float)
    s1, s0 = np.asarray(s_y1, float), np.asarray(s_y0, float)
    rng = np.random.default_rng(seed)
    delta = (s1.mean() - s0.mean()) / std_null
    point = float((o1.mean() - o0.mean()) / delta) if delta > 0 else float("nan")
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, o1.size, o1.size); j = rng.integers(0, o0.size, o0.size)
        d = (s1[i].mean() - s0[j].mean()) / std_null
        if d > 0:
            boots.append((o1[i].mean() - o0[j].mean()) / d)
    if len(boots) < 2:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def bootstrap_ratio_ci(terms, n_boot: int = 5000, seed: int = 0):
    """pooled_ratio and its 95% bootstrap interval over cases."""
    terms = list(terms)
    point = pooled_ratio(terms)
    if len(terms) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [pooled_ratio([terms[i] for i in rng.integers(0, len(terms), len(terms))]) for _ in range(n_boot)]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def auroc(scores, labels) -> float:
    """Detection AUROC of s_f against y_f; NaN if one class is missing."""
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels, int)
    return float(roc_auc_score(labels, np.asarray(scores, float))) if 0 < labels.sum() < labels.size else float("nan")


def spearman_ci(x, y, n_boot: int = 5000, seed: int = 0):
    """Spearman correlation over findings, its 95% bootstrap interval over findings, p and n."""
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    r, p = spearmanr(x, y)
    rng = np.random.default_rng(seed); boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if len(set(i)) >= 3:
            v = spearmanr(x[i], y[i])[0]
            if np.isfinite(v):
                boots.append(v)
    return dict(rho=float(r), ci=tuple(float(v) for v in np.percentile(boots, [2.5, 97.5])), p=float(p), n=len(x))
