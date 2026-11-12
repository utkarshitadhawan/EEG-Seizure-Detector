"""Seizure classifier training, conformal abstention, explanations and model-file loading.

A "model bundle" (`cp`) is a plain dict: the fitted pipeline, the two conformal thresholds, a linear surrogate used
for explanations, the feature names and a few settings. It contains only scikit-learn objects and numbers.
"""
import os

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .constants import PLAIN, PLAIN_EXT


def _qhat(s, alpha):
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(np.sort(s)[k - 1])


def conformal_q(pc, yc, alpha):
    """Conformal thresholds (q0, q1) for the two classes from calibration probabilities `pc`, labels `yc` and error level `alpha`."""
    return _qhat(pc[yc == 0], alpha), _qhat(1 - pc[yc == 1], alpha)      # q0, q1


def cp_proba(cp, F):
    """P(seizure) for feature rows F (the bundle's own feature columns are selected first)."""
    F = np.asarray(F, float)
    if cp.get("idx") is not None:
        F = F[:, cp["idx"]]
    return cp["model"].predict_proba(F)[:, 1]


def cp_label(cp, p):
    """0 = non-seizure, 1 = uncertain, 2 = seizure."""
    in1, in0 = (1 - p) <= cp["q1"], p <= cp["q0"]
    lab = np.ones(len(p), int)
    lab[in1 & ~in0] = 2
    lab[in0 & ~in1] = 0
    return lab


def cp_predict(cp, F):
    """Conformal label (0 non-seizure, 1 uncertain, 2 seizure) for feature rows F."""
    return cp_label(cp, cp_proba(cp, F))


def top_features(cp, F, k=3):
    """Plain-language list of the k features that pushed the explanation model most towards 'seizure' for the windows in F."""
    F = np.asarray(F, float)
    if cp.get("idx") is not None:
        F = F[:, cp["idx"]]
    lin = cp["lin"]
    z = lin.named_steps["s"].transform(F).mean(0)
    c = lin.named_steps["lr"].coef_[0] * z
    order = [i for i in np.argsort(c)[::-1][:k] if c[i] > 0]
    return "; ".join(f"{'high' if z[i] > 0 else 'low'} {cp['names'][i]} ({c[i]:+.1f})" for i in order) or "-"


def metrics_basic(y, pred, proba=None):
    """Accuracy, sensitivity, specificity, precision and confusion counts (plus ROC-AUC when probabilities are given)."""
    y, pred = np.asarray(y), np.asarray(pred)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    m = dict(accuracy=(tp + tn) / len(y), sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
             precision=tp / max(tp + fp, 1), TN=tn, FP=fp, FN=fn, TP=tp, n=int(len(y)))
    if proba is not None:
        m["roc_auc"] = float(roc_auc_score(y, proba))
    return m


def fit_cp(X, y, idx=None, names=None, alpha=0.01, seed=0, ext=False):
    """Train LR (small grid search), calibrate conformal thresholds, build explanation surrogate.
    Returns (bundle, test_info)."""
    X, y = np.asarray(X, float), np.asarray(y)
    names = names or (PLAIN_EXT if ext else PLAIN)
    Xs = X if idx is None else X[:, idx]
    nm = [names[i] for i in (idx if idx is not None else range(X.shape[1]))]
    itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    pipe = Pipeline([("s1", StandardScaler()), ("poly", "passthrough"), ("s2", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=5000))])
    grid = {"poly": ["passthrough", PolynomialFeatures(2, include_bias=False)], "lr__C": [0.1, 1, 10, 100]}  # small grid
    gs = GridSearchCV(pipe, grid, cv=3, n_jobs=-1).fit(Xs[itr], y[itr])
    best = gs.best_estimator_                                  # refit on the whole training part
    # cross-conformal calibration: out-of-fold scores of the training part, so the deployed model can use ALL training data
    oof = cross_val_predict(clone(best), Xs[itr], y[itr], cv=StratifiedKFold(5, shuffle=True, random_state=seed + 1),
                            method="predict_proba", n_jobs=-1)[:, 1]
    q0, q1 = conformal_q(oof, y[itr], alpha)
    C = best.get_params()["lr__C"]
    lin = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=C, max_iter=5000))]).fit(Xs[itr], y[itr])
    cp = dict(model=best, q0=q0, q1=q1, alpha=alpha, lin=lin, idx=idx, names=nm, cal=(oof, y[itr]), ext=ext,
              params=dict(C=C, poly=str(best.get_params()["poly"])))
    proba = cp_proba(cp, X[ite])
    lab = cp_label(cp, proba)
    dec = lab != 1
    info = dict(itr=itr, ite=ite, proba=proba, lab=lab, y=y[ite], cv_score=float(gs.best_score_),
                plain=metrics_basic(y[ite], (proba >= .5).astype(int), proba),
                uncertain_pct=float(100 * (~dec).mean()),
                acc_decided=float(((lab[dec] == 2) == y[ite][dec]).mean()),
                errors_decided=int(((lab[dec] == 2) != y[ite][dec]).sum()), n_decided=int(dec.sum()),
                surrogate_agreement=float((lin.predict(Xs[ite]) == (proba >= .5)).mean()))
    return cp, info


def conformal_tradeoff(cp, proba, y, alphas=(0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)):
    """For several error levels alpha: (alpha, % of windows uncertain, accuracy on the decided windows) on a test set."""
    pc, yc = cp["cal"]
    rows = []
    for a in alphas:
        q0, q1 = conformal_q(pc, yc, a)
        c = dict(cp, q0=q0, q1=q1)
        lab = cp_label(c, proba)
        dec = lab != 1
        rows.append((a, 100 * (~dec).mean(), ((lab[dec] == 2) == y[dec]).mean() if dec.any() else np.nan))
    return rows


def find_model(name, base):
    """Load models/<name> from the usual places next to the program; None if the file is absent."""
    for d in (os.path.join(base, "models"), base, os.getcwd(), os.path.join(os.getcwd(), "models")):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return joblib.load(p)
    return None


def load_models(base):
    """-> dict(seg=..., img=..., mc=...) ; missing/unloadable models are None; errors listed in 'errors'."""
    out, errs = {}, []
    for key, fn in (("seg", "seizure_model_cp.joblib"), ("img", "image_model_cp.joblib"), ("mc", "chb_models.joblib")):
        try:
            out[key] = find_model(fn, base)
        except Exception as e:
            out[key] = None
            errs.append(f"{fn}: {e}")
    out["errors"] = errs
    return out
