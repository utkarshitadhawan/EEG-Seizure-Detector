"""Headroom check: do nonlinear models beat the shipped logistic regression on the same split and test set?

    python compare_models.py                   (or: BONN_CSV=data.csv python compare_models.py)

Writes results/model_comparison.json and results/fig7_model_comparison.png. Hyper-parameters are chosen by 3-fold cross-validation on
the training part only; the exact McNemar test compares each model with the shipped one on the test set.
"""
import json
import os
import time

import matplotlib
import numpy as np
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                  # noqa: E402

import eegseizure as z                                                           # noqa: E402
from eegseizure.data import load_bonn, test_split                                # noqa: E402

SEED = 0
os.makedirs("results", exist_ok=True)


Xr, y, _ = load_bonn()
FEAT = {"19": z.feats_batch(Xr), "36": z.feats_ext(Xr)}
itr, ite = test_split(y, SEED)                              # the project-wide split (identical to z.fit_cp)
yte = y[ite]


def lr():
    """Logistic-regression pipeline and its hyper-parameter grid (same as the shipped model)."""
    pipe = Pipeline([("s1", StandardScaler()), ("poly", "passthrough"), ("s2", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=5000))])
    return pipe, {"poly": ["passthrough", PolynomialFeatures(2, include_bias=False)], "lr__C": [0.1, 1, 10, 100]}


MODELS = {
    "Logistic regression, 19 features (shipped)": ("19",) + lr(),
    "Logistic regression, 36 features": ("36",) + lr(),
    "SVM (RBF), 36 features": ("36", Pipeline([("s", StandardScaler()), ("svm", SVC())]),
                               {"svm__C": [1, 10, 100], "svm__gamma": ["scale", 0.01]}),
    "Random forest, 36 features": ("36", RandomForestClassifier(300, n_jobs=-1, random_state=SEED), {}),
    "Gradient boosting, 19 features": ("19", HistGradientBoostingClassifier(max_iter=300, random_state=SEED),
                                       {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}),
    "Gradient boosting, 36 features": ("36", HistGradientBoostingClassifier(max_iter=300, random_state=SEED),
                                       {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}),
}

rng = np.random.default_rng(SEED)
bs = [rng.integers(0, len(yte), len(yte)) for _ in range(1000)]
R, PRED = {}, {}
for name, (fk, est, grid) in MODELS.items():
    t0 = time.time()
    F = FEAT[fk]
    gs = GridSearchCV(est, grid, cv=3, n_jobs=-1).fit(F[itr], y[itr]) if grid else est.fit(F[itr], y[itr])
    best = gs.best_estimator_ if grid else est
    pred = best.predict(F[ite])
    score = best.predict_proba(F[ite])[:, 1] if hasattr(best, "predict_proba") else best.decision_function(F[ite])
    ok = (pred == yte).astype(float)
    cv5 = cross_val_score(clone(best), F, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    R[name] = dict(accuracy=round(float(ok.mean()), 4),
                   accuracy_ci=[round(float(np.percentile([ok[i].mean() for i in bs], q)), 4) for q in (2.5, 97.5)],
                   sensitivity=round(float(((pred == 1) & (yte == 1)).sum() / (yte == 1).sum()), 4),
                   specificity=round(float(((pred == 0) & (yte == 0)).sum() / (yte == 0).sum()), 4),
                   roc_auc=round(float(roc_auc_score(yte, score)), 4), cv5_accuracy=round(float(cv5.mean()), 4),
                   errors=int((pred != yte).sum()), best_params={k: str(v) for k, v in (gs.best_params_.items() if grid else {})},
                   seconds=round(time.time() - t0))
    PRED[name] = pred
    print(f"{name:45s} acc {R[name]['accuracy']:.4f}  sens {R[name]['sensitivity']:.4f}  spec {R[name]['specificity']:.4f}  cv5 {R[name]['cv5_accuracy']:.4f}  ({R[name]['seconds']}s)", flush=True)

ref = "Logistic regression, 19 features (shipped)"
for name in R:                                           # exact McNemar test against the shipped model
    a, b = PRED[ref] == yte, PRED[name] == yte
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())  # shipped right / other wrong ; shipped wrong / other right
    R[name]["mcnemar_p_vs_shipped"] = 1.0 if n01 + n10 == 0 else round(float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue), 4)
    R[name]["only_shipped_correct"], R[name]["only_this_correct"] = n01, n10

best_nl = max((n for n in R if "Logistic" not in n), key=lambda n: R[n]["accuracy"])
gain = R[best_nl]["accuracy"] - R[ref]["accuracy"]
sig = R[best_nl]["mcnemar_p_vs_shipped"] < 0.05
verdict = (f"Best nonlinear model ({best_nl}) reaches {100 * R[best_nl]['accuracy']:.2f}% vs {100 * R[ref]['accuracy']:.2f}% for the shipped "
           f"logistic regression ({100 * gain:+.2f} points, McNemar p = {R[best_nl]['mcnemar_p_vs_shipped']}). "
           + ("The difference is statistically significant." if sig and gain > 0 else "The difference is not statistically significant."))
print("\n" + verdict)
json.dump(dict(models=R, verdict=verdict, best_nonlinear=best_nl, gain_points=round(100 * gain, 2), significant=bool(sig and gain > 0),
               n_test=int(len(yte)), seed=SEED), open("results/model_comparison.json", "w"), indent=2)

names = list(R)
short = [n.replace("Logistic regression", "LogReg").replace("Gradient boosting", "GradBoost").replace(", ", "\n") for n in names]
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
acc = np.array([100 * R[n]["accuracy"] for n in names])
err = np.array([[acc[i] - 100 * R[n]["accuracy_ci"][0], 100 * R[n]["accuracy_ci"][1] - acc[i]] for i, n in enumerate(names)]).T
ax[0].bar(range(len(names)), acc, yerr=err, capsize=4, color=["#2e7d32" if "shipped" in n else "#90a4ae" for n in names])
ax[0].axhline(99, color="r", ls="--", lw=1, label="99% target"); ax[0].set_ylim(min(acc) - 1.5, 100.2)
ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(short, fontsize=7); ax[0].set_ylabel("Test accuracy (%) with 95% CI")
ax[0].set_title("Headroom check: nonlinear vs shipped logistic regression"); ax[0].legend()
w = 0.38
ax[1].bar(np.arange(len(names)) - w / 2, [100 * R[n]["sensitivity"] for n in names], w, label="Sensitivity")
ax[1].bar(np.arange(len(names)) + w / 2, [100 * R[n]["specificity"] for n in names], w, label="Specificity")
ax[1].set_ylim(90, 100.3); ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(short, fontsize=7)
ax[1].set_ylabel("%"); ax[1].set_title("Sensitivity / specificity"); ax[1].legend()
plt.tight_layout(); plt.savefig("results/fig7_model_comparison.png", dpi=170)
