"""Train and evaluate the seizure models on the Bonn / UCI data.

    python train_models.py                      # downloads the Kaggle copy of the data
    BONN_CSV=data.csv python train_models.py    # or use a local CSV

Writes (relative to the current folder):
    models/seizure_model_cp.joblib   single-channel segment model (also used by calibrated image mode)
    models/image_model_cp.joblib     amplitude-free model for images without a scale (uncalibrated image mode)
    results/metrics.json             every number reported in the README and in the GUI's training-evidence tab
    results/fig1..fig4 *.png         figures

Method in one paragraph: segments -> window features -> logistic regression (small grid search, 3-fold CV on the training part) ->
conformal thresholds from out-of-fold scores ("uncertain" zone). Everything is evaluated on a stratified 20% test set that is never
used for training or model selection. The image models are evaluated on simulated images of the same test segments.
"""
import json
import os
import sys

import joblib
import matplotlib
import numpy as np
import scipy
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, make_scorer, matthews_corrcoef,
                             precision_recall_curve, recall_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve, permutation_test_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                  # noqa: E402

import eegseizure as z                                                           # noqa: E402
from eegseizure.data import load_bonn                                            # noqa: E402

SEED = 0


# ------------------------------------------------------------------------------ 1. segment model
def fit_segment_model(X_raw, y):
    """Fit the model on the base (19) and the extended (36) feature sets and keep the one with the better
    cross-validation score on the TRAINING part. Returns (bundle, info, features, selection summary)."""
    F_base, F_ext = z.feats_batch(X_raw), z.feats_ext(X_raw)
    cp_b, info_b = z.fit_cp(F_base, y, seed=SEED)
    cp_e, info_e = z.fit_cp(F_ext, y, seed=SEED, ext=True)
    use_ext = info_e["cv_score"] > info_b["cv_score"]
    selection = dict(base_cv=round(info_b["cv_score"], 4), extended_cv=round(info_e["cv_score"], 4),
                     chosen="extended (36 features)" if use_ext else "base (19 features)",
                     base_test_accuracy=round(info_b["plain"]["accuracy"], 4),
                     extended_test_accuracy=round(info_e["plain"]["accuracy"], 4))
    print("feature selection:", selection)
    return (cp_e, info_e, F_ext, selection) if use_ext else (cp_b, info_b, F_base, selection)


def evaluate_segment_model(cp, info, F, y, selection):
    """Test-set metrics with bootstrap confidence intervals, cross-validation, baseline and shuffled-label checks."""
    y_test, proba = info["y"], info["proba"]
    pred = (proba >= 0.5).astype(int)
    rng = np.random.default_rng(SEED)
    boot = [rng.integers(0, len(y_test), len(y_test)) for _ in range(1000)]

    def ci(metric):                                                    # 95% bootstrap interval of a metric(y, pred)
        return [round(float(np.percentile([metric(y_test[i], pred[i]) for i in boot], q)), 4) for q in (2.5, 97.5)]

    sens = lambda a, b: recall_score(a, b, pos_label=1, zero_division=0)
    spec = lambda a, b: recall_score(a, b, pos_label=0, zero_division=0)
    acc = lambda a, b: float((a == b).mean())
    P = info["plain"]
    folds = StratifiedKFold(10, shuffle=True, random_state=SEED)
    scorers = {"accuracy": "accuracy", "sensitivity": "recall", "specificity": make_scorer(recall_score, pos_label=0), "roc_auc": "roc_auc"}
    cv = cross_validate(clone(cp["model"]), F, y, cv=folds, scoring=scorers, n_jobs=-1)
    baseline = cross_validate(DummyClassifier(strategy="most_frequent"), F, y, cv=folds, scoring="accuracy")["test_score"].mean()
    _, shuffled, perm_p = permutation_test_score(clone(cp["lin"]), F, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                                 n_permutations=100, scoring="accuracy", n_jobs=-1, random_state=SEED)
    metrics = dict(
        test=dict(accuracy=[round(P["accuracy"], 4), ci(acc)], sensitivity=[round(P["sensitivity"], 4), ci(sens)],
                  specificity=[round(P["specificity"], 4), ci(spec)], roc_auc=round(P["roc_auc"], 4),
                  mcc=round(float(matthews_corrcoef(y_test, pred)), 4), confusion={k: P[k] for k in ("TN", "FP", "FN", "TP")}, n_test=P["n"]),
        cv10={k: [round(float(cv["test_" + k].mean()), 4), round(float(cv["test_" + k].std()), 4)] for k in scorers},
        majority_baseline_acc=round(float(baseline), 4), shuffled_label_acc=round(float(shuffled.mean()), 4),
        permutation_p=float(perm_p), permutation_n=100, feature_selection=selection,
        conformal=dict(alpha=cp["alpha"], uncertain_pct=round(info["uncertain_pct"], 2), accuracy_on_decided=round(info["acc_decided"], 4),
                       errors=info["errors_decided"], n_decided=info["n_decided"], surrogate_agreement=round(info["surrogate_agreement"], 4)),
        params=cp["params"])
    return metrics, cv, pred


def plot_segment_figures(cp, info, F, y, cv, pred):
    """fig1 (six-panel performance overview), fig2 (explanation coefficients), fig3 (abstention trade-off)."""
    y_test, proba, P = info["y"], info["proba"], info["plain"]
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Non-seizure", "Seizure"], cmap="Blues", ax=ax[0, 0], colorbar=False)
    ax[0, 0].set_title("A. Confusion matrix (test set)")
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax[0, 1].plot(fpr, tpr, lw=2, label=f"AUC = {P['roc_auc']:.4f}")
    ax[0, 1].plot([0, 1], [0, 1], "k--")
    ax[0, 1].set(xlabel="False positive rate", ylabel="True positive rate", title="B. ROC curve")
    ax[0, 1].legend(loc="lower right")
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ax[0, 2].plot(recall, precision, lw=2, label=f"AP = {average_precision_score(y_test, proba):.4f}")
    ax[0, 2].set(xlabel="Recall", ylabel="Precision", title="C. Precision-recall curve")
    ax[0, 2].legend(loc="lower left")
    ax[1, 0].hist(proba[y_test == 0], 30, alpha=0.7, label="Non-seizure")
    ax[1, 0].hist(proba[y_test == 1], 30, alpha=0.7, label="Seizure", color="tab:red")
    ax[1, 0].axvline(0.5, color="k", ls="--")
    ax[1, 0].set(xlabel="Predicted P(seizure)", ylabel="Count", title="D. Score separation")
    ax[1, 0].legend()
    names = ["accuracy", "sensitivity", "specificity", "roc_auc"]
    ax[1, 1].boxplot([cv["test_" + k] for k in names], tick_labels=names)
    ax[1, 1].set_title("E. 10-fold cross-validation")
    sizes, train, valid = learning_curve(clone(cp["model"]), F, y, cv=5, train_sizes=np.linspace(0.1, 1, 6), n_jobs=-1, random_state=SEED)
    ax[1, 2].plot(sizes, train.mean(1), "o-", label="Train")
    ax[1, 2].plot(sizes, valid.mean(1), "o-", label="Validation")
    ax[1, 2].set(xlabel="Training samples", ylabel="Accuracy", title="F. Learning curve")
    ax[1, 2].legend()
    plt.tight_layout(); plt.savefig("results/fig1_performance.png", dpi=170); plt.close()

    w = cp["lin"].named_steps["lr"].coef_[0]
    order = np.argsort(w)
    plt.figure(figsize=(7, 6 if len(w) < 25 else 9))
    plt.barh(np.array(cp["names"])[order], w[order], color=np.where(w[order] > 0, "tab:red", "tab:blue"))
    plt.title("Explanation model: standardised coefficients\n(red = pushes towards 'seizure')")
    plt.tight_layout(); plt.savefig("results/fig2_coefficients.png", dpi=170); plt.close()

    trade = z.conformal_tradeoff(cp, proba, y_test)
    plt.figure(figsize=(6.5, 4.5))
    plt.plot([t[1] for t in trade], [100 * t[2] for t in trade], "o-")
    for alpha, uncertain, accuracy in trade:
        plt.annotate(f"alpha={alpha}", (uncertain, 100 * accuracy), fontsize=8, xytext=(3, 3), textcoords="offset points")
    plt.xlabel("Segments sent to review (%)")
    plt.ylabel("Accuracy on decided segments (%)")
    plt.title("Abstention trade-off")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("results/fig3_conformal_tradeoff.png", dpi=170); plt.close()


# ---------------------------------------------------------------------------------- 2. image models
def train_image_models(X_raw, y):
    """Uncalibrated image model: trained on simulated images that were rendered and digitised (so it learns digitisation noise).
    A second model trained on clean z-scored signals is only used for the comparison figure."""
    print("Rendering and digitising training images (about 1-2 min)...")
    F_digitised, fallbacks = z.make_image_dataset(X_raw, seed=SEED)
    F_clean = z.feats_batch(z.zscore_rows(X_raw))
    cp_img, info_img = z.fit_cp(F_digitised, y, idx=z.SF_IDX, seed=SEED)
    cp_clean, info_clean = z.fit_cp(F_clean, y, idx=z.SF_IDX, seed=SEED)
    assert (info_img["ite"] == info_clean["ite"]).all()                    # same test rows -> comparable
    cross = float(((z.cp_proba(cp_clean, F_digitised[info_img["ite"]]) >= 0.5).astype(int) == info_img["y"]).mean())
    summary = dict(accuracy_clean_trained_on_clean=round(info_clean["plain"]["accuracy"], 4),
                   accuracy_clean_trained_on_digitised=round(cross, 4),
                   accuracy_digitised_trained_on_digitised=round(info_img["plain"]["accuracy"], 4),
                   roc_auc=round(info_img["plain"]["roc_auc"], 4), sensitivity=round(info_img["plain"]["sensitivity"], 4),
                   specificity=round(info_img["plain"]["specificity"], 4), uncertain_pct=round(info_img["uncertain_pct"], 2),
                   accuracy_on_decided=round(info_img["acc_decided"], 4), digitisation_fallbacks=int(fallbacks), n_test=info_img["plain"]["n"])
    return cp_img, info_img, summary


def evaluate_calibrated_images(X_raw, y, cp_seg, test_idx):
    """Calibrated mode: fixed-gain images with a random known uV-per-pixel -> digitise -> multiply by the (possibly wrong) scale
    -> full segment model. `factor` = assumed scale / true scale (1.0 = exact)."""
    print("Evaluating calibrated image mode (fixed-gain images, known uV/pixel)...")
    rng = np.random.default_rng(SEED + 5)
    uv = rng.uniform(3, 12, len(test_idx))
    traces, fallbacks = [], 0
    for i, scale in zip(test_idx, uv):
        px = z.px_signal(z.render_fixed(X_raw[i], scale, rng))
        if px is None:                                                      # digitisation failed: fall back to the clean signal
            px, fallbacks = X_raw[i] / scale, fallbacks + 1
        traces.append(px * scale)                                           # microvolts at the exact scale
    traces = np.array(traces)
    results = {}
    for factor in (1.0, 0.7, 1.4, 2.0):
        proba = z.cp_proba(cp_seg, z.seg_feats(cp_seg, traces * factor))
        m = z.metrics_basic(y[test_idx], (proba >= 0.5).astype(int), proba)
        results[str(factor)] = {k: round(m[k], 4) for k in ("accuracy", "sensitivity", "specificity", "roc_auc")}
    labels = z.cp_label(cp_seg, z.cp_proba(cp_seg, z.seg_feats(cp_seg, traces)))
    decided = labels != 1
    results["conformal_exact_scale"] = dict(uncertain_pct=round(float(100 * (~decided).mean()), 2),
                                            accuracy_on_decided=round(float(((labels[decided] == 2) == y[test_idx][decided]).mean()), 4))
    return dict(scale_error_factor=results, digitisation_fallbacks=int(fallbacks), n_test=int(len(test_idx)),
                note="simulated fixed-gain images of Bonn test segments; factor = assumed / true uV-per-pixel")


def plot_image_figure(X_raw, y, info_img, summary, calibrated):
    """fig4: three digitisation examples, accuracy of the image modes, ROC of the uncalibrated model, effect of a wrong scale."""
    scale = calibrated["scale_error_factor"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    rng = np.random.default_rng(3)
    test_idx = info_img["ite"]
    for k, i in enumerate([test_idx[info_img["y"] == 0][0], test_idx[info_img["y"] == 1][0], test_idx[info_img["y"] == 1][1]]):
        img = z.render_trace(X_raw[i], rng)
        lanes, _ = z.digitize(img, 1)
        ax[0, k].imshow(img)
        ax[0, k].plot(lanes[0]["ypix"], color="red", lw=0.9)
        ax[0, k].set_title(("Seizure" if y[i] else "Non-seizure") + " (red = digitised trace)")
        ax[0, k].axis("off")
    ax[1, 0].bar(["uncalibrated\n(clean)", "uncalibrated\n(clean->digitised)", "uncalibrated\n(digitised-trained)", "CALIBRATED\n(exact scale)"],
                 [100 * summary["accuracy_clean_trained_on_clean"], 100 * summary["accuracy_clean_trained_on_digitised"],
                  100 * summary["accuracy_digitised_trained_on_digitised"], 100 * scale["1.0"]["accuracy"]],
                 color=["#9ecae1", "#fc9272", "#fdd835", "#74c476"])
    ax[1, 0].set_ylim(80, 100.5)
    ax[1, 0].set_ylabel("Test accuracy (%)")
    ax[1, 0].set_title("Image mode: why the amplitude scale matters")
    ax[1, 0].tick_params(axis="x", labelsize=7)
    fpr, tpr, _ = roc_curve(info_img["y"], info_img["proba"])
    ax[1, 1].plot(fpr, tpr, lw=2, label=f"AUC={summary['roc_auc']:.4f}")
    ax[1, 1].plot([0, 1], [0, 1], "k--")
    ax[1, 1].set(title="Uncalibrated image model: ROC", xlabel="FPR", ylabel="TPR")
    ax[1, 1].legend()
    factors = ["0.7", "1.0", "1.4", "2.0"]
    ax[1, 2].bar([f"x{f}" for f in factors], [100 * scale[f]["accuracy"] for f in factors], color="#74c476")
    ax[1, 2].set_ylim(80, 100.5)
    ax[1, 2].set_title("Calibrated mode vs error in the uV/pixel entry")
    ax[1, 2].set_ylabel("Test accuracy (%)")
    ax[1, 2].set_xlabel("assumed / true scale")
    plt.tight_layout(); plt.savefig("results/fig4_image_mode.png", dpi=170); plt.close()


# -------------------------------------------------------------------------------------------- main
def main():
    """Train and evaluate everything; write models/, results/metrics.json and the figures."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    X_raw, y, data_sha = load_bonn()
    print("data", X_raw.shape, "seizure fraction", y.mean())

    cp, info, F, selection = fit_segment_model(X_raw, y)
    joblib.dump(cp, "models/seizure_model_cp.joblib")
    seg_metrics, cv, pred = evaluate_segment_model(cp, info, F, y, selection)
    plot_segment_figures(cp, info, F, y, cv, pred)

    cp_img, info_img, img_summary = train_image_models(X_raw, y)
    joblib.dump(cp_img, "models/image_model_cp.joblib")
    calibrated = evaluate_calibrated_images(X_raw, y, cp, info_img["ite"])
    plot_image_figure(X_raw, y, info_img, img_summary, calibrated)

    metrics = dict(segment_model=seg_metrics, image_model=img_summary, calibrated_image=calibrated,
                   versions=dict(python=sys.version.split()[0], sklearn=sklearn.__version__, numpy=np.__version__,
                                 scipy=scipy.__version__, seed=SEED, data_sha256=data_sha))
    json.dump(metrics, open("results/metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
