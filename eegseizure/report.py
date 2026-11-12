"""Turning analysis results into things a person reads: verdict, event table, figure, CSV, HTML report, and the training report.

The GUI, the command-line tool and the reference-output generator all use these functions, so they always agree.
"""
import base64
import datetime
import hashlib
import io
import json
import os

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from . import __version__
from .events import find_events, movavg
from .minimarkdown import to_html
from .models import top_features

APP_NAME = "EEG Seizure Detector"
DISCLAIMER = ("Research prototype - not a medical device. Not for diagnosis or clinical decisions. "
              "Screening aid only; a qualified clinician must review the EEG.")
EVENT_COLUMNS = ("#", "Start (s)", "End (s)", "Duration (s)", "Peak prob.", "Mean prob.", "Main drivers")
RED, ORANGE, GREEN = "#c62828", "#ef6c00", "#2e7d32"


# ------------------------------------------------------------------------------------------ decisions
def decide(r, thr=0.5, kmin=1):
    """Events = runs of at least `kmin` consecutive windows whose probability is >= thr.
    (Uncalibrated image analyses enforce a minimum of 3 windows, see analysis.analyse_image.)  Returns (events, kmin used)."""
    kmin = max(int(kmin), r.get("kmin_min", 1))
    return find_events(r["p"] >= thr, kmin), kmin


def event_rows(r, events):
    """One table row per event: start, end, duration, peak/mean probability and the main driving features."""
    step, win, rows = r["step"], r["win"], []
    for k, (s, e) in enumerate(events, 1):
        seg = r["p"][s:e + 1]
        drivers = top_features(r["cp"], r["feats"][r["best"]][s:e + 1]) if r.get("cp") is not None else "-"
        rows.append((k, f"{s * step:.1f}", f"{e * step + win:.1f}", f"{(e - s) * step + win:.1f}",
                     f"{seg.max():.3f}", f"{seg.mean():.3f}", drivers))
    return rows


def summarise(r, thr=0.5, kmin=1):
    """Everything the user sees as text: verdict banner, colour, statistics, notes and the event table."""
    events, kmin = decide(r, thr, kmin)
    p, n, step = r["p"], r["n"], r["step"]
    flag = p >= thr
    nunc = int((r["lab"] == 1).sum()) if r["lab"] is not None else 0
    uncal = r["kind"] == "img" and not r.get("calibrated")
    where = " IN IMAGE" if r["kind"] == "img" else ""
    if events and uncal:
        level, text, color = "possible", "POSSIBLE SEIZURE-LIKE ACTIVITY IN IMAGE (uncalibrated - low reliability)", ORANGE
    elif events:
        level, text, color = "seizure", f"SEIZURE ACTIVITY DETECTED{where}", RED
    elif nunc:
        level, text, color = "uncertain", f"NO CONFIDENT SEIZURE - {nunc} UNCERTAIN WINDOW(S) TO REVIEW", ORANGE
    else:
        level, text, color = "none", f"NO SEIZURE DETECTED{where}", GREEN
    notes = list(r["warn"])
    if r["kind"] == "img":
        if r.get("calibrated"):
            notes.append(f"Image mode, CALIBRATED ({len(r['names'])} channel(s), amplitudes converted to uV): screening only - photos/scans can mislead.")
        else:
            notes.append(f"Image mode, UNCALIBRATED ({len(r['names'])} channel(s)): amplitude unknown, so reliability is low "
                         "(at least 3 consecutive windows are required). Enter 'uV per pixel' for accurate mode.")
    elif r["kind"] == "mc":
        notes.append("Multichannel model is experimental (patient-independent, pediatric training data); expect false alarms.")
        if r["pre"] is not None:
            warn_eps = find_events(movavg(r["pre"], 30) >= r["mc"].get("thr_pre", 0.8), 1)
            notes.append(f"Early-warning episodes (pre-seizure score): {len(warn_eps)}")
    elif r["rows"]:
        notes.append("Rows treated as independent 1.03 s segments (time axis is artificial).")
    elif len(r["names"]) > 1:
        notes.append(f"{len(r['names'])} channels analysed (flag if any channel exceeds threshold); plotted: {r['names'][r['best']]}.")
    if kmin > 1:
        notes.append(f"Minimum consecutive windows in use: {kmin}.")
    stats = (f"File: {os.path.basename(r['path'])}    Windows: {n}    Span: {n * step:.1f} s    Flagged: {int(flag.sum())} "
             f"({100 * flag.mean():.1f}%)    Events: {len(events)}    Max prob.: {p.max():.3f}    Uncertain: {nunc}")
    return dict(events=events, kmin=kmin, level=level, verdict=text, color=color, stats=stats, notes=notes,
                text=stats + ("\n" + "\n".join(notes) if notes else ""), rows=event_rows(r, events), n_uncertain=nunc)


def results_table(r, thr=0.5, kmin=1):
    """Per-window table (the CSV export): time span, probability, flag, event membership, status and extra scores."""
    events, _ = decide(r, thr, kmin)
    n, step, win = r["n"], r["step"], r["win"]
    in_event = np.zeros(n, int)
    for s, e in events:
        in_event[s:e + 1] = 1
    df = pd.DataFrame({"window": np.arange(n), "start_s": np.arange(n) * step, "end_s": np.arange(n) * step + win,
                       "probability": r["p"], "flagged": (r["p"] >= thr).astype(int), "in_event": in_event})
    if r["lab"] is not None:
        df["status"] = np.array(["non-seizure", "uncertain", "seizure"])[r["lab"].astype(int)]
    if r["pre"] is not None:
        df["pre_seizure_score"] = r["pre"]
    if len(r["names"]) > 1 and r["kind"] != "mc":
        for name, q in zip(r["names"], r["probs"]):
            df["p_" + name] = q
    return df


# ------------------------------------------------------------------------------------------------ figure
def plot_result(fig, r, thr, events):
    """Draw the result into a matplotlib Figure: signal (or image with traces), probability curve, optional pre-seizure score."""
    step, win = r["step"], r["win"]
    fig.clear()
    has_pre = r["kind"] == "mc" and r["pre"] is not None
    axes = fig.subplots(3 if has_pre else 2, 1, sharex=True)
    a1, a2 = axes[0], axes[1]
    if r["kind"] == "img":
        height, width = r["image"].shape[:2]
        a1.imshow(r["image"], extent=[-r["x0"] / r["rate"], (width - r["x0"]) / r["rate"], height, 0], aspect="auto")
        for lane in r["lanes"]:
            a1.plot(np.arange(len(lane["ypix"])) / r["rate"], lane["ypix"], color="#e53935", lw=0.7, alpha=0.8)
        a1.set_xlim(0, r["seconds"])
        a1.set_yticks([])
        a1.set_title("Image with digitised traces (red); seizure windows shaded red, uncertain orange")
    else:
        sig = r["sigs"][r["best"]]
        stride = max(1, len(sig) // 20000)                                   # keep the plot light for long recordings
        a1.plot(np.arange(0, len(sig), stride) / r["fs"], sig[::stride], lw=0.6, color="#1f4e79")
        a1.set_ylabel("Amplitude (uV)")
        a1.set_title("EEG signal: seizure windows (red), uncertain windows (orange)")
    centres = np.arange(r["n"]) * step + win / 2
    a2.plot(centres, r["p"], lw=1, color="#333", drawstyle="steps-mid")
    a2.axhline(thr, color="k", ls="--", lw=0.8)
    a2.set_ylim(-0.02, 1.02)
    a2.set_ylabel("P(seizure)")
    if has_pre:
        axes[2].plot(centres, movavg(r["pre"], 30), color="#6a1b9a", lw=1)
        axes[2].axhline(r["mc"].get("thr_pre", 0.8), color="k", ls="--", lw=0.8)
        axes[2].set_ylim(-0.02, 1.02)
        axes[2].set_ylabel("Pre-seizure\nscore (1 min avg)")
    axes[-1].set_xlabel("Time (s)")
    spans = [(s * step, e * step + win, "#d62728", 0.25) for s, e in events]
    if r["lab"] is not None:
        spans += [(s * step, e * step + win, "#ff9800", 0.3) for s, e in find_events(r["lab"] == 1, 1)]
    for ax in axes[:2]:
        for t0, t1, color, alpha in spans:
            ax.axvspan(t0, t1, color=color, alpha=alpha, lw=0)
    fig.tight_layout()


# ------------------------------------------------------------------------------------------- HTML / files
def html_report(r, fig, thr=0.5, kmin=1):
    """Self-contained HTML report: verdict, statistics, figure, event table, checksum of the input file and the disclaimer."""
    s = summarise(r, thr, kmin)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    sha = hashlib.sha256(open(r["path"], "rb").read()).hexdigest() if os.path.isfile(r["path"]) else "n/a"
    esc = lambda t: str(t).replace("&", "&amp;").replace("<", "&lt;")
    rows = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in s["rows"])
    head = "".join(f"<th>{c}</th>" for c in EVENT_COLUMNS)
    return f"""<html><head><meta charset="utf-8"><title>EEG report</title></head>
<body style="font-family:Arial;max-width:980px;margin:auto">
<h2>EEG seizure screening report</h2>
<p>Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by {APP_NAME} v{__version__}<br>File: {esc(os.path.basename(r['path']))}<br>SHA-256: {sha}</p>
<h3 style="color:{s['color']}">{esc(s['verdict'])}</h3>
<pre>{esc(s['text'])}</pre>
<img style="width:100%" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">
<table border="1" cellpadding="4" style="border-collapse:collapse"><tr>{head}</tr>{rows}</table>
<p style="color:#a33"><b>{DISCLAIMER}</b></p></body></html>"""


def write_outputs(r, prefix, thr=0.5, kmin=1):
    """Write <prefix>_results.csv, <prefix>_report.html and <prefix>_figure.png (what the GUI exports). Returns the summary."""
    events, _ = decide(r, thr, kmin)
    fig = Figure(figsize=(11, 6.5))
    plot_result(fig, r, thr, events)
    results_table(r, thr, kmin).to_csv(prefix + "_results.csv", index=False)
    open(prefix + "_report.html", "w", encoding="utf-8").write(html_report(r, fig, thr, kmin))
    fig.savefig(prefix + "_figure.png", dpi=110)
    return summarise(r, thr, kmin)


# ------------------------------------------------------------------------------------- training report
def _load(base, rel):
    p = os.path.join(base, rel)
    return json.load(open(p)) if os.path.isfile(p) else None


def training_report_markdown(base="."):
    """Readable summary of everything the training scripts produced (reads results/*.json; figures are referenced by path)."""
    m, cmp_, chb = (_load(base, f"results/{f}.json") for f in ("metrics", "model_comparison", "chb_metrics"))
    if m is None:
        return "# Training report\n\nNo training results found (results/metrics.json is missing).\n"
    S, V = m["segment_model"], m["versions"]
    T, C, I, CAL = S["test"], S["conformal"], m["image_model"], m["calibrated_image"]["scale_error_factor"]
    pc = lambda v: f"{100 * v:.2f}%"
    ci = lambda k: f"{pc(T[k][0])} (95% CI {pc(T[k][1][0])} - {pc(T[k][1][1])})"
    cm, fs = T["confusion"], S["feature_selection"]
    out = [f"# Training report - {APP_NAME} v{__version__}", "",
           "This report is generated automatically from the output of the training scripts (`train_models.py`, `compare_models.py`, "
           "`train_multichannel.py`). Every number was measured on **held-out test data** that the models never saw during training. "
           "Nothing here needs to be re-run to use the program: the trained models are already in the `models/` folder.", "",
           "## 1. Data and set-up",
           "- **Data:** Bonn / UCI *Epileptic Seizure Recognition* - 11,500 one-second segments (178 samples at 173.61 Hz); seizure = class 1, all other classes = non-seizure.",
           "- **Split:** 80% training / 20% test (stratified, seed " + str(V["seed"]) + "). Hyper-parameters were chosen by cross-validation on the training part only.",
           f"- **Software:** Python {V['python']}, scikit-learn {V['sklearn']}, numpy {V['numpy']}, scipy {V['scipy']}.",
           f"- **Data file SHA-256:** `{V['data_sha256']}`", "",
           "## 2. Segment model (single-channel detector)",
           "| Metric | Held-out test set (n = " + str(T["n_test"]) + ") |", "|---|---|",
           f"| Accuracy | {ci('accuracy')} |", f"| Sensitivity (seizures found) | {ci('sensitivity')} |",
           f"| Specificity (non-seizure correct) | {ci('specificity')} |", f"| ROC-AUC | {T['roc_auc']} |", f"| Matthews correlation | {T['mcc']} |", "",
           f"Errors: {cm['FN']} of {cm['FN'] + cm['TP']} seizure segments missed; {cm['FP']} of {cm['FP'] + cm['TN']} non-seizure segments flagged.",
           f"10-fold cross-validation accuracy: {S['cv10']['accuracy'][0]:.4f} +/- {S['cv10']['accuracy'][1]:.4f}.", "",
           "**Sanity checks** (is the result real?)",
           f"- Majority-class baseline accuracy: {S['majority_baseline_acc']} (always answering 'non-seizure').",
           f"- With shuffled labels the same pipeline scores {S['shuffled_label_acc']} - no better than the baseline ({S['permutation_n']} permutations, p = {S['permutation_p']:.3f}), so the pipeline does not leak labels.",
           f"- Feature set: **{fs['chosen']}**, chosen by cross-validation on the training part (CV accuracy base {fs['base_cv']} vs extended {fs['extended_cv']}).", "",
           "![Performance overview](results/fig1_performance.png)", "",
           "The learning curve (panel F) shows the validation accuracy still rising slowly with more training data.", "",
           "![Explanation model coefficients](results/fig2_coefficients.png)", "",
           "## 3. The 'uncertain' zone (conformal abstention)",
           f"The model may answer 'uncertain - review' instead of guessing. With a target error of {C['alpha'] * 100:.0f}%: **{C['uncertain_pct']}% of test segments are sent to review**, "
           f"and on the rest accuracy is **{pc(C['accuracy_on_decided'])}** ({C['errors']} errors in {C['n_decided']}). "
           "The calibration uses out-of-fold scores of the training data (cross-conformal), so the error target is approximate and assumes new data resemble the training data.", "",
           "![Abstention trade-off](results/fig3_conformal_tradeoff.png)", ""]
    if cmp_:
        out += ["## 4. Why logistic regression? Headroom check", "Same split and test set; hyper-parameters chosen on the training part only.", "",
                "| Model | Test accuracy (95% CI) | Sensitivity | Specificity | 5-fold CV | McNemar p vs shipped |", "|---|---|---|---|---|---|"]
        out += [f"| {n} | {pc(v['accuracy'])} ({100 * v['accuracy_ci'][0]:.2f} - {100 * v['accuracy_ci'][1]:.2f}) | {pc(v['sensitivity'])} | {pc(v['specificity'])} | "
                f"{pc(v['cv5_accuracy'])} | {v['mcnemar_p_vs_shipped']} |" for n, v in cmp_["models"].items()]
        out += ["", f"**{cmp_['verdict']}** The ceiling of about 99% comes from the data (overlapping classes), not from the model type, so the simple, explainable linear model is used. "
                "The shipped model is the first row.", "", "![Model comparison](results/fig7_model_comparison.png)", ""]
    out += ["## 5. Image mode (tested on simulated images of held-out segments)",
            "| Mode | Accuracy | Sensitivity | Specificity |", "|---|---|---|---|",
            f"| **Calibrated** (correct uV per pixel) | **{pc(CAL['1.0']['accuracy'])}** | {pc(CAL['1.0']['sensitivity'])} | {pc(CAL['1.0']['specificity'])} |"]
    out += [f"| Calibrated, scale entered {lab} | {pc(CAL[k]['accuracy'])} | {pc(CAL[k]['sensitivity'])} | {pc(CAL[k]['specificity'])} |"
            for k, lab in (("0.7", "30% too small"), ("1.4", "40% too large"), ("2.0", "2x too large"))]
    out += [f"| Uncalibrated (scale unknown) | {pc(I['accuracy_digitised_trained_on_digitised'])} | {pc(I['sensitivity'])} | {pc(I['specificity'])} |", "",
            f"Calibrated mode abstains on {m['calibrated_image']['scale_error_factor']['conformal_exact_scale']['uncertain_pct']}% of windows. "
            f"Uncalibrated mode abstains on {I['uncertain_pct']}% and is a weak fallback: a picture without a scale hides how large the waves are. "
            "**These are simulated images, not real photographs.**", "", "![Image mode](results/fig4_image_mode.png)", ""]
    if chb:
        thr = str(chb["chosen_thr_det"])
        p = chb["pooled_lopo"][thr]
        out += ["## 6. Stage 2: multichannel detector and pre-seizure score (experimental)",
                f"CHB-MIT subset: {len(chb['patients'])} patients ({', '.join(chb['patients'])}), {chb['hours_total']:.1f} hours, evaluated **leave-one-patient-out** "
                "(the model never sees the test patient).", "",
                f"At threshold {thr}: **{100 * p['sens']:.1f}% of seizures detected** with **{p['fa_per_h']:.2f} false alarms per hour (about {24 * p['fa_per_h']:.0f} per day)** - far too many for practical use.", "",
                "| Patient | Hours | Seizures | Detected | False alarms | Pre-seizure AUC |", "|---|---|---|---|---|---|"]
        out += [f"| {k} | {v['hours']:.1f} | {v['seizures']} | {v['at_chosen_thr']['tp']} | {v['at_chosen_thr']['fp']} | "
                f"{'n/a' if v['preictal_auc'] is None else format(v['preictal_auc'], '.2f')} |" for k, v in chb["per_patient"].items()]
        out += ["", f"**Pre-seizure ('early warning') score:** mean held-out ROC-AUC {chb['preictal']['mean_auc']:.3f} (0.5 = chance) - not predictive, so it is **disabled in the program**.", "",
                "![Stage 2 sensitivity vs false alarms](results/fig5_chb_sensitivity_vs_fa.png)", ""]
        if os.path.isfile(os.path.join(base, "results/fig6_preictal_auc.png")):
            out += ["![Pre-seizure AUC per patient](results/fig6_preictal_auc.png)", ""]
    out += ["## 7. How to read these numbers",
            "- The Bonn data have no patient identifiers, so segments from one recording can sit in both training and test sets: accuracy is optimistic and does not show performance on new patients.",
            "- Continuous clinical EEG is much harder than these short, clean segments - see the false-alarm rate in section 6.",
            "- Image results come from simulated images; test real screenshots before relying on image mode.",
            "- This is a research prototype and not a medical device.", "",
            "## 8. Raw outputs", "- `results/metrics.json`, `results/model_comparison.json`, `results/chb_metrics.json` - all numbers in machine-readable form.",
            "- `results/logs/` - console output of the training run.", "- `results/SHA256SUMS.txt` - checksums of the models and code.", ""]
    return "\n".join(out)


def training_report_html(base="."):
    """The same report as a single self-contained HTML page (figures embedded) - opens in any browser."""
    return to_html(training_report_markdown(base), base, "Training report")
