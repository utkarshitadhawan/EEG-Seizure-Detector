"""Stage 2 (experimental): multichannel seizure detector and pre-seizure score on CHB-MIT, evaluated leave-one-patient-out.

    python train_multichannel.py                        downloads a subset of CHB-MIT from PhysioNet (about 3 GB, 20-60 min)
    python train_multichannel.py --patients chb01 chb02 chb03 chb05 --ns-files 2 --max-sz-files 4      (smaller / faster)
    python train_multichannel.py --local /path/to/chbmit        use files that are already downloaded

Windows: 4 s, hop 2 s, 18 bipolar channels, per-channel features summarised by mean / max / std. Labels per window: ictal, pre-seizure
(30 min before onset, ending 5 min before), interictal. Ambiguous windows (just before onset, 30 min after a seizure) are excluded.
Events need 3 consecutive flagged windows; sensitivity and false alarms per hour are pooled over the held-out patients.
Writes models/chb_models.joblib, results/chb_metrics.json and results/fig5/fig6.
"""
import argparse
import json
import os
import re
import time
import urllib.request

import joblib
import matplotlib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                  # noqa: E402

import eegseizure as z                                                           # noqa: E402

URL = "https://physionet.org/files/chbmit/1.0.0/"
WIN, STEP = 4.0, 2.0
PRE_S, SPH_S, POST_S = 1800, 300, 1800        # pre-seizure period 30 min, intervention gap 5 min, post-ictal 30 min
K, GAP = 3, 15                                # >=3 consecutive windows; merge events closer than 15 windows (30 s)
THRS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
DEFAULT = ["chb01", "chb02", "chb03", "chb05", "chb07", "chb08", "chb09", "chb10"]


def fetch(url, dest):
    """Download `url` to `dest` (three tries); skip if the file already exists."""
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    err = None
    for _ in range(3):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:
            err = e
            time.sleep(3)
    raise RuntimeError(f"download failed: {url}: {err}")


def parse_summary(txt):
    """Seizure (start, end) seconds per file from a CHB-MIT *-summary.txt."""
    out = {}
    for blk in re.split(r"File Name:\s*", txt)[1:]:
        name = blk.split()[0]
        s = [int(v) for v in re.findall(r"Seizure\s*\d*\s*Start Time:\s*(\d+)", blk)]
        e = [int(v) for v in re.findall(r"Seizure\s*\d*\s*End Time:\s*(\d+)", blk)]
        out[name] = list(zip(s, e))
    return out


def label_windows(n, seizures):
    """Window labels: 1 ictal, 2 pre-seizure, 0 interictal, -1 excluded (ambiguous periods)."""
    tc = np.arange(n) * STEP + WIN / 2
    y = np.zeros(n, int)
    for s, e in seizures:
        y[(tc >= s) & (tc <= e)] = 1
    for s, e in seizures:
        y[(tc >= s - PRE_S) & (tc < s - SPH_S) & (y == 0)] = 2
        y[(((tc >= s - SPH_S) & (tc < s)) | ((tc > e) & (tc <= e + POST_S))) & (y == 0)] = -1
    return y


def choose_files(summary, n_ns, max_sz):
    """All seizure files (up to max_sz) plus evenly spaced seizure-free files."""
    sz = [f for f, v in summary.items() if v][:max_sz]
    ns = [f for f, v in summary.items() if not v]
    ns = [ns[i] for i in np.linspace(0, len(ns) - 1, min(n_ns, len(ns))).astype(int)] if ns else []
    return sz + ns


def load_patient(pt, a):
    """Download / read one patient's files; returns window features and labels per file."""
    d = os.path.join(a.local or a.dest, pt)
    sp = os.path.join(d, f"{pt}-summary.txt")
    if not a.local:
        fetch(f"{URL}{pt}/{pt}-summary.txt", sp)
    summ = parse_summary(open(sp).read())
    files = []
    for fn in choose_files(summ, a.ns_files, a.max_sz_files):
        p = os.path.join(d, fn)
        if not a.local:
            fetch(f"{URL}{pt}/{fn}", p)
        data, sf, names = z.read_edf(p)
        B, miss = z.to_bipolar(data, names)
        if B is None:
            print("  skip", fn, "missing", miss)
            continue
        F = z.mc_features(z.preprocess(z.resample(B, sf, z.FS_MC), z.FS_MC))
        files.append(dict(name=fn, F=F, y=label_windows(len(F), summ[fn]), sz=summ[fn], hours=len(F) * STEP / 3600))
        print(f"  {fn}: {len(F)} windows, {len(summ[fn])} seizure(s)")
    return files


def fit_lr(X, y):
    """Standardise + class-balanced logistic regression; None if y contains a single class."""
    if len(set(y)) < 2:
        return None
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)).fit(X, y)


def stack(files, task):
    """Stack the windows of several files for the 'det' (ictal vs rest) or 'pre' (pre-seizure vs interictal) task."""
    X = np.vstack([f["F"] for f in files]); y = np.concatenate([f["y"] for f in files])
    keep = (y >= 0) if task == "det" else np.isin(y, [0, 2])
    return X[keep], ((y[keep] == 1) if task == "det" else (y[keep] == 2)).astype(int)


def events_for(p, thr):
    """Events (in seconds) from window probabilities at a threshold."""
    ev = z.merge_events(z.find_events(p >= thr, K), GAP)
    return [(s * STEP, e * STEP + WIN) for s, e in ev]


def main():
    """Leave-one-patient-out evaluation, final model fit, metrics and figures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="*", default=DEFAULT)
    ap.add_argument("--ns-files", type=int, default=3)
    ap.add_argument("--max-sz-files", type=int, default=6)
    ap.add_argument("--local", default=None)
    ap.add_argument("--dest", default="chbmit_data")
    a = ap.parse_args()
    os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)
    P = {}
    for pt in a.patients:
        print(pt)
        P[pt] = load_patient(pt, a)
    P = {k: v for k, v in P.items() if sum(len(f["sz"]) for f in v) > 0}
    if len(P) < 3:
        raise SystemExit("Need at least 3 patients with seizures for leave-one-patient-out evaluation.")

    res = {pt: {t: dict(tp=0, nref=0, fp=0) for t in THRS} for pt in P}
    hours = {pt: sum(f["hours"] for f in P[pt]) for pt in P}
    auc, pre_all = {}, {"y0": [], "y2": []}
    for pt in P:
        others = [f for q in P if q != pt for f in P[q]]
        det = fit_lr(*stack(others, "det"))
        pre = fit_lr(*stack(others, "pre"))      # None if no pre-seizure windows are available
        for f in P[pt]:
            p = det.predict_proba(f["F"])[:, 1]
            for t in THRS:
                tp, nref, fp = z.score_events(f["sz"], events_for(p, t))
                for k, v in zip(("tp", "nref", "fp"), (tp, nref, fp)):
                    res[pt][t][k] += v
            if pre is not None:
                q = z.movavg(pre.predict_proba(f["F"])[:, 1], 30)
                pre_all["y0"] += list(q[f["y"] == 0]); pre_all["y2"] += list(q[f["y"] == 2])
        Xp, yp = stack(P[pt], "pre")
        if pre is not None and yp.min() != yp.max():
            auc[pt] = float(roc_auc_score(yp, pre.predict_proba(Xp)[:, 1]))
        print(pt, "held out ->", {t: res[pt][t] for t in (0.9, 0.99)}, "pre AUC", auc.get(pt))

    H = sum(hours.values())
    pooled = {t: dict(sens=sum(res[p][t]["tp"] for p in P) / max(1, sum(res[p][t]["nref"] for p in P)),
                      fa_per_h=sum(res[p][t]["fp"] for p in P) / H) for t in THRS}
    ok = [t for t in THRS if pooled[t]["fa_per_h"] <= 1.0]
    thr_det = max(ok, key=lambda t: (pooled[t]["sens"], t)) if ok else min(THRS, key=lambda t: pooled[t]["fa_per_h"])
    thr_pre = float(np.percentile(pre_all["y0"], 95)) if pre_all["y0"] else 0.8
    pre_sens = float(np.mean(np.array(pre_all["y2"]) >= thr_pre)) if pre_all["y2"] else float("nan")

    allf = [f for pt in P for f in P[pt]]
    pre_useful = bool(auc and np.mean(list(auc.values())) >= 0.60)      # early warning only enabled if it beats chance clearly
    models = dict(det=fit_lr(*stack(allf, "det")), pre=fit_lr(*stack(allf, "pre")), thr_det=float(thr_det),
                  thr_pre=thr_pre, k=K, names=z.MC_NAMES, patients=list(P), pre_useful=pre_useful)
    joblib.dump(models, "models/chb_models.joblib")
    out = dict(sklearn=sklearn.__version__, patients=list(P), hours_total=H, window_s=WIN, step_s=STEP, k=K,
               pooled_lopo={str(t): v for t, v in pooled.items()}, chosen_thr_det=thr_det,
               per_patient={pt: dict(hours=hours[pt], seizures=sum(len(f["sz"]) for f in P[pt]),
                                     at_chosen_thr=res[pt][thr_det], preictal_auc=auc.get(pt)) for pt in P},
               preictal=dict(enabled_in_app=pre_useful, mean_auc=float(np.mean(list(auc.values()))) if auc else None, thr=thr_pre,
                             frac_preictal_windows_above_thr=pre_sens,
                             note="30 min pre-seizure vs interictal; labels are file-local (see README limitations)"))
    json.dump(out, open("results/chb_metrics.json", "w"), indent=2)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot([pooled[t]["fa_per_h"] for t in THRS], [100 * pooled[t]["sens"] for t in THRS], "o-")
    for t in THRS:
        plt.annotate(str(t), (pooled[t]["fa_per_h"], 100 * pooled[t]["sens"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.xlabel("False alarms per hour"); plt.ylabel("Seizure sensitivity (%)"); plt.grid(alpha=.3)
    plt.title("Leave-one-patient-out detection (CHB-MIT subset)"); plt.tight_layout()
    plt.savefig("results/fig5_chb_sensitivity_vs_fa.png", dpi=170); plt.close()
    if auc:
        plt.figure(figsize=(7, 4)); plt.bar(list(auc), list(auc.values()), color="#9575cd")
        plt.axhline(0.5, color="k", ls="--"); plt.ylim(0, 1); plt.ylabel("ROC-AUC")
        plt.title("Pre-seizure vs interictal, held-out patients"); plt.tight_layout()
        plt.savefig("results/fig6_preictal_auc.png", dpi=170); plt.close()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
