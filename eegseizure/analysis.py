"""High-level analysis of a file or an image: returns one result dict that the GUI, the command line and the reports share.

Result dict keys (all analysers): kind ("rec" | "mc" | "img"), path, names, sigs, probs (per channel), p (combined probability per window),
best (index of the most seizure-like channel), n (windows), win / step (window length and hop in seconds), lab (0 non-seizure,
1 uncertain, 2 seizure; or None), feats, cp (model bundle used), warn (list of warnings), fs, plus image-only keys.
"""
import numpy as np

from .constants import FS, FS_MC, N, PAIRS, SEG_S
from .features import feats_batch, mc_features, preprocess, resample, seg_feats, zscore_rows
from .images import digitize
from .models import cp_label, cp_proba
from .readers import load_signals, read_edf, to_bipolar


def _finish(**kw):
    kw.setdefault("lab", None); kw.setdefault("pre", None); kw.setdefault("image", None)
    kw.setdefault("win", SEG_S); kw.setdefault("step", SEG_S); kw.setdefault("warn", [])
    return kw


def analyse_rec(path, fs, ch, cp, cb=lambda k, v: None, filt=True):
    """Analyse a recording file with the single-channel segment model `cp`: read, optional band-pass, 1.03 s windows, probability and conformal label per window. `ch` selects an EDF channel, `filt` toggles the 0.5-40 Hz band-pass, `cb(kind, value)` receives progress updates. Returns the result dict described at the top of this module."""
    cb("status", "Reading file...")
    sigs, rows = load_signals(path, fs, ch)
    sigs = [(n, s) for n, s in sigs if len(s) >= N]
    if not sigs:
        raise ValueError("The recording is shorter than one %.2f s segment." % SEG_S)
    if filt and not rows:
        sigs = [(n, preprocess(s)) for n, s in sigs]
    n = min(len(s) // N for _, s in sigs)
    probs, Fs, labs, warn = [], [], [], []
    for k, (name, s) in enumerate(sigs):
        cb("status", f"Extracting features ({name})...")
        cb("prog", k / len(sigs))
        F = seg_feats(cp, s[:n * N].reshape(n, N))
        Fs.append(F)
        p = cp_proba(cp, F)
        probs.append(p)
        labs.append(cp_label(cp, p))
        if np.median(s[:n * N].reshape(n, N).std(1)) > 5000 or np.median(s[:n * N].reshape(n, N).std(1)) < 1e-3:
            warn.append(f"{name}: amplitude looks unusual - are the units microvolts?")
    return _finish(kind="rec", path=path, names=[s[0] for s in sigs], sigs=[s[1][:n * N] for s in sigs],
                   probs=probs, p=np.max(probs, 0), best=int(np.argmax([q.max() for q in probs])), n=n,
                   rows=rows, feats=Fs, lab=np.max(labs, 0), cp=cp, warn=warn, fs=FS)


def analyse_mc(path, mc, cb=lambda k, v: None):
    """Analyse an EDF/BDF file with the experimental multichannel model `mc` (18 bipolar channels, 4 s windows, 2 s hop). The pre-seizure score is returned only when the model bundle marks it as useful."""
    cb("status", "Reading EDF...")
    if not path.lower().endswith((".edf", ".bdf")):
        raise ValueError("The multichannel model needs an EDF/BDF file.")
    data, sf, names = read_edf(path)
    B, missing = to_bipolar(data, names)
    if B is None:
        raise ValueError("Channels missing for the multichannel model: " + ", ".join(missing) +
                         "\nIt needs the 18 standard bipolar pairs (or the electrodes to derive them).")
    cb("status", "Filtering / resampling...")
    B = preprocess(resample(B, sf, FS_MC), FS_MC)
    cb("status", "Extracting features...")
    F = mc_features(B)
    if len(F) == 0:
        raise ValueError("The recording is shorter than 4 s.")
    p = mc["det"].predict_proba(F)[:, 1]
    useful = mc.get("pre") is not None and mc.get("pre_useful", False)
    pre = mc["pre"].predict_proba(F)[:, 1] if useful else None
    b = int(np.argmax(np.abs(B).mean(1)))
    return _finish(kind="mc", path=path, names=[PAIRS[b]], sigs=[B[b]], probs=[p], p=p, best=0, n=len(p),
                   rows=False, feats=[F], pre=pre, win=4.0, step=2.0, mc=mc, fs=FS_MC,
                   warn=([] if useful else ["Pre-seizure (early-warning) score disabled: it was not predictive in validation (AUC near 0.5)."]))


def analyse_image(img, cp, seconds=10.0, n_lanes=0, trim_l=0.0, trim_r=0.0, uv_per_px=None, cp_seg=None,
                  flat_px=1.5, cb=lambda k, v: None):
    """cp = amplitude-free image model.  If uv_per_px (microvolts per vertical pixel) and cp_seg are given,
    the accurate 'calibrated' mode is used: the traces are converted to microvolts and classified by the full segment model."""
    cb("status", "Digitising image...")
    lanes, info = digitize(img, n_lanes, (trim_l, trim_r))
    if not lanes:
        raise ValueError("No waveform trace found. Use a clear plot image (dark trace on light background).")
    calibrated = bool(uv_per_px) and cp_seg is not None
    use = cp_seg if calibrated else cp
    rate = info["W"] / float(seconds)
    probs, labs, Fs, sigs, warn = [], [], [], [], []
    for k, L in enumerate(lanes):
        sig = resample(L["y"], rate)
        n = len(sig) // N
        if n < 1:
            raise ValueError("Image covers less than 1 s - check 'seconds across image'.")
        raw = sig[:n * N].reshape(n, N)
        flat = raw.std(1) < flat_px
        if calibrated:
            raw = raw * float(uv_per_px)
            F = seg_feats(cp_seg, raw)
            med = float(np.median(raw.std(1)))
            if med > 3000 or med < 0.5:
                warn.append(f"lane {k + 1}: amplitude {med:.0f} uV looks implausible - check the uV-per-pixel value")
        else:
            F = feats_batch(zscore_rows(raw))
        p = cp_proba(use, F)
        lab = cp_label(use, p)
        p[flat], lab[flat] = 0.0, 0
        probs.append(p); labs.append(lab); Fs.append(F); sigs.append(sig[:n * N] * (float(uv_per_px) if calibrated else 1.0))
        if L["cov"] < 0.6:
            warn.append(f"lane {k + 1}: trace found in only {100 * L['cov']:.0f}% of columns - low quality")
    n = min(len(p) for p in probs)
    p = np.max([q[:n] for q in probs], 0)
    return _finish(kind="img", path="image", names=[f"lane {i + 1}" for i in range(len(lanes))], sigs=sigs,
                   probs=[q[:n] for q in probs], p=p, best=int(np.argmax([q.max() for q in probs])), n=n,
                   rows=False, feats=Fs, lab=np.max([q[:n] for q in labs], 0), cp=use, warn=warn, fs=FS,
                   image=np.asarray(img.convert("RGB")), lanes=lanes, rate=rate, x0=info["x0"], seconds=float(seconds),
                   calibrated=calibrated, kmin_min=1 if calibrated else 3)
