"""Signal processing: per-window features, band-pass filtering and resampling.

All feature functions are vectorised: they take a 2-D array (one window per row) and return one feature row per window.
"""
from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt, welch
from scipy.stats import kurtosis, skew

from .constants import BANDS, E, FINE, FS, FS_MC, PLAIN


def feats_batch(X, fs=FS):
    """19 features for every row of X (vectorised)."""
    X = np.atleast_2d(np.asarray(X, float))
    f, p = welch(X, fs, nperseg=X.shape[1], axis=-1)
    tot = p.sum(1) + E
    bp = np.stack([p[:, (f >= a) & (f < b)].sum(1) for a, b in BANDS], 1)
    d1 = np.diff(X, axis=1)
    d2 = np.diff(d1, axis=1)
    v0, v1, v2 = X.var(1), d1.var(1), d2.var(1)
    mob = np.sqrt(v1 / (v0 + E)) + E
    comp = np.sqrt(v2 / (v1 + E)) / mob
    pn = p / tot[:, None]
    sent = -(pn * np.log(pn + E)).sum(1)
    zc = (np.diff(np.sign(X - X.mean(1, keepdims=True)), axis=1) != 0).sum(1)
    out = np.column_stack([np.log(X.std(1) + E), skew(X, axis=1), kurtosis(X, axis=1),
                           np.log(np.abs(d1).sum(1) + E), np.log(X.max(1) - X.min(1) + E),
                           np.log(bp + E), bp / tot[:, None], mob, comp, sent, zc])
    return np.nan_to_num(out)


def feats_ext(X, fs=FS):
    """Extended feature set (36): the 19 base features + spectral shape, rhythmicity, Teager energy, fractal dimension, fine bands."""
    X = np.atleast_2d(np.asarray(X, float))
    base = feats_batch(X, fs)
    f, p = welch(X, fs, nperseg=X.shape[1], axis=-1)
    tot = p.sum(1) + E
    cum = np.cumsum(p, 1) / tot[:, None]
    last = len(f) - 1
    sef95, med = f[np.minimum((cum < 0.95).sum(1), last)], f[np.minimum((cum < 0.5).sum(1), last)]
    pk, ppr = f[p.argmax(1)], p.max(1) / tot
    flat = np.exp(np.log(p + E).mean(1)) / (p.mean(1) + E)
    tk = np.log(np.abs((X[:, 1:-1] ** 2 - X[:, :-2] * X[:, 2:]).mean(1)) + E)
    d1 = np.diff(X, axis=1)
    n = X.shape[1] - 1
    katz = np.log10(n) / (np.log10(n) + np.log10(np.abs(X - X[:, :1]).max(1) / (np.abs(d1).sum(1) + E) + E))
    peaks = ((d1[:, :-1] * d1[:, 1:]) < 0).sum(1) / n
    fine = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(1) for a, b in FINE], 1) + E)
    return np.nan_to_num(np.column_stack([base, sef95, med, pk, ppr, flat, tk, katz, peaks, fine]))


def seg_feats(cp, X, fs=FS):
    """Features for a segment-model bundle (base or extended, as the bundle was trained)."""
    return feats_ext(X, fs) if cp.get("ext") else feats_batch(X, fs)


def zscore_rows(S):
    """Standardise every row to zero mean and unit variance."""
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def mc_features(B, fs=FS_MC, win_s=4.0, step_s=2.0, chunk=300):
    """Window features for a multichannel recording B (channels x samples) -> (windows x 57)."""
    W, S = int(win_s * fs), int(step_s * fs)
    n = (B.shape[1] - W) // S + 1
    if n < 1:
        return np.zeros((0, 3 * len(PLAIN)))
    out = []
    for a in range(0, n, chunk):
        ix = np.arange(a, min(n, a + chunk))[:, None] * S + np.arange(W)
        seg = B[:, ix]
        C, m, _ = seg.shape
        F = feats_batch(seg.reshape(C * m, W), fs).reshape(C, m, -1)
        out.append(np.concatenate([F.mean(0), F.max(0), F.std(0)], 1))
    return np.vstack(out)


def preprocess(x, fs=FS, lo=0.5, hi=40.0):
    """Band-pass 0.5-40 Hz (the Bonn data were reported to be filtered similarly)."""
    x = np.nan_to_num(np.asarray(x, float))
    if x.shape[-1] < 300:
        return x
    return sosfiltfilt(butter(4, [lo, hi], btype="band", fs=fs, output="sos"), x, axis=-1)


def resample(d, fs, target=FS):
    """Resample the last axis of `d` from `fs` Hz to `target` Hz (rational polyphase filter)."""
    d = np.nan_to_num(np.asarray(d, float))
    if abs(fs - target) < 0.05:
        return d
    fr = Fraction(target / fs).limit_denominator(1000)
    return resample_poly(d, fr.numerator, fr.denominator, axis=-1)
