"""Reading EEG files (EDF/BDF/CSV/TXT/NPY) and building the standard bipolar channels."""
import os
import re

import numpy as np
import pandas as pd

from .constants import N, PAIRS
from .features import resample


SKIP = ("ecg", "ekg", "eog", "emg", "resp", "stim", "trigger", "status", "photic")   # non-EEG channel names


def read_edf(path):
    """Returns (data in microvolts [channels x samples], sampling rate, channel names)."""
    import mne
    rd = mne.io.read_raw_bdf if path.lower().endswith(".bdf") else mne.io.read_raw_edf
    raw = rd(path, preload=False, verbose=False)
    return raw.get_data() * 1e6, float(raw.info["sfreq"]), list(raw.ch_names)


def edf_channels(path):
    """Channel names of an EDF/BDF file (reads the header only)."""
    import mne
    rd = mne.io.read_raw_bdf if path.lower().endswith(".bdf") else mne.io.read_raw_edf
    return list(rd(path, preload=False, verbose=False).ch_names)


def read_table(path):
    """Read a numeric text/CSV file (comma, semicolon or whitespace separated) into a 2-D float array; header lines are dropped."""
    for kw in ({}, {"sep": r"[,;\s]+", "engine": "python"}):
        try:
            a = pd.read_csv(path, header=None, **kw).apply(pd.to_numeric, errors="coerce")
            a = a.dropna(axis=0, how="all").dropna(axis=1, how="all")
            if a.size:
                return a.values.astype(float)
        except Exception:
            pass
    raise ValueError("Could not read numeric data from this file.")


def load_signals(path, fs, ch="Auto"):
    """-> ([(name, 1-D signal at FS Hz)], rows_format_flag)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".edf", ".bdf"):
        data, sf, names = read_edf(path)
        if ch in names:
            idx = [names.index(ch)]
        else:
            idx = [i for i, n in enumerate(names) if not n.lower().startswith(SKIP)] or list(range(len(names)))
        return [(names[i], resample(data[i], sf)) for i in idx], False
    if ext == ".npy":
        a = np.asarray(np.load(path), float)
        a = a.reshape(-1, 1) if a.ndim == 1 else a
        if a.shape[1] not in (N, N + 1, N + 2) and a.shape[0] < a.shape[1]:
            a = a.T
    else:
        a = read_table(path)
    if a.shape[1] == N + 2:
        a = a[:, 1:N + 1]
    elif a.shape[1] == N + 1:
        a = a[:, :N]
    if a.shape[1] == N:
        a = a[~np.isnan(a).any(axis=1)]
        return [("segments", a.ravel())], True
    return [(f"col{j + 1}", resample(a[:, j], fs)) for j in range(a.shape[1])], False


def norm_name(n):
    """Normalise an EDF channel label (case, 'EEG', '-REF', old T3/T4/T5/T6 names) so it can be matched against constants.PAIRS."""
    n = n.upper().replace("EEG", "").replace(" ", "")
    n = re.sub(r"-(REF|LE|AR|AVG)$", "", n)
    n = re.sub(r"(?<=[A-Z0-9])-[01]$", "", n)
    for a, b in (("T3", "T7"), ("T4", "T8"), ("T5", "P7"), ("T6", "P8")):
        n = re.sub(rf"\b{a}\b", b, n)
    return n


def to_bipolar(data, names):
    """Build the 18 standard bipolar channels from bipolar or referential EDF channels."""
    d = {norm_name(n): i for i, n in enumerate(names)}
    rows, missing = [], []
    for pr in PAIRS:
        a, b = pr.split("-")
        if pr in d:
            rows.append(data[d[pr]])
        elif a in d and b in d:
            rows.append(data[d[a]] - data[d[b]])
        else:
            missing.append(pr)
    return (None if missing else np.array(rows)), missing
