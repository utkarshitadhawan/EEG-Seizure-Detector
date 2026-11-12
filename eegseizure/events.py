"""Turning per-window probabilities into events, plus event-based scoring."""
import numpy as np


def movavg(x, k):
    """Moving average of length k with edge padding (the result has the same length as x)."""
    x = np.asarray(x, float)
    k = max(1, min(k, len(x)))
    c = np.cumsum(np.insert(x, 0, 0.0))
    return np.pad((c[k:] - c[:-k]) / k, (k // 2, k - 1 - k // 2), mode="edge")


def find_events(flag, kmin=1):
    """Runs of at least `kmin` consecutive True values in `flag`, as inclusive (start, end) window indices."""
    ev, i, n = [], 0, len(flag)
    while i < n:
        if flag[i]:
            j = i
            while j + 1 < n and flag[j + 1]:
                j += 1
            if j - i + 1 >= kmin:
                ev.append((i, j))
            i = j + 1
        else:
            i += 1
    return ev


def merge_events(ev, gap):
    """Merge events that are separated by at most `gap` windows."""
    out = []
    for s, e in sorted(ev):
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(e, out[-1][1]))
        else:
            out.append((s, e))
    return out


def score_events(ref, hyp, pre=30.0, post=60.0):
    """Event-based scoring (seconds). Returns (true detections, n reference, false alarms)."""
    hit = lambda a, b, r0, r1: a <= r1 + post and b >= r0 - pre
    tp = sum(any(hit(h0, h1, r0, r1) for h0, h1 in hyp) for r0, r1 in ref)
    fp = sum(not any(hit(h0, h1, r0, r1) for r0, r1 in ref) for h0, h1 in hyp)
    return tp, len(ref), fp
