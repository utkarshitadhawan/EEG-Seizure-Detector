"""Real-time style detection on a stream of samples (ring buffer + sliding window + alarm smoothing)."""
from collections import deque

import numpy as np

from .constants import FS, FS_MC, SEG_S
from .features import mc_features, preprocess, resample, seg_feats
from .models import cp_proba


class StreamDetector:
    """Ring-buffer detector for live data. Push chunks of samples; get one result per hop.
    mode 'single': 1 channel, Bonn segment model (bundle = cp).  mode 'mc': 18 bipolar channels
    in PAIRS order (bundle = chb_models)."""

    def __init__(self, bundle, mode="single", fs_in=FS, thr=0.5, k=3, n=5):
        self.b, self.mode, self.fs, self.thr, self.k = bundle, mode, float(fs_in), thr, k
        self.win, self.hop, self.ctx, self.target = ((SEG_S, SEG_S, 6.0, FS) if mode == "single"
                                                     else (4.0, 2.0, 12.0, FS_MC))
        self.buf, self.t, self.hist = None, 0.0, deque(maxlen=n)
        self.pend = 0

    def push(self, chunk):
        """Feed a chunk of samples (channels x samples, or 1-D); returns one result dict per completed hop."""
        chunk = np.atleast_2d(np.asarray(chunk, float))
        hop_raw = int(self.hop * self.fs)
        out = []
        for a in range(0, chunk.shape[1], hop_raw):
            piece = chunk[:, a:a + hop_raw]
            self.buf = piece if self.buf is None else np.hstack([self.buf, piece])[:, -int(self.ctx * self.fs):]
            self.pend += piece.shape[1]
            if self.pend >= hop_raw:
                self.pend -= hop_raw
                r = self._step()
                if r:
                    out.append(r)
        return out

    def _step(self):
        x = preprocess(resample(self.buf, self.fs, self.target), self.target)
        W = int(self.win * self.target)
        if x.shape[1] < W:
            return None
        seg = x[:, -W:]
        self.t += self.hop
        if self.mode == "single":
            p = float(cp_proba(self.b, seg_feats(self.b, seg[0:1]))[0]); pre = None
        else:
            F = mc_features(seg, self.target, self.win, self.hop)
            p = float(self.b["det"].predict_proba(F)[0, 1])
            pre = (float(self.b["pre"].predict_proba(F)[0, 1])
                   if self.b.get("pre") is not None and self.b.get("pre_useful", False) else None)
        self.hist.append(p >= self.thr)
        return dict(t=self.t, p=p, pre=pre, alarm=sum(self.hist) >= self.k)
