"""EEG images: trace digitisation (image -> signal) and the simulated images used for training and testing."""
import io

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d

from .constants import FS, N, SEG_S
from .features import feats_batch, resample, zscore_rows


def otsu(g):
    """Otsu threshold of grey values (0-255): the split that best separates dark trace pixels from the background."""
    h, e = np.histogram(g, bins=64, range=(0, 255))
    h = h.astype(float)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = ((h * c).sum() - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(e[np.argmax(w0 * w1 * (m0 - m1) ** 2) + 1])


def _lane_bounds(prof, H, k):
    """k equal bands; shift them so the boundaries sit in low-density rows (between traces)."""
    L, best = H / k, None
    for o in np.linspace(-L / 2, L / 2, 25):
        b = [int(np.clip(round(o + j * L), 0, H)) for j in range(k + 1)]
        cost = sum(prof[min(v, H - 1)] for v in b[1:-1])
        if best is None or cost < best[0]:
            best = (cost, b)
    b = best[1]
    b[0], b[-1] = 0, H
    return b


def digitize(img, n_lanes=0, trim=(0.0, 0.0)):
    """Image -> list of traces (one per lane). n_lanes: 0 = auto, 1 = single trace, k = k stacked channels."""
    g = np.asarray(img.convert("RGB"), float) @ np.array([0.299, 0.587, 0.114])
    H, W0 = g.shape
    a, b = int(W0 * trim[0]), W0 - int(W0 * trim[1])
    g = g[:, a:b]
    W = g.shape[1]
    if W < 20:
        raise ValueError("Image too narrow after trimming.")
    if np.median(g) < 110:                      # dark theme -> invert
        g = 255 - g
    med = np.median(g)
    mask = g < min(otsu(g), med - 0.5 * (med - g.min()))   # trace pixels (light grid lines excluded)
    prof = gaussian_filter1d(mask.sum(1).astype(float), max(2.0, H * 0.01))
    bounds = [0, H]
    if n_lanes > 1:
        bounds = _lane_bounds(prof, H, n_lanes)
    elif n_lanes == 0 and prof.max() > 0:
        for k in range(min(40, H // 12), 1, -1):          # largest lane count whose lanes all have a trace everywhere
            bb = _lane_bounds(prof, H, k)
            if min(mask[t:u].any(0).mean() for t, u in zip(bb[:-1], bb[1:])) >= 0.85:
                bounds = bb
                break
    lanes = []
    for top, bot in zip(bounds[:-1], bounds[1:]):
        m = mask[top:bot]
        cnt = m.sum(0)
        rows = np.arange(bot - top)[:, None]
        y = np.where(cnt > 0, (m * rows).sum(0) / np.maximum(cnt, 1), np.nan)
        cov = float(np.mean(cnt > 0))
        if cov < 0.3:
            continue
        ix = np.arange(W)
        ok = ~np.isnan(y)
        y = np.interp(ix, ix[ok], y[ok])
        lanes.append(dict(y=-(y - np.median(y)), ypix=y + top, cov=cov, top=top, bot=bot))
    return lanes, dict(W=W, x0=a, H=H, bounds=bounds)


def _theme(rng):
    if rng.random() < 0.15:
        return dict(bg=(20, 20, 24), fg=tuple(int(v) for v in rng.integers(180, 256, 3)), grid=(45, 45, 52))
    v = int(rng.integers(235, 256))
    return dict(bg=(v,) * 3, fg=[(0, 0, 0), (10, 10, 120), (120, 10, 10), (20, 20, 20)][int(rng.integers(4))],
                grid=(int(rng.integers(205, 235)),) * 3)


def _degrade(img, rng):
    if rng.random() < 0.5:
        a = np.asarray(img, float) + rng.normal(0, rng.uniform(0, 6), (img.height, img.width, 1))
        img = Image.fromarray(np.clip(a, 0, 255).astype("uint8"))
    if rng.random() < 0.5:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(rng.integers(30, 96)))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def _grid(d, W, H, th, rng):
    g = int(rng.integers(10, 40))
    for k in range(0, W, g):
        d.line([(k, 0), (k, H)], fill=th["grid"])
    for k in range(0, H, g):
        d.line([(0, k), (W, k)], fill=th["grid"])


def render_trace(x, rng=None):
    """Render one 1-s segment as a randomly styled plot image (training simulation)."""
    rng = rng or np.random.default_rng()
    xs = np.asarray(x, float)
    xs = xs - np.median(xs)
    W, H, th = max(40, int(len(xs) / FS * rng.uniform(60, 400))), int(rng.integers(100, 301)), None
    th = _theme(rng)
    img = Image.new("RGB", (W, H), th["bg"])
    d = ImageDraw.Draw(img)
    if rng.random() < 0.4:
        _grid(d, W, H, th, rng)
    sc = rng.uniform(0.25, 0.9) * H / (xs.max() - xs.min() + 1e-9)
    d.line(list(zip(np.linspace(0, W - 1, len(xs)), H / 2 - xs * sc)), fill=th["fg"],
           width=int(rng.integers(1, 4)), joint="curve")
    return _degrade(img, rng)


def render_fixed(x, uv_per_px, rng=None):
    """Render a segment with a FIXED gain (like a real EEG viewer): amplitude = pixels * uv_per_px."""
    rng = rng or np.random.default_rng()
    xs = np.asarray(x, float)
    xs = xs - np.median(xs)
    W = max(40, int(len(xs) / FS * rng.uniform(60, 400)))
    H = int(np.clip(2.2 * np.percentile(np.abs(xs), 99) / uv_per_px + 20, 100, 600))
    th = _theme(rng)
    img = Image.new("RGB", (W, H), th["bg"])
    d = ImageDraw.Draw(img)
    if rng.random() < 0.4:
        _grid(d, W, H, th, rng)
    ys = np.clip(H / 2 - xs / uv_per_px, 0, H - 1)
    d.line(list(zip(np.linspace(0, W - 1, len(xs)), ys)), fill=th["fg"], width=int(rng.integers(1, 4)), joint="curve")
    return _degrade(img, rng)


def px_signal(img, n=N):
    """Digitise a single-trace image -> n samples (in pixels) at FS Hz, or None."""
    lanes, info = digitize(img, n_lanes=1)
    if not lanes:
        return None
    sig = resample(lanes[0]["y"], info["W"] / SEG_S)
    return np.pad(sig, (0, max(0, n - len(sig))), mode="edge")[:n]


def render_multilane(sigs, seconds=10, pps=120, lane_h=90, seed=0, grid=True, uv_per_px=None, degrade=True):
    """Render stacked channels (like a screenshot of an EEG viewer). sigs: arrays at FS Hz."""
    rng = np.random.default_rng(seed)
    W, H = int(seconds * pps), lane_h * len(sigs)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    if grid:
        for k in range(0, W, pps // 2):
            d.line([(k, 0), (k, H)], fill=(225, 225, 225))
    nS = int(seconds * FS)
    segs = [np.asarray(s[:nS], float) - np.median(s[:nS]) for s in sigs]
    sc = (1.0 / uv_per_px) if uv_per_px else 0.45 * lane_h / (np.percentile(np.abs(np.concatenate(segs)), 99) + 1e-9)
    for i, s in enumerate(segs):
        ys = np.clip(lane_h * (i + 0.5) - s * sc, i * lane_h + 1, (i + 1) * lane_h - 2)
        d.line(list(zip(np.linspace(0, W - 1, len(s)), ys)), fill=(0, 0, 0), width=1)
    return _degrade(img, rng) if degrade else img


def digitize_segment(x, rng):
    """render -> digitise -> resample one segment (digitisation-in-the-loop training)."""
    lanes, info = digitize(render_trace(x, rng), n_lanes=1)
    if not lanes:
        return None
    sig = resample(lanes[0]["y"], info["W"] / SEG_S)
    sig = np.pad(sig, (0, max(0, N - len(sig))), mode="edge")
    return sig[:N]


def make_image_dataset(Xraw, seed=0):
    """z-scored digitised segments -> (features, number of fallbacks)."""
    rng = np.random.default_rng(seed)
    out, fail = [], 0
    for x in Xraw:
        s = digitize_segment(x, rng)
        if s is None:
            s, fail = np.asarray(x, float), fail + 1
        out.append(s)
    return feats_batch(zscore_rows(np.array(out))), fail
