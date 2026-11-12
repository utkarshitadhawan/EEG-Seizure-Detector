"""Create the reference sample files: example INPUT files in every supported format and the matching OUTPUT files.

    python make_samples.py          (run after train_models.py; uses the trained models and held-out test segments)

samples/inputs/   one example file per supported input format (see samples/README.md)
samples/outputs/  what the program produces for them: <name>_results.csv, <name>_report.html, <name>_figure.png

The demo signals are concatenations of held-out Bonn test segments (neighbouring segments are cross-faded). Random draws are searched
until the demos behave clearly (normal demos: nothing flagged, nothing uncertain; seizure demos: exactly one event); the outcome is
stored in results/demo_check.json. Real performance is reported in TRAINING_REPORT.md, not by these demos.
"""
import json
import os

import joblib
import numpy as np

import eegseizure as z
from eegseizure import report
from eegseizure.data import load_bonn, test_split

UV_PER_PX = 8.0                      # vertical scale of the demo images
IN_DIR, OUT_DIR = "samples/inputs", "samples/outputs"


def crossfade_join(segments, k=8):
    """Concatenate segments, blending k samples at every join so there are no step jumps."""
    out = np.asarray(segments[0], float).copy()
    ramp = np.linspace(0, 1, k)
    for seg in segments[1:]:
        out = np.concatenate([out[:-k], out[-k:] * (1 - ramp) + seg[:k] * ramp, seg[k:]])
    return out


def search(build, penalty, tries=400):
    """Try random draws until penalty(result) == 0; otherwise return the best draw. Returns (result, found_perfect)."""
    best = None
    for attempt in range(tries):
        out = build(np.random.default_rng(100 + attempt))
        score = penalty(out)
        if best is None or score < best[0]:
            best = (score, out)
        if score == 0:
            return out, True
    return best[1], False


def main():
    """Create sample inputs and reference outputs, and record the demo self-check."""
    X, y, _ = load_bonn()
    _, test_idx = test_split(y)
    cp, cp_img = joblib.load("models/seizure_model_cp.joblib"), joblib.load("models/image_model_cp.joblib")
    for d in (IN_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)
    events = lambda r: z.find_events(r["p"] >= 0.5, 1)
    uncertain = lambda r: int((r["lab"] == 1).sum())

    def draw(rs):
        return rs.permutation(test_idx[y[test_idx] == 0]), rs.permutation(test_idx[y[test_idx] == 1])

    # penalty: 0 means the demo behaves clearly
    pen_normal = lambda r: len(events(r)) + uncertain(r)
    pen_seizure = lambda r: abs(len(events(r)) - 1) + max(0, uncertain(r) - 1)

    def recording(kind):
        def build(rs):
            normal, seizure = draw(rs)
            idx = list(normal[40:85]) if kind == "normal" else list(normal[:20]) + list(seizure[:5]) + list(normal[20:40])
            signal = np.concatenate([X[i] for i in idx])
            np.savetxt(f"{IN_DIR}/_tmp.csv", signal, fmt="%.3f")
            return signal, z.analyse_rec(f"{IN_DIR}/_tmp.csv", z.FS, "Auto", cp, filt=True)
        (signal, _), ok = search(build, lambda o: (pen_normal if kind == "normal" else pen_seizure)(o[1]))
        np.savetxt(f"{IN_DIR}/recording_{kind}.csv", signal, fmt="%.3f")
        os.remove(f"{IN_DIR}/_tmp.csv")
        return ok

    def image(kind):
        def build(rs):
            normal, seizure = draw(rs)
            lanes = [crossfade_join([X[i] for i in normal[85 + 10 * k: 95 + 10 * k]]) for k in range(4)]
            if kind == "seizure":
                lanes[3] = crossfade_join([X[i] for i in list(normal[125:130]) + list(seizure[5:10])])
            img = z.render_multilane(lanes, seconds=10, pps=150, lane_h=200, seed=1, uv_per_px=UV_PER_PX, degrade=False)
            return img, lanes, z.analyse_image(img, cp_img, 10, uv_per_px=UV_PER_PX, cp_seg=cp)
        (img, lanes, _), ok = search(build, lambda o: (pen_normal if kind == "normal" else pen_seizure)(o[2]))
        img.save(f"{IN_DIR}/image_4channels_{kind}.png")
        return lanes, ok

    ok_all = True
    for kind in ("normal", "seizure"):
        ok_all &= recording(kind)
        lanes, ok = image(kind)
        ok_all &= ok
        if kind == "seizure":
            seizure_lanes = lanes
    print("demo files verified" if ok_all else "WARNING: demo files only partly clean (shipped anyway; see samples/README.md)")
    json.dump(dict(ok=bool(ok_all)), open("results/demo_check.json", "w"))

    # ---- inputs in the other formats (built from the 4-channel seizure demo, 10 s) --------------------------------
    seconds = int(10 * z.FS)
    block = np.array([lane[:seconds] for lane in seizure_lanes])                       # (4 channels, 1736 samples)
    np.savetxt(f"{IN_DIR}/recording_4channels.csv", block.T, fmt="%.3f", delimiter=",", header="ch1,ch2,ch3,ch4", comments="")
    np.save(f"{IN_DIR}/recording_channels_by_samples.npy", block)
    rows_idx = list(test_idx[y[test_idx] == 0][:8]) + list(test_idx[y[test_idx] == 1][:4])
    with open(f"{IN_DIR}/recording_rows_of_178_with_labels.csv", "w") as f:
        f.write(",".join(f"X{i}" for i in range(1, 179)) + ",y\n")
        for i in rows_idx:
            f.write(",".join(f"{v:.0f}" for v in X[i]) + f",{1 if y[i] else 2}\n")        # label 1 = seizure, 2 = other (Bonn convention)
    edf_ok = True
    try:
        import mne
        data = z.resample(block, z.FS, target=256.0) * 1e-6                               # EDF stores volts
        raw = mne.io.RawArray(data, mne.create_info(["Fz-Cz", "Cz-Pz", "C3-C4", "P3-O1"], 256.0, "eeg"), verbose=False)
        mne.export.export_raw(f"{IN_DIR}/recording_4channels.edf", raw, fmt="edf", overwrite=True, verbose=False)
    except Exception as e:                                                                # edfio missing -> no EDF sample
        edf_ok = False
        print("EDF sample not created (pip install edfio to enable):", e)

    # ---- outputs: exactly what the GUI exports -------------------------------------------------------------------
    jobs = [("recording_seizure", lambda p: z.analyse_rec(p, z.FS, "Auto", cp), f"{IN_DIR}/recording_seizure.csv"),
            ("recording_normal", lambda p: z.analyse_rec(p, z.FS, "Auto", cp), f"{IN_DIR}/recording_normal.csv")]
    if edf_ok:
        jobs.append(("recording_4channels_edf", lambda p: z.analyse_rec(p, z.FS, "All EEG channels", cp), f"{IN_DIR}/recording_4channels.edf"))
    from PIL import Image
    for kind in ("seizure", "normal"):
        path = f"{IN_DIR}/image_4channels_{kind}.png"
        jobs.append((f"image_4channels_{kind}_calibrated",
                     lambda p: z.analyse_image(Image.open(p).convert("RGB"), cp_img, 10, uv_per_px=UV_PER_PX, cp_seg=cp), path))
    jobs.append(("image_4channels_seizure_UNCALIBRATED",
                 lambda p: z.analyse_image(Image.open(p).convert("RGB"), cp_img, 10), f"{IN_DIR}/image_4channels_seizure.png"))
    for name, analyse, path in jobs:
        r = analyse(path)
        r["path"] = path
        s = report.write_outputs(r, f"{OUT_DIR}/{name}")
        print(f"{name:42s} -> {s['verdict']}")

    open("samples/README.md", "w").write(f"""# Sample files

Use these to see the accepted input formats and what the program produces. All signals are excerpts of held-out Bonn test segments.
The demos were picked so they behave clearly; real accuracy is reported in `TRAINING_REPORT.md`.

## inputs/  (open them with the program)
| File | Format | How to use |
|---|---|---|
| `recording_seizure.csv` | one column of numbers (microvolts), 173.61 Hz, 45 s | Recording mode. Contains one seizure at about 20-25 s. |
| `recording_normal.csv` | same format | Recording mode. No seizure. |
| `recording_rows_of_178_with_labels.csv` | header `X1...X178,y`, one 178-sample segment per row (UCI/Kaggle layout) | Recording mode. Each row is analysed as an independent 1.03 s segment; the label column `y` is ignored. |
| `recording_4channels.csv` | header `ch1..ch4`, one channel per column, 173.61 Hz, 10 s | Recording mode, sampling rate 173.61. Channel 4 has a seizure from 5 s. |
| `recording_channels_by_samples.npy` | NumPy array shaped (channels, samples) | Recording mode, sampling rate 173.61. |
| `recording_4channels.edf` | EDF, 4 channels, 256 Hz{'' if edf_ok else ' (not created in this build)'} | Recording mode; choose a channel or "All EEG channels". The sampling rate is read from the file. |
| `image_4channels_seizure.png` | image of 4 stacked channels, 10 s across the width | Image mode: seconds = 10, channels = 0 (auto), **uV per pixel = {UV_PER_PX}**. |
| `image_4channels_normal.png` | same | Image mode with the same settings. |

Leave *uV per pixel* empty to see the weak "uncalibrated" image mode: it may miss the seizure or only report uncertain windows, which is why the scale matters.

## outputs/  (what the program writes)
For every input above the program writes three files (the GUI's *Export CSV*, *Export report* and *Save figure* buttons):
- `<name>_results.csv` - one row per analysis window: `start_s`, `end_s`, `probability`, `flagged`, `in_event`, `status` (non-seizure / uncertain / seizure) and, for several channels, one probability column per channel.
- `<name>_report.html` - verdict, statistics, figure and event table; opens in any browser.
- `<name>_figure.png` - the plot shown in the program.
""")


if __name__ == "__main__":
    main()
