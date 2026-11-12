"""Command-line analysis of recordings and EEG images (single files or whole folders).

    python predict.py samples/inputs                                  every recording / image in a folder
    python predict.py recording.edf --channel Fz-Cz                   one EDF channel
    python predict.py image.png --seconds 10 --uv-per-px 2            image, accurate (calibrated) mode
    python predict.py image.png --seconds 10                          image, weak uncalibrated mode
    python predict.py recording.edf --model mc                        experimental multichannel model
    python predict.py samples/inputs --out summary.csv --outputs results_folder     also write CSV / HTML / figure for every file

The verdicts are produced by eegseizure.report.summarise, the same code the GUI uses.
Research prototype - not a medical device.
"""
import argparse
import glob
import os
import sys

import pandas as pd
from PIL import Image

import eegseizure as z
from eegseizure import report

BASE = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
RECORDING_EXT = (".edf", ".bdf", ".csv", ".txt", ".npy")


def analyse_one(path, models, args):
    """Run the right analysis for one file and return (result dict, summary dict)."""
    is_image = path.lower().endswith(IMAGE_EXT)
    multi = args.model == "mc" and not is_image
    if is_image:
        r = z.analyse_image(Image.open(path).convert("RGB"), models["img"], args.seconds, args.lanes,
                            uv_per_px=args.uv_per_px, cp_seg=models["seg"])
    elif multi:
        r = z.analyse_mc(path, models["mc"])
    else:
        r = z.analyse_rec(path, args.fs, args.channel, models["seg"])
    r["path"] = path
    thr = args.thr if args.thr is not None else (models["mc"].get("thr_det", 0.5) if multi else 0.5)
    kmin = args.min_windows or (models["mc"].get("k", 3) if multi else 1)
    return r, thr, kmin


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a file or a folder")
    ap.add_argument("--model", choices=["seg", "mc"], default="seg", help="seg = single-channel (default), mc = experimental multichannel")
    ap.add_argument("--seconds", type=float, default=10.0, help="image: seconds across the image width")
    ap.add_argument("--lanes", type=int, default=0, help="image: number of channels in the image (0 = automatic)")
    ap.add_argument("--uv-per-px", type=float, default=None, help="image: microvolts per vertical pixel (enables accurate mode)")
    ap.add_argument("--channel", default="Auto", help="EDF channel name (default: all EEG channels)")
    ap.add_argument("--fs", type=float, default=z.FS, help="sampling rate of text/NPY files (Hz)")
    ap.add_argument("--thr", type=float, default=None, help="decision threshold (default 0.5)")
    ap.add_argument("--min-windows", type=int, default=None, help="minimum consecutive flagged windows")
    ap.add_argument("--out", default=None, help="write the summary table to this CSV file")
    ap.add_argument("--outputs", default=None, help="folder for per-file results CSV, HTML report and figure")
    args = ap.parse_args()

    models = z.load_models(BASE)
    if os.path.isdir(args.path):
        files = sorted(f for f in glob.glob(os.path.join(args.path, "*"))
                       if f.lower().endswith(IMAGE_EXT + RECORDING_EXT) and not os.path.basename(f).lower().startswith("readme"))
    else:
        files = [args.path]
    if args.outputs:
        os.makedirs(args.outputs, exist_ok=True)
    rows = []
    for f in files:
        try:
            r, thr, kmin = analyse_one(f, models, args)
            s = report.summarise(r, thr, kmin)
            mode = ("image-calibrated" if r.get("calibrated") else "image-UNCALIBRATED") if r["kind"] == "img" else r["kind"]
            rows.append(dict(file=os.path.basename(f), mode=mode, verdict=s["level"].upper(), events=len(s["events"]),
                             event_times_s=str([(float(row[1]), float(row[2])) for row in s["rows"]]),
                             max_prob=round(float(r["p"].max()), 3), uncertain_windows=s["n_uncertain"], windows=r["n"],
                             warnings="; ".join(r["warn"])))
            if args.outputs:
                report.write_outputs(r, os.path.join(args.outputs, os.path.splitext(os.path.basename(f))[0]), thr, kmin)
        except Exception as e:
            rows.append(dict(file=os.path.basename(f), verdict="ERROR", warnings=f"{type(e).__name__}: {e}"))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print("saved", args.out)
    print("\nResearch prototype - not a medical device.")
    return 0 if len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
