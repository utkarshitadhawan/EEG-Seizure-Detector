"""Live-detection demo with a ring buffer, sliding windows and k-of-n alarm smoothing.
  python live_demo.py samples/inputs/recording_seizure.csv     replay a file as if it were live (20x speed)
  python live_demo.py file.edf --mode mc --speed 1    multichannel model, real time
  python live_demo.py --brainflow                     BrainFlow synthetic board (needs: pip install brainflow)
Research demo only - not for real-time medical monitoring."""
import argparse
import os
import time

import eegseizure as z

BASE = os.path.dirname(os.path.abspath(__file__))


def show(r):
    """Print one live result line."""
    pre = f"  pre-score {r['pre']:.2f}" if r["pre"] is not None else ""
    print(f"t={r['t']:7.1f}s  P(seizure)={r['p']:.3f}{pre}  {'*** ALARM ***' if r['alarm'] else ''}", flush=True)


def main():
    """Command-line entry point: replay a file (or a BrainFlow board) through the stream detector."""
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--mode", choices=["single", "mc"], default="single")
    ap.add_argument("--speed", type=float, default=20.0)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--brainflow", action="store_true")
    a = ap.parse_args()
    M = z.load_models(BASE)
    bundle = M["mc"] if a.mode == "mc" else M["seg"]
    if bundle is None:
        raise SystemExit("Model file not found in models/.")
    thr = a.thr if a.thr is not None else (bundle.get("thr_det", 0.5) if a.mode == "mc" else 0.5)
    if a.brainflow:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
        bid = BoardIds.SYNTHETIC_BOARD.value
        board = BoardShim(bid, BrainFlowInputParams()); board.prepare_session(); board.start_stream()
        fs, ch = BoardShim.get_sampling_rate(bid), BoardShim.get_eeg_channels(bid)[0]
        det = z.StreamDetector(bundle, "single", fs, thr)
        print("Streaming from BrainFlow synthetic board (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(0.5)
                for r in det.push(board.get_board_data()[ch]):
                    show(r)
        finally:
            board.stop_stream(); board.release_session()
        return
    if not a.path:
        raise SystemExit("Give a file to replay, or use --brainflow.")
    if a.mode == "mc":
        data, fs, names = z.read_edf(a.path)
        X, miss = z.to_bipolar(data, names)
        if X is None:
            raise SystemExit("Missing channels: " + ", ".join(miss))
    else:
        sigs, rows = z.load_signals(a.path, z.FS)
        X, fs = sigs[0][1][None, :], z.FS
    det = z.StreamDetector(bundle, a.mode, fs, thr, k=bundle.get("k", 3) if a.mode == "mc" else 3)
    chunk = int(0.25 * fs)
    for i in range(0, X.shape[1], chunk):
        for r in det.push(X[:, i:i + chunk]):
            show(r)
        time.sleep(0.25 / a.speed)


if __name__ == "__main__":
    main()
