"""Shared constants: sampling rates, window sizes, frequency bands, feature names and channel lists."""

# --- single-channel segment model (trained on the Bonn / UCI data) -------------------------------------------
FS = 173.61                    # sampling rate of the Bonn recordings (Hz); all single-channel input is resampled to it
N = 178                        # samples per analysis segment
SEG_S = N / FS                 # segment length in seconds (about 1.025 s)

# --- multichannel model (trained on CHB-MIT) -----------------------------------------------------------------
FS_MC = 256.0                  # CHB-MIT sampling rate (Hz)

E = 1e-12                      # tiny number that keeps logarithms and divisions finite

# --- classic EEG bands used for the band-power features (Hz) -------------------------------------------------
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 60)]            # delta, theta, alpha, beta, gamma
# finer bands used only by the extended feature set (the Bonn data are low-passed at ~40 Hz)
FINE = [(0.5, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 40)]

# --- feature names (same order as the columns produced by features.feats_batch / feats_ext) -----------------
PLAIN = ["amplitude (std)", "skewness", "kurtosis", "line length", "peak-to-peak",
         "delta power", "theta power", "alpha power", "beta power", "gamma power",
         "relative delta", "relative theta", "relative alpha", "relative beta", "relative gamma",
         "Hjorth mobility", "Hjorth complexity", "spectral entropy", "zero crossings"]
PLAIN_EXT = PLAIN + ["spectral edge 95%", "median frequency", "peak frequency", "peak power ratio", "spectral flatness",
                     "Teager energy", "Katz fractal dimension", "local peaks"] + [f"power {a}-{b} Hz" for a, b in FINE]
# columns of PLAIN that do not depend on the signal amplitude (used by the uncalibrated image model)
SF_IDX = [1, 2, 10, 11, 12, 13, 14, 15, 16, 17, 18]

# --- multichannel model: the 18 standard bipolar channels (double banana montage) ----------------------------
PAIRS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4",
         "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ"]
# per-window summary of the per-channel features: mean, max and standard deviation across channels
MC_NAMES = [f"{stat} {name}" for stat in ("mean", "max", "std") for name in PLAIN]
