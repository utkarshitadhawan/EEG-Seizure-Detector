"""eegseizure - seizure screening for EEG recordings and EEG images (research prototype, not a medical device).

Module guide (read in this order):
    constants   sampling rates, window sizes, frequency bands, feature and channel names
    features    per-window features, band-pass filter, resampling
    readers     EDF/BDF/CSV/TXT/NPY reading, bipolar channel construction
    models      training (logistic regression + conformal abstention), explanations, loading the model files
    events      probabilities -> events, event-based scoring
    images      image digitisation and the simulated images used for training/testing
    analysis    analyse_rec / analyse_mc / analyse_image: one result dict per file or image
    stream      StreamDetector for live-style detection
    report      verdict text, tables, figures, CSV/HTML export, training report
    minimarkdown  tiny Markdown reader used by the GUI and the HTML export
    data        loading the Bonn training data
"""
__version__ = "1.0.0"

from .constants import (BANDS, E, FINE, FS, FS_MC, MC_NAMES, N, PAIRS, PLAIN, PLAIN_EXT, SEG_S, SF_IDX)      # noqa: F401
from .features import feats_batch, feats_ext, mc_features, preprocess, resample, seg_feats, zscore_rows       # noqa: F401
from .readers import edf_channels, load_signals, norm_name, read_edf, read_table, to_bipolar                  # noqa: F401
from .models import (conformal_q, conformal_tradeoff, cp_label, cp_predict, cp_proba, find_model, fit_cp,
                     load_models, metrics_basic, top_features)                                                # noqa: F401
from .events import find_events, merge_events, movavg, score_events                                           # noqa: F401
from .images import (digitize, digitize_segment, make_image_dataset, otsu, px_signal, render_fixed,
                     render_multilane, render_trace)                                                          # noqa: F401
from .analysis import analyse_image, analyse_mc, analyse_rec                                                  # noqa: F401
from .stream import StreamDetector                                                                            # noqa: F401
