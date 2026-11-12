# Code guide

For readers who open this repository for the first time: this guide describes the code as it is.

## 1. Big picture
~~~
 EEG recording (EDF/CSV/...)  --readers-->  signal --features--> window features --models--> probabilities --events/report--> verdict
 EEG image (PNG/JPG)          --images (digitise)-->  signal  ------------------^
~~~
1. **Windows.** A signal is cut into windows (1.03 s for the single-channel model, 4 s for the multichannel model).
2. **Features.** Each window becomes 19 numbers (amplitude, band powers, Hjorth parameters, entropy, ...); an extended set has 36.
3. **Model.** Logistic regression gives P(seizure). Two conformal thresholds turn it into *non-seizure / uncertain / seizure*.
4. **Events.** Runs of consecutive windows above the threshold become events; the report module formats verdict, tables, figures, CSV and HTML.
5. **Images.** The image is turned into a signal (digitisation); with a known uV-per-pixel scale the same model as for recordings is used.

## 2. Folder layout
| Path | What it is |
|---|---|
| `eegseizure/` | The library (see section 3). |
| `gui.py`, `predict.py`, `live_demo.py` | The three programs for users: window, command line, live-stream demo. |
| `train_models.py` | Trains and evaluates the single-channel and image models; writes `models/` and `results/`. |
| `compare_models.py` | Headroom check against SVM / random forest / gradient boosting. |
| `train_multichannel.py` | Experimental CHB-MIT stage (leave-one-patient-out evaluation). |
| `train_seizure_type_template.py` | Untested template for seizure-type classification (needs the TUH corpus). |
| `make_samples.py` | Creates `samples/inputs` and `samples/outputs`. |
| `build_release.py` | Assembles the repository / release folder from the results (README, report, checksums, zip). |
| `models/` | Trained model files (`*.joblib`). |
| `results/` | Metrics (`*.json`), figures, logs, checksums. |
| `docs/USER_GUIDE.md` | The guide shown inside the program. |
| `tests/` | Unit tests (`python -m pytest tests`). |

## 3. The `eegseizure` package
| Module | Responsibility |
|---|---|
| `constants.py` | Sampling rates, window length, frequency bands, feature names, bipolar channel list. |
| `features.py` | `feats_batch` (19 features), `feats_ext` (36), `mc_features` (multichannel summary), `preprocess` (band-pass), `resample`. |
| `readers.py` | `load_signals` (any supported file -> signals), `read_edf`, `to_bipolar`. |
| `models.py` | `fit_cp` (training), `cp_proba` / `cp_label` (prediction), `top_features` (explanations), `load_models`. |
| `events.py` | `find_events`, `merge_events`, `score_events` (event-based scoring), `movavg`. |
| `images.py` | `digitize` (image -> traces), simulated image rendering for training and testing. |
| `analysis.py` | `analyse_rec`, `analyse_mc`, `analyse_image` - each returns one **result dict** (documented at the top of the file). |
| `stream.py` | `StreamDetector` (ring buffer, sliding window, alarm smoothing). |
| `report.py` | `summarise`, `results_table`, `plot_result`, `html_report`, `write_outputs`, `training_report_markdown`. |
| `minimarkdown.py` | Tiny Markdown reader used by the GUI (Information tab) and the HTML export. |
| `data.py` | `load_bonn` (training data), `test_split` (the project-wide 80/20 split). |

## 4. Model files
`models/*.joblib` are plain Python dicts ("bundles") that contain only scikit-learn objects and numbers:
- `model` - the fitted pipeline (standardise -> optional degree-2 interactions -> logistic regression);
- `q0`, `q1`, `alpha` - conformal thresholds for the two classes and the target error rate;
- `lin` - a linear surrogate used only to name the main drivers of an event;
- `names`, `idx`, `ext` - feature names, selected feature columns, and whether the extended feature set is used;
- `cal`, `params` - calibration scores and the chosen hyper-parameters.
`chb_models.joblib` (experimental) holds `det`, `pre` pipelines and their thresholds. Models are tied to the scikit-learn version recorded in `results/metrics.json`.

## 5. Design decisions (and why)
- **Logistic regression.** Explainable, fast and good enough: `compare_models.py` shows SVM, random forest and gradient boosting are not significantly better on this data.
- **Conformal abstention.** A "not sure" answer is safer than a forced guess. Thresholds come from out-of-fold training scores, so the deployed model can use all training data.
- **Image mode needs a scale.** Without uV per pixel, the size of the waves is lost and accuracy drops, so the accurate path asks for the scale.
- **Held-out evidence.** Hyper-parameters and feature sets are chosen by cross-validation on the training part only; the test set is used once for reporting.
- **One report module.** GUI, command line and sample outputs all call `report.summarise`, so they never disagree.

## 6. Reproducing everything
~~~
pip install -r requirements-train.txt
python -m pytest tests
python train_models.py          # models, metrics, figures
python make_samples.py          # sample inputs and reference outputs
python compare_models.py        # headroom check
python train_multichannel.py    # optional, downloads ~3 GB
python build_release.py         # README, TRAINING_REPORT, checksums, zip
~~~
All random seeds are fixed (seed 0); versions and the data checksum are recorded in `results/metrics.json`. The Colab notebook in `notebooks/` runs the same commands.

## 7. Tests
`tests/test_core.py` covers features, resampling, events, conformal labels, bipolar channels, digitisation round trip, calibrated and uncalibrated image analysis, streaming, the report module and the Markdown reader.
