"""Assemble the release folder (the GitHub repository) and its zip from the trained models and results.

    python build_release.py        (run after train_models.py, make_samples.py, compare_models.py [, train_multichannel.py])

Creates ./eeg-seizure-detector/ and ./eeg-seizure-detector_v<version>.zip containing the code, trained models, results, readable
training report, sample inputs/outputs, documentation, launchers and repository files (README, MIT license, citation metadata, CI).
"""
import datetime
import hashlib
import json
import os
import shutil

import eegseizure as z
from eegseizure import minimarkdown, report

AUTHOR_GIVEN, AUTHOR_FAMILY = "Utkarshita", "Dhawan"
GITHUB_USER, REPO = "utkarshitadhawan", "eeg-seizure-detector"
VERSION = z.__version__
TODAY, YEAR = datetime.date.today().isoformat(), datetime.date.today().year
TITLE = "EEG Seizure Detector: explainable, uncertainty-aware seizure screening for EEG recordings and EEG images"
R = REPO

CODE_FILES = ["gui.py", "predict.py", "live_demo.py", "train_models.py", "make_samples.py", "compare_models.py",
              "train_multichannel.py", "train_seizure_type_template.py", "build_release.py"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ipynb_checkpoints", "_tmp*")


def write(path, text, newline=None):
    """Write a text file inside the release folder."""
    with open(f"{R}/{path}", "w", encoding="utf-8", newline=newline) as f:
        f.write(text)


# ---------------------------------------------------------------------------------------- copy files
shutil.rmtree(R, ignore_errors=True)
os.makedirs(R)
for name in CODE_FILES:
    shutil.copy(name, f"{R}/{name}")
for folder in ("eegseizure", "docs", "tests", "models", "results", "samples"):
    shutil.copytree(folder, f"{R}/{folder}", ignore=IGNORE)
for folder in (".github/workflows", "notebooks"):
    os.makedirs(f"{R}/{folder}")

M = json.load(open("results/metrics.json"))
V, S = M["versions"], M["segment_model"]
T, CF, CAL = S["test"], S["conformal"], M["calibrated_image"]["scale_error_factor"]
SKV = V["sklearn"]
HAS_CHB = os.path.exists("results/chb_metrics.json")
pct = lambda v: f"{100 * v:.2f}%"
ci = lambda k: f"{pct(T[k][0])} (95% CI {pct(T[k][1][0])} - {pct(T[k][1][1])})"

# ---------------------------------------------------------------------------- readable training report
write("TRAINING_REPORT.md", report.training_report_markdown(R))
write("TRAINING_REPORT.html", report.training_report_html(R))
write("results/README.md", """# results/

Everything the training scripts produced. Read `../TRAINING_REPORT.md` (or `.html`) first; it explains the numbers.

| File | Content |
|---|---|
| `metrics.json` | Segment model, image model and calibrated-image numbers, software versions, data checksum |
| `model_comparison.json` | Headroom check against SVM / random forest / gradient boosting |
| `chb_metrics.json` | Experimental multichannel stage (leave-one-patient-out) |
| `fig1_performance.png` ... `fig7_model_comparison.png` | Figures used in the report |
| `demo_check.json` | Result of the self-check of the sample files |
| `logs/` | Console output of the training run |
| `SHA256SUMS.txt` | Checksums of the models and code |
""")

# ------------------------------------------------------------------------------------------ launchers
bat = r"""@echo off
setlocal
cd /d "%~dp0"
title EEG Seizure Detector
set "ENVDIR=%USERPROFILE%\envs\seizure"
set "VPY=%ENVDIR%\Scripts\python.exe"
set "SKV=@@SKV@@"
if not exist "%~dp0gui.py" goto NOFILES
if not exist "%~dp0eegseizure\__init__.py" goto NOFILES
if not exist "%~dp0models" goto NOFILES
if exist "%VPY%" "%VPY%" -c "import numpy, pandas, scipy, joblib, matplotlib, PIL, mne, tkinter, sklearn; assert sklearn.__version__=='%SKV%'" >nul 2>&1 && goto RUN
set "PY="
python --version >nul 2>&1 && set "PY=python"
if not defined PY py --version >nul 2>&1 && set "PY=py"
if not defined PY goto NOPY
echo.
echo First-time setup (about 3-6 minutes, needs internet). Please wait...
echo.
%PY% -m venv "%ENVDIR%" || goto FAIL
"%VPY%" -m ensurepip --upgrade >nul 2>&1
"%VPY%" -m pip install --upgrade pip >nul 2>&1
"%VPY%" -m pip install "scikit-learn==%SKV%" numpy pandas scipy joblib matplotlib pillow mne || goto FAIL
"%VPY%" -c "import tkinter" >nul 2>&1 || goto NOTK
:RUN
start "" "%ENVDIR%\Scripts\pythonw.exe" "%~dp0gui.py"
exit /b 0
:NOFILES
echo Keep START_HERE.bat, gui.py, the eegseizure folder and the models folder together in ONE folder.
pause
exit /b 1
:NOPY
echo Python is not installed. Trying to install it automatically...
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements || start "" https://www.python.org/downloads/
echo When Python is installed, CLOSE this window and double-click START_HERE.bat again.
echo (If installing by hand from python.org, tick "Add python.exe to PATH".)
pause
exit /b 1
:FAIL
echo SETUP FAILED. Check your internet connection (turn off VPN / try another network), then run this file again.
echo If scikit-learn==%SKV% cannot be found, install Python 3.11 or 3.12.
pause
exit /b 1
:NOTK
echo Python was installed without Tkinter. Re-run the Python installer, choose Modify, tick "tcl/tk and IDLE".
pause
exit /b 1
""".replace("@@SKV@@", SKV)
write("START_HERE.bat", bat, newline="\r\n")
sh = """#!/usr/bin/env bash
# macOS / Linux launcher. On Linux you may first need:  sudo apt install python3-venv python3-tk
cd "$(dirname "$0")"
ENVDIR="$HOME/.venvs/seizure"
if [ ! -x "$ENVDIR/bin/python" ] || ! "$ENVDIR/bin/python" -c "import sklearn, mne, PIL, tkinter; assert sklearn.__version__=='@@SKV@@'" 2>/dev/null; then
  echo "First-time setup (3-6 minutes, needs internet)..."
  python3 -m venv "$ENVDIR" && "$ENVDIR/bin/pip" install -q --upgrade pip && \\
  "$ENVDIR/bin/pip" install -q "scikit-learn==@@SKV@@" numpy pandas scipy joblib matplotlib pillow mne || { echo "Setup failed"; exit 1; }
fi
exec "$ENVDIR/bin/python" gui.py "$@"
""".replace("@@SKV@@", SKV)
write("start.sh", sh, newline="\n")
os.chmod(f"{R}/start.sh", 0o755)

# ---------------------------------------------------------------------------------- repository files
write("requirements.txt", f"scikit-learn=={SKV}\nnumpy\npandas\nscipy\njoblib\nmatplotlib\npillow\nmne\n")
write("requirements-train.txt", f"scikit-learn=={SKV}\nnumpy\npandas\nscipy\njoblib\nmatplotlib\npillow\nmne\nedfio\nkagglehub\npytest\n")
write(".gitignore", "__pycache__/\n*.pyc\n.pytest_cache/\n.ipynb_checkpoints/\ndata.csv\nchbmit_data/\n.venv/\n*.zip\n")
write(".gitattributes", "*.bat text eol=crlf\n*.sh text eol=lf\n*.joblib binary\n*.png binary\n*.edf binary\n*.npy binary\n")
write(".github/workflows/tests.yml", """name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install -r requirements.txt pytest
      - run: python -m pytest -q tests
""")
write("LICENSE", f"""MIT License

Copyright (c) {YEAR} {AUTHOR_GIVEN} {AUTHOR_FAMILY}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""")
write("CITATION.cff", f"""cff-version: 1.2.0
message: "If you use this software, please cite it as below."
title: "{TITLE}"
type: software
authors:
  - family-names: "{AUTHOR_FAMILY}"
    given-names: "{AUTHOR_GIVEN}"
version: "{VERSION}"
date-released: "{TODAY}"
license: MIT
repository-code: "https://github.com/{GITHUB_USER}/{REPO}"
abstract: "Open-source research prototype for screening recorded EEG and EEG images for seizure activity with logistic regression, conformal abstention and per-event explanations. Not a medical device."
keywords: [EEG, epilepsy, seizure detection, logistic regression, conformal prediction, image digitisation]
""")
write(".zenodo.json", json.dumps(dict(
    title=TITLE, upload_type="software", access_right="open", license="mit", version=VERSION, language="eng",
    description="Open-source research prototype for screening pre-recorded EEG (EDF/BDF/CSV/TXT/NPY) and EEG images (PNG/JPG) for seizure activity. "
                "Logistic regression with conformal abstention, per-event explanations, image digitisation with optional amplitude calibration, "
                "a desktop application with built-in user guide and training evidence, a command-line tool, an experimental multichannel stage and "
                "a live-stream demo. Trained models are included. Research prototype - not a medical device.",
    creators=[dict(name=f"{AUTHOR_FAMILY}, {AUTHOR_GIVEN}")],
    keywords=["EEG", "epilepsy", "seizure detection", "machine learning", "logistic regression", "conformal prediction", "explainable AI"],
    related_identifiers=[dict(identifier=f"https://github.com/{GITHUB_USER}/{REPO}", relation="isSupplementTo", scheme="url")]), indent=2))
write("CHANGELOG.md", f"""# Changelog

## {VERSION} - {TODAY}
First public release.
- Single-channel segment detector (logistic regression) with an "uncertain - review" zone and per-event explanations.
- Image mode: trace digitisation with optional amplitude calibration (uV per pixel).
- Desktop application with built-in user guide and training evidence, command-line tool, CSV / HTML / figure export, live-stream demo.
- Experimental multichannel CHB-MIT stage (leave-one-patient-out evaluation); the pre-seizure score is disabled because it was not predictive.
- Trained models, readable training report, sample inputs and reference outputs, unit tests, continuous integration.
""")
write("CONTRIBUTING.md", """# Contributing
Thanks for your interest! This is a research prototype, so the most useful contributions are:
- **Bug reports** with the file type, the exact steps, the error text and your Python / scikit-learn versions.
- **Real-world validation**: results on EEG recordings or screenshots you are allowed to share (no patient-identifiable data, please).
- **Tests and documentation** improvements.

Read `docs/CODE_GUIDE.md` first. Run `python -m pytest tests` before opening a pull request. Please do not add claims of clinical performance.
""")

# ----------------------------------------------------------------------------------------- README
chb_line = ("An experimental multichannel stage (leave-one-patient-out on CHB-MIT) is documented in the training report: it detects seizures with many "
            "false alarms, and its pre-seizure score is not predictive (disabled)." if HAS_CHB else
            "The experimental multichannel stage is not included in this build (`python train_multichannel.py`).")
D = dict(USER=GITHUB_USER, REPO=REPO, SKV=SKV, VERSION=VERSION, NT=T["n_test"], ACC=ci("accuracy"), SEN=ci("sensitivity"), SPE=ci("specificity"),
         AUC=T["roc_auc"], UNC=CF["uncertain_pct"], AD=pct(CF["accuracy_on_decided"]), CAL1=pct(CAL["1.0"]["accuracy"]),
         CALS=pct(CAL["1.0"]["sensitivity"]), CHB=chb_line, IMG_UNCAL=pct(M["image_model"]["accuracy_digitised_trained_on_digitised"]),
         FSEL=S["feature_selection"]["chosen"])
readme = """# EEG Seizure Detector

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10--3.13-blue)
[![tests](https://github.com/@@USER@@/@@REPO@@/actions/workflows/tests.yml/badge.svg)](https://github.com/@@USER@@/@@REPO@@/actions)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/@@USER@@/@@REPO@@/blob/main/notebooks/reproduce_in_colab.ipynb)

A desktop program (and command-line tool) that screens **EEG recordings and EEG images** for seizure activity, shows where it found it, explains why,
and says "uncertain - please review" when it is not sure. **The trained models are included: download, double-click, use. No training needed.**

> **IMPORTANT: research prototype, not a medical device.** It must not be used for diagnosis, treatment decisions or patient monitoring.
> A qualified clinician must review every EEG. The accuracy figures come mainly from a small public research data set and do **not** show how
> the program performs on new patients or real clinical recordings.

## Download and use (no training required)
1. Download this repository as a zip (**Code > Download ZIP**, or the zip in the **Releases** page) and unzip it.
2. **Windows:** double-click **`START_HERE.bat`**. The first start installs the needed Python packages (3-6 minutes, internet needed); later starts open instantly.
   **macOS / Linux:** `./start.sh` (Linux: `sudo apt install python3-venv python3-tk` first).
3. In the program, open the **Information** tab: *1. How to use and input formats* explains everything, with buttons that load a ready-made example;
   *2. Training evidence* shows the graphs and numbers from training.

Manual install: `pip install -r requirements.txt` then `python gui.py`. Needs Python 3.10-3.13 with Tkinter. The models were trained with
**scikit-learn @@SKV@@** and may not load with other versions, so `requirements.txt` pins it.

## What you get
| Part | Description |
|---|---|
| **Recording mode** | EDF, BDF, CSV, TXT or NPY. The signal is cut into 1.03 s windows; you get a verdict, an event table (start / end / duration), a plot and exports. |
| **Image mode** | PNG/JPG screenshot or scan of an EEG. The curves are read from the picture; enter the vertical scale (**uV per pixel**) for accurate results. |
| **Uncertain zone** | Ambiguous windows are marked "uncertain" instead of being forced into yes / no. |
| **Explanations** | Each detected event lists the signal features that drove it. |
| **In-program guide** | Information tab: usage, accepted input formats, how to find the image scale, and the training evidence (graphs). |
| **Sample files** | `samples/inputs` (one example per input format) and `samples/outputs` (what the program writes). |
| **Training report** | `TRAINING_REPORT.md` / `.html`: all results of the training run in readable form. |
| **Command line / live demo** | `predict.py` for files and folders; `live_demo.py` replays a file as a live stream. |

## Results at a glance
Held-out test set (n = @@NT@@ one-second segments from the Bonn / UCI data):

| Metric | Value |
|---|---|
| Accuracy | @@ACC@@ |
| Sensitivity (seizures found) | @@SEN@@ |
| Specificity | @@SPE@@ |
| ROC-AUC | @@AUC@@ |

- **Uncertain zone:** @@UNC@@% of segments are sent to review; on the rest accuracy is @@AD@@.
- **Image mode:** accuracy @@CAL1@@ (sensitivity @@CALS@@) when the uV-per-pixel scale is entered correctly, but only @@IMG_UNCAL@@ without it - on *simulated* images.
- **Nonlinear models** (SVM, random forest, gradient boosting) are not significantly better than the logistic regression used here.
- @@CHB@@

Full details, all figures and the sanity checks: **[TRAINING_REPORT.md](TRAINING_REPORT.md)**. A web version is in `TRAINING_REPORT.html`.

![performance](results/fig1_performance.png)

## How it works (short)
Windows -> 19 signal features (feature set used: @@FSEL@@) -> logistic regression -> conformal thresholds (non-seizure / uncertain / seizure) -> events.
Images are digitised (trace pixels -> signal) and, with the scale known, analysed by the same model. The code is organised in the `eegseizure` package;
read **[docs/CODE_GUIDE.md](docs/CODE_GUIDE.md)** for the architecture, and **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)** for usage.

## Limitations and responsible use
- The Bonn data have no patient IDs; neighbouring segments of one recording can land in both training and test sets, so the accuracy is optimistic
  and says nothing about unseen patients. Continuous clinical EEG is much harder (many false alarms).
- Image mode was tested only on simulated images; a wrong scale or a photo of a screen hurts accuracy.
- The single-channel model was trained on a few adults; the multichannel model on a few children and is experimental.
- Amplitudes must be in microvolts. Outputs can be wrong - always review the EEG yourself.

## Reproduce the results (optional)
The trained models and results are already in this repository. To reproduce them: open `notebooks/reproduce_in_colab.ipynb` in Google Colab (badge above)
or run locally:
~~~
pip install -r requirements-train.txt
python -m pytest tests
python train_models.py          # models, metrics, figures
python make_samples.py          # sample inputs and reference outputs
python compare_models.py        # headroom check against nonlinear models
python train_multichannel.py    # optional: experimental stage, downloads ~3 GB
python build_release.py         # README, TRAINING_REPORT, checksums, zip
~~~
Seed 0. Software versions and the SHA-256 of the data file are in `results/metrics.json`; checksums of models and code in `results/SHA256SUMS.txt`.
Data: Kaggle `harunshimanto/epileptic-seizure-recognition` (a CSV copy of the UCI / Bonn data); CHB-MIT from PhysioNet.

## Repository layout
~~~
START_HERE.bat  start.sh          launchers
gui.py  predict.py  live_demo.py  the program, command line, live demo
eegseizure/                       library (see docs/CODE_GUIDE.md)
models/                           trained models (used by the program)
results/                          metrics, figures, logs
samples/inputs  samples/outputs   example inputs and reference outputs
docs/                             USER_GUIDE.md (also shown in the program), CODE_GUIDE.md
train_models.py  make_samples.py  compare_models.py  train_multichannel.py  build_release.py     reproduction scripts
tests/                            unit tests
TRAINING_REPORT.md / .html  MODEL_CARD.md  CHANGELOG.md  CITATION.cff  LICENSE
~~~

## FAQ and troubleshooting
- **"A model file could not be loaded"** - install the pinned scikit-learn: `pip install scikit-learn==@@SKV@@`, or use `START_HERE.bat`.
- **EDF will not open** - `pip install mne`; or export to CSV. **No window on Linux** - `sudo apt install python3-tk`.
- **Image: wrong channel count** - set *Channels* by hand and trim labels. **Everything "uncertain"** - check units (uV) and sampling rate.
- **Can I use it clinically?** No.
More in the guide (Information tab).

## Citation
Cite the software via `CITATION.cff` (Zenodo mints a DOI for every GitHub release) and the data:
- Andrzejak RG, Lehnertz K, Mormann F, Rieke C, David P, Elger CE (2001). Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity. *Phys. Rev. E* 64, 061907.
- Shoeb AH (2009). Application of machine learning to epileptic seizure onset detection and treatment. PhD thesis, MIT; Goldberger AL et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23), e215-e220 (CHB-MIT).
Check each data set's terms before redistributing data; this repository contains trained models, figures and small demo excerpts only.

## License
MIT - see [LICENSE](LICENSE) (code, trained models and documentation). Data sets keep their own licenses. See also [MODEL_CARD.md](MODEL_CARD.md), [CHANGELOG.md](CHANGELOG.md), [CONTRIBUTING.md](CONTRIBUTING.md).
"""
for key, value in D.items():
    readme = readme.replace(f"@@{key}@@", str(value))
assert "@@" not in readme
write("README.md", readme)

write("MODEL_CARD.md", f"""# Model card - EEG Seizure Detector v{VERSION}

**Intended use:** research and education; a screening aid for pre-recorded EEG files and EEG images, always with human review.
**Out of scope:** diagnosis, treatment decisions, real-time patient monitoring, neonatal/pediatric use, unsupervised clinical deployment.

## Models
| File | Input | Method |
|---|---|---|
| models/seizure_model_cp.joblib | 1-channel 1.03 s segments (178 samples, 173.61 Hz); also used by calibrated image mode | {S['feature_selection']['chosen']} -> logistic regression (grid-searched), cross-conformal abstention, linear explanation surrogate |
| models/image_model_cp.joblib | digitised images, amplitude unknown (uncalibrated fallback) | 11 amplitude-independent features -> logistic regression trained on simulated rendered + digitised images |
| models/chb_models.joblib | 18 bipolar channels, 4 s windows, 256 Hz (experimental) | mean / max / std of per-channel features -> logistic regression (detection; pre-seizure score disabled) |

## Data
Bonn / UCI Epileptic Seizure Recognition (11,500 segments; seizure = class 1 vs 2-5), seed {V['seed']}, data SHA-256 `{V['data_sha256']}`.
CHB-MIT Scalp EEG subset (PhysioNet) for the experimental stage. Training environment: Python {V['python']}, scikit-learn {SKV}, numpy {V['numpy']}, scipy {V['scipy']}.

## Evaluation (held-out test set, n = {T['n_test']})
Accuracy {ci('accuracy')}, sensitivity {ci('sensitivity')}, specificity {ci('specificity')}, ROC-AUC {T['roc_auc']}.
Abstention: {CF['uncertain_pct']}% uncertain, accuracy {pct(CF['accuracy_on_decided'])} on decided segments.
Image mode (simulated images): calibrated accuracy {pct(CAL['1.0']['accuracy'])}, uncalibrated {pct(M['image_model']['accuracy_digitised_trained_on_digitised'])}.
Everything is explained in `TRAINING_REPORT.md`; machine-readable numbers are in `results/*.json`.

## Known limitations and risks
No patient-wise evaluation on Bonn data (optimistic); simulated rather than real image tests; many false alarms on continuous multichannel EEG;
pre-seizure score not predictive (disabled); amplitude units assumed to be microvolts; small training data of unknown demographics.
Outputs can be wrong - always review the EEG. Not a medical device.

## Maintenance
scikit-learn is pinned to {SKV} (models may not load with other versions). Report issues on GitHub.
""")

# ------------------------------------------------------- public reproduce notebook (clones the repository)
md = lambda t: {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True)}
code = lambda t: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(True)}
notebook = {"cells": [
    md("# Reproduce the EEG Seizure Detector training\n"
       "**You do not need to run this to use the program** - the trained models are already in the repository (`models/`) and the results are in "
       "`TRAINING_REPORT.md`. This notebook re-creates them from scratch so anyone can verify the numbers. Not a medical device.\n\n"
       "If the install cell changes scikit-learn, use *Runtime > Restart session* and continue from the next cell."),
    code(f"!git clone https://github.com/{GITHUB_USER}/{REPO}.git\n%cd {REPO}\n!pip -q install -r requirements-train.txt"),
    code("# unit tests\n!python -m pytest -q tests"),
    code("# train the models, evaluate on held-out data, make figures (10-15 min)\n!python train_models.py"),
    code("# create the sample input files and the reference output files\n!python make_samples.py"),
    code("# headroom check against nonlinear models (5-10 min)\n!python compare_models.py"),
    code("# OPTIONAL experimental stage: downloads ~3 GB from PhysioNet (20-60 min). Remove the leading # to run.\n# !python train_multichannel.py"),
    code("# rebuild the readable training report from the results and show the figures\n"
         "from eegseizure import report\nfrom IPython.display import Image, display\nimport glob\n"
         "open('TRAINING_REPORT.md', 'w', encoding='utf-8').write(report.training_report_markdown('.'))\n"
         "for f in sorted(glob.glob('results/fig*.png')):\n    print(f); display(Image(f, width=900))"),
], "metadata": {"colab": {"name": "reproduce_in_colab.ipynb"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "nbformat": 4, "nbformat_minor": 5}
write("notebooks/reproduce_in_colab.ipynb", json.dumps(notebook, indent=1))
assert minimarkdown.parse(open(f"{R}/TRAINING_REPORT.md", encoding="utf-8").read())        # the report parses

# --------------------------------------------------------------------------------------- checksums, zip
with open(f"{R}/results/SHA256SUMS.txt", "w") as fh:
    for f in sorted(os.listdir(f"{R}/models")):
        fh.write(hashlib.sha256(open(f"{R}/models/{f}", "rb").read()).hexdigest() + f"  models/{f}\n")
    for f in CODE_FILES + [f"eegseizure/{n}" for n in sorted(os.listdir("eegseizure")) if n.endswith(".py")]:
        fh.write(hashlib.sha256(open(f"{R}/{f}", "rb").read()).hexdigest() + f"  {f}\n")
shutil.make_archive(f"{REPO}_v{VERSION}", "zip", ".", R)
print("Built", f"{REPO}_v{VERSION}.zip")
