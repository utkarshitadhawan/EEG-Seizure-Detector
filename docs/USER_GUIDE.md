# EEG Seizure Detector - user guide

**Research prototype - not a medical device.** It must not be used for diagnosis, treatment decisions or patient monitoring. A qualified clinician must review every EEG.

## 1. What this program does
It screens an EEG for seizure activity and tells you **where** (start and end times) and **how sure** it is. You can give it:
- **a recording file** (EDF, BDF, CSV, TXT or NPY) - this is the accurate, professional mode;
- **a picture of an EEG** (PNG or JPG screenshot or scan) - a quick screening mode that reads the curves from the picture.

The program is already trained. You do not need to run any training code.

## 2. Quick start (two minutes)
- Open the **Analyse** tab. In this **Information** tab, press **Try a recording example** or **Try an image example**.
- Press **Analyse**. A red banner means seizure activity was found; the plot shows where.
- Look at the files in the `samples` folder (button above): `samples/inputs` shows every accepted file format, `samples/outputs` shows what the program writes.

## 3. The Analyse tab
- **1. Input**: choose Recording or Image, then **Browse** for the file.
- **2. Settings**: they depend on the mode (see sections 4 and 5). **Decision threshold** and **Min. consecutive windows** can be changed after the analysis without re-running.
- **Verdict banner**, a line of statistics, the **plot**, and the **event table**.
- Buttons: **Export CSV**, **Export report (HTML)**, **Save figure**, **Clear**.

## 4. Recording mode: accepted input files
The recording is cut into windows of 1.03 s (178 samples). Each window gets a seizure probability. The recording must be at least 1.03 s long. **Amplitudes must be in microvolts (uV).**

| Format | What the file must look like | Settings you must give |
|---|---|---|
| EDF / BDF | Standard EEG file with one or more channels. | Choose a channel, or "All EEG channels". The sampling rate is read from the file. |
| CSV or TXT, one column | One number per line = one channel (comma, semicolon or space separated is fine). Header lines are ignored. | Sampling rate in Hz (default 173.61). |
| CSV or TXT, several columns | One channel per column, one row per time sample. | Sampling rate in Hz. All columns are analysed; a window is flagged if any channel is. |
| CSV, rows of 178 | One 178-sample segment per row (the UCI / Kaggle layout), optional header `X1...X178` and optional label column `y`. | None. Each row is analysed as an independent 1.03 s segment. |
| NPY | A NumPy array: 1-D (one channel) or 2-D shaped (channels, samples) or (samples, channels). | Sampling rate in Hz. |

- Example files for every row of this table are in `samples/inputs`.
- Other sampling rates are converted to 173.61 Hz automatically.
- **Band-pass 0.5-40 Hz** (on by default) matches the filtering of the training data; switch it off only for pre-filtered signals.
- **Model**: *Single-channel segments* is the main model. *Multichannel 18-ch* is experimental, needs an EDF with the 18 standard bipolar channels (or the electrodes to derive them), and produces many false alarms.

## 5. Image mode: accepted pictures and settings
Use a **clean screenshot or export** (PNG, JPG, BMP, TIFF) of the EEG viewer: dark traces on a light background (dark themes work too), channels stacked vertically. Photographs of a screen (glare, tilt) and traces that overlap each other work badly.

| Setting | What to enter |
|---|---|
| Seconds shown across the image width | The time covered by the picture from its left to its right edge (read the time axis; typical pages are 10, 15 or 30 s). A wrong value distorts all frequencies. |
| Channels in image | 0 = detect automatically. If detection looks wrong, type the exact number of stacked channels. |
| Trim left / right % | Cut away channel labels, scale bars or margins at the picture edges. |
| uV per pixel (optional but important) | How many microvolts one vertical pixel represents. See below. |

**How to find uV per pixel.** Look for a scale bar in the picture. A bar labelled 50 uV that is 25 pixels tall means 50 / 25 = **2 uV per pixel**. (If the viewer shows a sensitivity such as 7 uV/mm, divide it by the number of pixels per millimetre in your screenshot.)

- **With uV per pixel (calibrated mode):** the curves are converted to microvolts and checked by the full model. This is the accurate mode. If your number is off by more than about 30% the accuracy drops (see the Training evidence tab).
- **Without it (uncalibrated mode):** the amplitude is unknown, so reliability is low. If it flags something, the banner says "POSSIBLE" and at least 3 consecutive windows are required; it can also miss real seizures or only report uncertain windows (the uncalibrated sample result in `samples/outputs` shows this). Treat it only as a hint.

## 6. Reading the result
| Banner | Meaning |
|---|---|
| Red - SEIZURE ACTIVITY DETECTED | At least one event: consecutive windows above the threshold. |
| Orange - NO CONFIDENT SEIZURE, N UNCERTAIN | Nothing flagged, but some windows are ambiguous. Review them (shaded orange in the plot). |
| Orange - POSSIBLE ... (uncalibrated) | Image mode without a scale: low reliability. |
| Green - NO SEIZURE DETECTED | Nothing flagged and nothing uncertain. |

- **Plot**: top = the EEG signal (or the image with the digitised traces in red); bottom = seizure probability per window with the threshold as a dashed line. Red shading = detected events, orange = uncertain windows.
- **Event table**: start, end and duration (seconds), peak and mean probability, and the **main drivers** - the signal properties that pushed the model towards "seizure" (for example "high delta power").
- **Uncertain** windows are ones the model is not sure about. They are deliberately not forced into yes/no: look at them yourself.
- **Threshold** lower = more sensitive but more false alarms. **Min. consecutive windows** higher = fewer false alarms but short events are missed.

## 7. Saving results
- **Export CSV** - one row per window: `start_s`, `end_s`, `probability`, `flagged`, `in_event`, `status` (non-seizure / uncertain / seizure), one probability column per channel when there are several channels.
- **Export report (HTML)** - verdict, statistics, figure, event table and the checksum of the input file; opens in any browser.
- **Save figure** - the plot as PNG or PDF.
- Reference outputs for every sample input are in `samples/outputs`.

## 8. Command line and live demo
~~~
python predict.py samples/inputs --outputs my_results
python predict.py image.png --seconds 10 --uv-per-px 2
python live_demo.py samples/inputs/recording_seizure.csv
~~~
`predict.py` handles files or whole folders and writes the same CSV / HTML / figure files as the buttons. `live_demo.py` replays a file as a live stream.

## 9. Limits and safe use
- The models were trained on a small public research data set (single-channel recordings of a few adults). Accuracy on new patients and on real clinical EEG is unknown and will be lower.
- On continuous multichannel EEG the experimental model raises many false alarms; its pre-seizure ("early warning") score is not predictive and is disabled.
- Image mode was tested on simulated images only.
- Always review the EEG yourself. The program can be wrong.

## 10. Troubleshooting
- **A model file could not be loaded** - the installed scikit-learn version differs from the one used for training; use START_HERE.bat (it installs the right version) or `pip install -r requirements.txt`.
- **EDF will not open** - install the `mne` package, or export the data to CSV.
- **Image: wrong number of channels** - set "Channels in image" by hand and trim labels.
- **Everything is "uncertain"** - check the units (must be uV) and the sampling rate.
- **"Channels missing for the multichannel model"** - use the single-channel model instead.
