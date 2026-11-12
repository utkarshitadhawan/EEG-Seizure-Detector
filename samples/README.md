# Sample files

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
| `recording_4channels.edf` | EDF, 4 channels, 256 Hz | Recording mode; choose a channel or "All EEG channels". The sampling rate is read from the file. |
| `image_4channels_seizure.png` | image of 4 stacked channels, 10 s across the width | Image mode: seconds = 10, channels = 0 (auto), **uV per pixel = 8.0**. |
| `image_4channels_normal.png` | same | Image mode with the same settings. |

Leave *uV per pixel* empty to see the weak "uncalibrated" image mode: it may miss the seizure or only report uncertain windows, which is why the scale matters.

## outputs/  (what the program writes)
For every input above the program writes three files (the GUI's *Export CSV*, *Export report* and *Save figure* buttons):
- `<name>_results.csv` - one row per analysis window: `start_s`, `end_s`, `probability`, `flagged`, `in_event`, `status` (non-seizure / uncertain / seizure) and, for several channels, one probability column per channel.
- `<name>_report.html` - verdict, statistics, figure and event table; opens in any browser.
- `<name>_figure.png` - the plot shown in the program.
