# Changelog

## 1.0.0 - 2026-11-11
First public release.
- Single-channel segment detector (logistic regression) with an "uncertain - review" zone and per-event explanations.
- Image mode: trace digitisation with optional amplitude calibration (uV per pixel).
- Desktop application with built-in user guide and training evidence, command-line tool, CSV / HTML / figure export, live-stream demo.
- Experimental multichannel CHB-MIT stage (leave-one-patient-out evaluation); the pre-seizure score is disabled because it was not predictive.
- Trained models, readable training report, sample inputs and reference outputs, unit tests, continuous integration.
