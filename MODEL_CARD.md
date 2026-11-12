# Model card - EEG Seizure Detector v1.0.0

**Intended use:** research and education; a screening aid for pre-recorded EEG files and EEG images, always with human review.
**Out of scope:** diagnosis, treatment decisions, real-time patient monitoring, neonatal/pediatric use, unsupervised clinical deployment.

## Models
| File | Input | Method |
|---|---|---|
| models/seizure_model_cp.joblib | 1-channel 1.03 s segments (178 samples, 173.61 Hz); also used by calibrated image mode | base (19 features) -> logistic regression (grid-searched), cross-conformal abstention, linear explanation surrogate |
| models/image_model_cp.joblib | digitised images, amplitude unknown (uncalibrated fallback) | 11 amplitude-independent features -> logistic regression trained on simulated rendered + digitised images |
| models/chb_models.joblib | 18 bipolar channels, 4 s windows, 256 Hz (experimental) | mean / max / std of per-channel features -> logistic regression (detection; pre-seizure score disabled) |

## Data
Bonn / UCI Epileptic Seizure Recognition (11,500 segments; seizure = class 1 vs 2-5), seed 0, data SHA-256 `5ac55c71de5df20d2f47c39300c5b61df18ac5d1c40315e586310b794c32a98b`.
CHB-MIT Scalp EEG subset (PhysioNet) for the experimental stage. Training environment: Python 3.13.18, scikit-learn 1.6.1, numpy 2.4.6, scipy 1.16.3.

## Evaluation (held-out test set, n = 2300)
Accuracy 99.00% (95% CI 98.61% - 99.35%), sensitivity 96.52% (95% CI 94.87% - 98.16%), specificity 99.62% (95% CI 99.30% - 99.89%), ROC-AUC 0.9992.
Abstention: 2.26% uncertain, accuracy 99.42% on decided segments.
Image mode (simulated images): calibrated accuracy 98.22%, uncalibrated 92.30%.
Everything is explained in `TRAINING_REPORT.md`; machine-readable numbers are in `results/*.json`.

## Known limitations and risks
No patient-wise evaluation on Bonn data (optimistic); simulated rather than real image tests; many false alarms on continuous multichannel EEG;
pre-seizure score not predictive (disabled); amplitude units assumed to be microvolts; small training data of unknown demographics.
Outputs can be wrong - always review the EEG. Not a medical device.

## Maintenance
scikit-learn is pinned to 1.6.1 (models may not load with other versions). Report issues on GitHub.
