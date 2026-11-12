# results/

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
