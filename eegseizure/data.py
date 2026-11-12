"""Loading the training data (Bonn / UCI "Epileptic Seizure Recognition")."""
import glob
import hashlib
import os

import numpy as np
import pandas as pd

KAGGLE_DATASET = "harunshimanto/epileptic-seizure-recognition"     # a CSV copy of the UCI / Bonn data


def load_bonn():
    """Return (X, y, sha256) for the Bonn data.

    X: (11500, 178) segments in microvolts.  y: 1 = seizure (original class 1), 0 = everything else (classes 2-5).
    Source order: the CSV named in the environment variable BONN_CSV, else the Kaggle copy (via kagglehub),
    else - on Google Colab only - a file upload dialog.
    """
    path = os.environ.get("BONN_CSV")
    if not path:
        try:
            import kagglehub
            path = glob.glob(kagglehub.dataset_download(KAGGLE_DATASET) + "/**/*.csv", recursive=True)[0]
        except Exception as e:                                                  # no internet / Kaggle unavailable
            print("Automatic download failed:", e)
            from google.colab import files
            print("Please upload the Kaggle 'Epileptic Seizure Recognition' data.csv now")
            path = list(files.upload().keys())[0]
    df = pd.read_csv(path)
    cols = [c for c in df.columns if str(c).startswith("X") and str(c)[1:].isdigit()]
    X = df[cols].values.astype(float)
    y = (df["y"].values == 1).astype(int)
    return X, y, hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_split(y, seed=0):
    """Indices of the (train, test) split used everywhere in this project (80 / 20, stratified)."""
    from sklearn.model_selection import train_test_split
    return train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
