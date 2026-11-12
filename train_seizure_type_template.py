"""TEMPLATE (untested on real data): seizure-TYPE classifier on the TUH EEG Seizure Corpus (TUSZ).
TUSZ needs free registration at https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ (credentials by e-mail).
Download the corpus, then:   python train_seizure_type_template.py /path/to/tusz/edf/train
Groups windows by patient folder (GroupKFold) so no patient is in both train and test.
Types are merged into focal vs generalized because the fine-grained classes are too small to learn reliably."""
import glob, os, sys
import numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import eegseizure as z

TYPE = {"fnsz": "focal", "cpsz": "focal", "spsz": "focal", "gnsz": "generalized", "absz": "generalized",
        "tnsz": "generalized", "tcsz": "generalized", "cnsz": "generalized", "mysz": "generalized"}


def read_events(csv_path):
    """TUSZ .csv annotation: lines 'channel,start,stop,label,confidence' (comments start with #)."""
    ev = set()
    for ln in open(csv_path):
        if ln.startswith("#") or ln.startswith("channel") or not ln.strip():
            continue
        p = ln.strip().split(",")
        if p[3] in TYPE:
            ev.add((float(p[1]), float(p[2]), TYPE[p[3]]))
    return sorted(ev)


def main(root):
    X, y, g = [], [], []
    for edf in glob.glob(os.path.join(root, "**", "*.edf"), recursive=True):
        csv = edf[:-4] + ".csv"
        if not os.path.isfile(csv):
            continue
        ev = read_events(csv)
        if not ev:
            continue
        try:
            data, sf, names = z.read_edf(edf)
            B, _ = z.to_bipolar(data, names)
            if B is None:
                continue
            F = z.mc_features(z.preprocess(z.resample(B, sf, z.FS_MC), z.FS_MC))
        except Exception as e:
            print("skip", edf, e)
            continue
        tc = np.arange(len(F)) * 2.0 + 2.0
        for s, e, t in ev:
            m = (tc >= s) & (tc <= e)
            X.append(F[m]); y += [t] * int(m.sum()); g += [edf.split(os.sep)[-3]] * int(m.sum())
    X, y, g = np.vstack(X), np.array(y), np.array(g)
    print("windows:", len(y), {t: int((y == t).sum()) for t in set(y)}, "patients:", len(set(g)))
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000))
    pred = cross_val_predict(pipe, X, y, groups=g, cv=GroupKFold(5))
    print("patient-wise balanced accuracy:", balanced_accuracy_score(y, pred))
    print(confusion_matrix(y, pred, labels=sorted(set(y))))
    os.makedirs("models", exist_ok=True)
    joblib.dump(dict(model=pipe.fit(X, y), classes=sorted(set(y))), "models/seizure_type_model.joblib")


if __name__ == "__main__":
    main(sys.argv[1])
