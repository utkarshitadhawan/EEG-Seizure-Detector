import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import eegseizure as z                                  # noqa: E402
from eegseizure import minimarkdown, report             # noqa: E402
from matplotlib.figure import Figure                    # noqa: E402

rng = np.random.default_rng(0)
t = np.arange(z.N) / z.FS


def seg(seiz):
    return rng.normal(0, 20, z.N) + (rng.uniform(80, 300) * np.sin(2 * np.pi * rng.uniform(3, 6) * t) if seiz else 0)


def small_model():
    y = np.array([0, 1] * 700)
    F = z.feats_batch(np.array([seg(s) for s in y]))
    return z.fit_cp(F, y, alpha=0.05)


def test_features_shape():
    assert z.feats_batch(rng.normal(size=(5, z.N))).shape == (5, 19)


def test_resample_length():
    assert abs(len(z.resample(np.zeros(2560), 256.0)) - 1736) <= 2


def test_events_and_scoring():
    ev = z.find_events(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool), 2)
    assert ev == [(1, 2), (4, 6)]
    assert z.merge_events([(0, 2), (4, 6), (30, 32)], 3) == [(0, 6), (30, 32)]
    assert z.score_events([(100, 120)], [(95, 110), (500, 510)]) == (1, 1, 1)


def test_conformal_labels_and_surrogate():
    cp, info = small_model()
    assert set(np.unique(info["lab"])) <= {0, 1, 2}
    assert info["acc_decided"] >= 0.95
    assert isinstance(z.top_features(cp, z.feats_batch(np.array([seg(1) for _ in range(3)]))), str)


def test_bipolar_from_referential():
    names = [f"EEG {e}-REF" for e in sorted({x for p in z.PAIRS for x in p.split("-")})]
    B, miss = z.to_bipolar(rng.normal(size=(len(names), 100)), names)
    assert B is not None and B.shape == (18, 100) and not miss


def test_digitize_roundtrip():
    tt = np.arange(int(10 * z.FS)) / z.FS
    s1, s2 = np.sin(2 * np.pi * 3 * tt), np.sin(2 * np.pi * 8 * tt)
    lanes, info = z.digitize(z.render_multilane([s1, s2], seconds=10, pps=150, seed=0))
    assert len(lanes) == 2
    for L, ref in zip(lanes, (s1, s2)):
        sig = z.resample(L["y"], info["W"] / 10.0)
        n = min(len(sig), len(ref))
        assert np.corrcoef(sig[:n], ref[:n])[0, 1] > 0.9


def test_image_analysis_runs():
    y = np.array([0, 1] * 300)
    Fi, _ = z.make_image_dataset(np.array([seg(s) for s in y]), seed=0)
    cpi, _ = z.fit_cp(Fi, y, idx=z.SF_IDX)
    sigs = [np.concatenate([seg(0) for _ in range(10)]), np.concatenate([seg(1) for _ in range(10)])]
    r = z.analyse_image(z.render_multilane(sigs, seconds=10, pps=150), cpi, seconds=10)
    assert r["n"] >= 8 and len(r["names"]) == 2 and r["probs"][1].mean() > r["probs"][0].mean()


def test_cross_conformal_calibration():
    cp, info = small_model()
    assert len(cp["cal"][0]) == len(info["itr"]) and cp["model"].predict_proba(np.zeros((1, 19)))[0].sum() > 0.99


def test_extended_features():
    assert z.feats_ext(rng.normal(size=(4, z.N))).shape == (4, len(z.PLAIN_EXT))


def test_calibrated_image_mode():
    y = np.array([0, 1] * 700)
    cp, _ = z.fit_cp(z.feats_batch(np.array([seg(s) for s in y])), y, alpha=0.05)
    Fi, _ = z.make_image_dataset(np.array([seg(s) for s in y[:600]]), seed=0)
    cpi, _ = z.fit_cp(Fi, y[:600], idx=z.SF_IDX)
    sigs = [np.concatenate([seg(0) for _ in range(10)]), np.concatenate([seg(0) for _ in range(5)] + [seg(1) for _ in range(5)])]
    img = z.render_multilane(sigs, seconds=10, pps=150, lane_h=200, uv_per_px=2.0)
    cal = z.analyse_image(img, cpi, seconds=10, uv_per_px=2.0, cp_seg=cp)
    unc = z.analyse_image(img, cpi, seconds=10)
    assert cal["calibrated"] and cal["kmin_min"] == 1 and not unc["calibrated"] and unc["kmin_min"] == 3
    assert cal["probs"][1][-3:].mean() > 0.8 and cal["probs"][0].max() < 0.5


def test_stream_detector():
    cp, _ = small_model()
    det = z.StreamDetector(cp, "single", z.FS)
    out = det.push(np.concatenate([seg(0) for _ in range(10)]))
    assert len(out) >= 8 and {"t", "p", "alarm"} <= set(out[0])


# ------------------------------------------------------------------ input formats and reports
def test_input_formats(tmp_path):
    """Every documented input layout is read correctly."""
    cp, _ = small_model()
    one = np.concatenate([seg(0) for _ in range(5)])
    np.savetxt(tmp_path / "one.csv", one)
    assert z.analyse_rec(str(tmp_path / "one.csv"), z.FS, "Auto", cp)["n"] == 5
    np.savetxt(tmp_path / "multi.csv", np.c_[one, one], delimiter=",", header="a,b", comments="")
    assert len(z.analyse_rec(str(tmp_path / "multi.csv"), z.FS, "Auto", cp)["names"]) == 2
    rows = np.array([seg(0) for _ in range(6)])
    np.savetxt(tmp_path / "rows.csv", np.c_[rows, np.ones(6)], delimiter=",", header=",".join(f"X{i}" for i in range(1, 179)) + ",y", comments="")
    r = z.analyse_rec(str(tmp_path / "rows.csv"), z.FS, "Auto", cp)
    assert r["rows"] and r["n"] == 6
    np.save(tmp_path / "ch.npy", np.array([one, one]))
    assert len(z.analyse_rec(str(tmp_path / "ch.npy"), z.FS, "Auto", cp)["names"]) == 2


def test_report_outputs(tmp_path):
    """summarise / results_table / write_outputs agree and produce the three reference files."""
    cp, _ = small_model()
    np.savetxt(tmp_path / "s.csv", np.concatenate([seg(0) for _ in range(5)] + [seg(1) for _ in range(5)]))
    r = z.analyse_rec(str(tmp_path / "s.csv"), z.FS, "Auto", cp)
    s = report.summarise(r, 0.5, 1)
    assert s["level"] in ("seizure", "uncertain", "none") and len(report.results_table(r)) == r["n"]
    report.write_outputs(r, str(tmp_path / "out"))
    assert all((tmp_path / f"out_{k}").exists() for k in ("results.csv", "report.html", "figure.png"))
    fig = Figure()
    report.plot_result(fig, r, 0.5, s["events"])
    assert "<html" in report.html_report(r, fig)


def test_markdown_reader():
    blocks = minimarkdown.parse("# T\n\ntext **b**\n\n- x\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert [b[0] for b in blocks] == ["h1", "p", "li", "table"] and blocks[3][1] == [["a", "b"], ["1", "2"]]
    assert "<table>" in minimarkdown.to_html("| a | b |\n|---|---|\n| 1 | 2 |")
