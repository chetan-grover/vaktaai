import numpy as np

from pipeline.physio import PhysioAnalyzer, chrom, heart_rate, pos, rmssd_ms, stress_series, windowed_estimates

FPS = 30.0


def _trace(bpm: float, seconds: float = 20, noise: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FPS)) / FPS
    pulse = np.sin(2 * np.pi * bpm / 60 * t)
    base = np.array([180.0, 120.0, 100.0])
    rgb = base + np.outer(pulse, [0.3, 0.8, 0.2]) + noise * rng.standard_normal((len(t), 3))
    return rgb + np.outer(np.linspace(0, 3, len(t)), [1, 1, 1])  # slow illumination drift


def test_pos_recovers_heart_rate():
    bpm, sqi = heart_rate(pos(_trace(72), FPS), FPS)
    assert abs(bpm - 72) < 3 and sqi > 0.5


def test_chrom_recovers_heart_rate():
    bpm, _ = heart_rate(chrom(_trace(90), FPS), FPS)
    assert abs(bpm - 90) < 3


def test_noise_only_has_low_quality():
    rng = np.random.default_rng(1)
    rgb = np.array([180.0, 120.0, 100.0]) + rng.standard_normal((600, 3))
    _, sqi = heart_rate(pos(rgb, FPS), FPS)
    assert sqi < 0.5


def test_rmssd_of_a_regular_pulse_is_small():
    t = np.arange(int(20 * FPS)) / FPS
    value = rmssd_ms(np.sin(2 * np.pi * 1.2 * t), FPS, 72)
    assert value is not None and value < 15


def test_windows_and_stress_curve():
    rgb = np.concatenate([_trace(70, 25, seed=2), _trace(95, 25, seed=3) + 3])
    windows = windowed_estimates(pos(rgb, FPS), FPS)
    bpm_s, stress_s, summary = stress_series(windows, 50)
    assert abs(summary["bpm_baseline"] - 70) < 5
    assert summary["bpm_rise"] > 15
    late = [s for s in stress_s[35:] if s is not None]
    assert late and np.mean(late) > 0.5


def test_physio_analyzer_on_synthetic_video(synthetic_video, make_ctx, helpers):
    from conftest import SKIN_BOX
    from pipeline.ingest import IngestAnalyzer

    ctx = make_ctx(synthetic_video)
    IngestAnalyzer().run(ctx)
    t = [round(i * 0.1, 2) for i in range(120)]
    # Landmark bbox for which the forehead/cheek ROIs fall inside the pulsing patch.
    x, y, w, h = SKIN_BOX
    helpers.write_module(ctx, "face", {"frames": {"t": t, "bbox": [[x, y, w, h]] * len(t)}, "fps": 10})
    env = PhysioAnalyzer().run(ctx)
    assert env["status"] == "ok", env["error"]
    data = env["data"]
    assert data["models"]["rppg"] == "consensus(pos,chrom)"  # no PhysNet weights in tests
    reliable = [w for w in data["windows"] if w["reliable"]]
    assert reliable, data["windows"]
    assert all(abs(w["bpm"] - 72) < 4 for w in reliable)
    assert "Indicative" in data["disclaimer"]


def test_consensus_needs_two_agreeing_methods():
    from pipeline.physio import consensus_windows

    def w(bpm, sqi=0.8):
        return [{"start": 0, "end": 10, "bpm": bpm, "rmssd_ms": 40.0, "sqi": sqi, "reliable": True}]

    agree = consensus_windows({"physnet": w(72), "pos": w(75), "chrom": w(110)})[0]
    assert agree["reliable"] and abs(agree["bpm"] - 73.5) < 0.1 and set(agree["agree"]) == {"physnet", "pos"}
    assert not consensus_windows({"physnet": w(60), "pos": w(90), "chrom": w(120)})[0]["reliable"]
    edge = consensus_windows({"physnet": w(44), "pos": w(45), "chrom": w(46)})[0]  # hugging the 42 BPM edge
    assert not edge["reliable"]
