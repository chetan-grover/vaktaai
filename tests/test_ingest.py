from pipeline.ingest import IngestAnalyzer, iter_frames, load_audio


def test_ingest_normalises_video_and_audio(synthetic_video, make_ctx):
    ctx = make_ctx(synthetic_video)
    env = IngestAnalyzer().run(ctx)
    assert env["status"] == "ok", env["error"]
    data = env["data"]
    assert ctx.video_path.exists() and ctx.audio_path.exists()
    assert abs(data["duration_s"] - 12.0) < 0.3
    assert data["has_audio"] is True
    assert data["video"]["fps"] == 30
    samples, sr = load_audio(ctx.audio_path)
    assert sr == 16000 and abs(len(samples) / sr - 12.0) < 0.3


def test_frame_sampling_at_10fps(synthetic_video, make_ctx):
    ctx = make_ctx(synthetic_video)
    IngestAnalyzer().run(ctx)
    frames = list(iter_frames(ctx.video_path, fps=10))
    assert 118 <= len(frames) <= 122
    assert frames[1][1] - frames[0][1] == 0.1 or abs(frames[1][1] - frames[0][1] - 0.1) < 0.02


def test_duration_limits_reject_short_recordings(synthetic_video, make_ctx, monkeypatch):
    from pipeline.settings import get_settings

    monkeypatch.setenv("VAKTA_MIN_DURATION_S", "30")
    get_settings.cache_clear()
    env = IngestAnalyzer().run(make_ctx(synthetic_video))
    assert env["status"] == "error"
    assert "minimum" in env["error"]


def test_write_json_survives_a_locked_target(make_ctx, monkeypatch):
    """Windows: replace() fails while a reader holds the file; the write must still land."""
    import json
    from pathlib import Path

    ctx = make_ctx()
    calls = {"n": 0}

    def always_locked(self, target):
        calls["n"] += 1
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "replace", always_locked)
    ctx.write_json("progress.json", {"ok": 1})
    assert json.loads(ctx.path("progress.json").read_text()) == {"ok": 1}
    assert calls["n"] >= 2 and not ctx.path("progress.tmp").exists()
