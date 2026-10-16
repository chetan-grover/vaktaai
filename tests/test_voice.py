import numpy as np

from pipeline.voice import (VoiceAnalyzer, detect_fillers, detect_pauses, energy_track, pitch_track,
                            semitone_stats, wpm_windows)


def test_hard_fillers_always_count(helpers):
    words = helpers.fake_words("so um I think uh this works")
    texts = [f["text"] for f in detect_fillers(words)]
    assert "um" in texts and "uh" in texts


def test_soft_fillers_need_a_pause_or_comma(helpers):
    words = helpers.fake_words("I like pizza and like, you know, pasta")
    texts = [f["text"] for f in detect_fillers(words)]
    assert texts.count("like") == 1  # only the one with a comma
    assert "you know" in texts


def test_hinglish_fillers(helpers):
    words = helpers.fake_words("matlab the plan is toh simple", gaps={1: 0.5, 4: 0.5})
    fillers = detect_fillers(words)
    assert {f["text"] for f in fillers} >= {"matlab", "toh"}
    assert all(f["lang"] == "hinglish" for f in fillers)


def test_pauses_and_wpm(helpers):
    words = helpers.fake_words("one two three four five six seven eight", gaps={4: 1.0})
    pauses = detect_pauses(words)
    assert len(pauses) == 1 and abs(pauses[0]["duration"] - 1.07) < 0.05
    windows = wpm_windows(words, duration=5.0, size=5.0)
    assert windows[0]["words"] == 8 and windows[0]["wpm"] == 96.0


def test_pitch_of_a_pure_tone():
    sr = 16000
    t = np.arange(sr * 2) / sr
    _, f0 = pitch_track((0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)
    assert abs(np.nanmedian(f0) - 200) < 3
    stats = semitone_stats(f0)
    assert stats["std_st"] < 0.2  # a pure tone is perfectly monotone


def test_energy_track_db():
    sr = 16000
    loud = 0.5 * np.ones(sr, dtype=np.float32)
    _, db = energy_track(loud, sr)
    assert abs(db.mean() - 20 * np.log10(0.5)) < 0.1


def test_voice_analyzer_end_to_end(synthetic_video, make_ctx, helpers):
    from pipeline.ingest import IngestAnalyzer

    ctx = make_ctx(synthetic_video)
    IngestAnalyzer().run(ctx)
    words = helpers.fake_words("Hello everyone um today I want to like, share three ideas with you all", gaps={8: 0.6})
    helpers.write_module(ctx, "asr", {"words": words, "text": " ".join(w["w"] for w in words)})
    env = VoiceAnalyzer().run(ctx)
    assert env["status"] == "ok", env["error"]
    s = env["data"]["summary"]
    assert s["word_count"] == 14 and s["filler_count"] >= 2
    assert 140 < s["pitch_mean_hz"] < 210
    assert s["pitch_std_st"] > 0.5  # the synthetic voice glides
    assert len(env["data"]["series"]["pitch_hz"]) in (12, 13)  # ceil(duration incl. AAC padding)
