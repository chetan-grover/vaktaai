"""Fusion (rule baseline), timeline events and drill mapping on hand-built module outputs."""
import numpy as np

from pipeline.drills import DrillsAnalyzer
from pipeline.fusion import FusionAnalyzer, goodness, load_trait_config, rule_mismatch, trait_scores
from pipeline.timeline import TimelineAnalyzer, build_segments

DURATION = 30.0


def test_segments_snap_to_sentence_ends(helpers):
    words = helpers.fake_words("one two three four five six seven eight nine ten. " * 6, start=0.0, rate=0.5)
    segs = build_segments(DURATION, words, 5.0)
    assert segs[0]["start"] == 0 and segs[-1]["end"] == DURATION
    assert all(abs((s["end"] - s["start"]) - 5) <= 1.5 for s in segs[:-1])
    # boundaries land on sentence ends ('ten.' words end at 4.9, 9.9, ...)
    assert segs[0]["end"] == 4.9


def test_trait_scores_and_missing_modalities():
    cfg = load_trait_config()
    good = goodness({"wpm": 140, "filler_rate": 0, "pause_ratio": 0.1, "on_grid": 0.9, "head_motion": 2}, cfg)
    scores = trait_scores(good, cfg)
    assert scores["confidence"] > 90
    assert trait_scores(goodness({"wpm": 140}, cfg), cfg)["confidence"] is None  # too little signal


def test_rule_mismatch_smile_with_bad_news():
    cfg = load_trait_config()
    reasons = rule_mismatch({"face_valence": 0.6, "text_sentiment": -1.0, "audio_arousal": 0.0}, cfg)
    assert reasons == ["Smiling while saying something negative"]
    assert rule_mismatch({"face_valence": 0.6, "text_sentiment": 0.5}, cfg) == []


def _build_session(ctx, helpers):
    n = int(DURATION)
    text = ("Good morning everyone. um I am, like, really nervous. uh we failed badly and lost everything terrible. "
            "But we learned a lot. so we kept going and it worked. Thank you.")
    words = helpers.fake_words(text, start=0.5, rate=0.25)  # fast speech: 240 wpm
    helpers.write_module(ctx, "ingest", {"duration_s": DURATION})
    helpers.write_module(ctx, "asr", {"words": words, "text": text})
    from pipeline.voice import detect_fillers, detect_pauses

    helpers.write_module(ctx, "voice", {
        "summary": {"wpm": 240.0, "filler_rate_per_min": 8.0, "pitch_std_st": 1.2, "energy_std_db": 5.0,
                    "pause_ratio": 0.1, "long_pause_count": 0},
        "fillers": detect_fillers(words), "pauses": detect_pauses(words),
        "series": {"pitch_hz": list(150 + 5 * np.sin(np.arange(n))), "energy_db": [-20.0] * n}})
    t = [i * 0.5 for i in range(2 * n)]
    face_val = [0.7 if 5 <= ti < 15 else 0.0 for ti in t]  # smiling through the bad news
    helpers.write_module(ctx, "face_va", {"t": t, "valence": face_val, "arousal": [0.0] * len(t), "proxy": False,
                                          "per_second": {"valence": face_val[::2], "arousal": [0.0] * n},
                                          "summary": {"expressiveness": 0.05}})
    helpers.write_module(ctx, "audio_va", {"t": t, "valence": [-0.4] * len(t), "arousal": [-0.5] * len(t),
                                           "proxy": False, "per_second": {"valence": [-0.4] * n, "arousal": [-0.5] * n},
                                           "summary": {}})
    windows = [{"start": s, "end": s + 10, "bpm": 72 if s < 10 else 92, "rmssd_ms": 40, "sqi": 0.8, "reliable": True}
               for s in range(0, 25, 5)]
    helpers.write_module(ctx, "physio", {"windows": windows,
                                         "summary": {"bpm_baseline": 72, "bpm_rise": 20},
                                         "series": {"bpm": [72] * 10 + [92] * 20, "stress": [0.0] * 10 + [0.8] * 20}})


def test_fusion_timeline_drills_pipeline(make_ctx, helpers):
    ctx = make_ctx()
    _build_session(ctx, helpers)
    fusion = FusionAnalyzer().run(ctx)
    assert fusion["status"] == "ok", fusion["error"]
    f = fusion["data"]
    assert f["method"] == "rule"
    assert set(f["traits"]) == {"confidence", "authoritativeness", "engagement"}
    assert all("why" in t for t in f["traits"].values())
    assert f["mismatches"], "smiling through bad news should be flagged"
    assert f["congruence"]["score"] is not None

    timeline = TimelineAnalyzer().run(ctx)
    assert timeline["status"] == "ok", timeline["error"]
    tl = timeline["data"]
    assert len(tl["channels"]["confidence"]) == int(DURATION)
    assert tl["counts"]["mismatch"] >= 1 and tl["counts"]["bpm_spike"] >= 1
    assert tl["counts"]["filler_cluster"] >= 1

    drills = DrillsAnalyzer().run(ctx)
    assert drills["status"] == "ok", drills["error"]
    ids = {w["id"] for w in drills["data"]["weaknesses"]}
    assert {"too_fast", "fillers", "physio_stress"} <= ids
    fast = next(w for w in drills["data"]["weaknesses"] if w["id"] == "too_fast")
    assert 1 <= len(fast["drills"]) <= 3 and fast["evidence"]["segment"] is not None


def test_missing_transcript_and_proxy_affect_do_not_inflate_traits(make_ctx, helpers):
    """No ASR -> filler/pace unknown (not 'perfect'); proxy V-A -> used for congruence only."""
    ctx = make_ctx()
    n = int(DURATION)
    helpers.write_module(ctx, "ingest", {"duration_s": DURATION})
    helpers.write_module(ctx, "voice", {"summary": {}, "fillers": [], "pauses": [],
                                        "series": {"pitch_hz": [150.0] * n, "energy_db": [-20.0] * n}})
    t = [i * 0.5 for i in range(2 * n)]
    for name in ("audio_va", "face_va"):
        helpers.write_module(ctx, name, {"t": t, "valence": [0.0] * len(t), "arousal": [0.0] * len(t), "proxy": True,
                                         "per_second": {"valence": [0.0] * n, "arousal": [0.0] * n}, "summary": {}})
    f = FusionAnalyzer().run(ctx)["data"]
    seg = f["segments"][0]
    assert "filler_rate" not in seg["features"] and "fluency" not in seg["goodness"]
    assert "vocal_arousal" not in seg["goodness"] and "facial_warmth" not in seg["goodness"]
    assert seg["congruence_distance"] == 0.0  # proxies still feed congruence
    assert f["traits"]["confidence"]["score"] is None
