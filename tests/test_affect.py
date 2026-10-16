import numpy as np

from pipeline.affect import AudioAffectAnalyzer, FaceAffectAnalyzer, landmark_proxy_va, prosody_proxy_va


def test_prosody_proxy_tracks_loudness():
    n = 20
    energy = [-30.0] * 10 + [-10.0] * 10
    pitch = [150.0] * n
    t, va = prosody_proxy_va({"series": {"energy_db": energy, "pitch_hz": pitch}})
    assert va.shape == (n, 2) and np.all(np.abs(va) <= 1)
    assert va[15, 1] > va[5, 1]  # louder -> higher arousal


def test_landmark_proxy_tracks_smile():
    n = 40
    smile = [0.45] * 20 + [0.6] * 20
    frames = {"t": list(np.arange(n) * 0.1), "smile": smile, "corner_lift": [0.0] * 20 + [0.05] * 20,
              "brow": [0.2] * n, "mouth_open": [0.05] * n, "ear": [0.3] * n}
    _, va = landmark_proxy_va({"frames": frames})
    assert va[30, 0] > va[5, 0]


def test_audio_affect_falls_back_to_proxy_offline(make_ctx, helpers):
    ctx = make_ctx()
    helpers.write_module(ctx, "ingest", {"duration_s": 10.0, "has_audio": True})
    helpers.write_module(ctx, "voice", {"series": {"energy_db": list(np.linspace(-30, -10, 10)),
                                                   "pitch_hz": [150.0] * 10}})
    env = AudioAffectAnalyzer().run(ctx)
    assert env["status"] == "ok", env["error"]
    assert env["data"]["proxy"] is True and env["data"]["models"]["audio_va"] == "prosody-proxy"
    assert len(env["data"]["per_second"]["arousal"]) == 10


def test_face_affect_falls_back_to_landmarks(make_ctx, helpers):
    ctx = make_ctx()
    n = 50
    frames = {"t": [i * 0.1 for i in range(n)], "smile": list(np.linspace(0.4, 0.6, n)),
              "corner_lift": [0.0] * n, "brow": [0.2] * n, "mouth_open": [0.05] * n, "ear": [0.3] * n,
              "bbox": [[0, 0, 10, 10]] * n}
    helpers.write_module(ctx, "ingest", {"duration_s": 5.0})
    helpers.write_module(ctx, "face", {"frames": frames, "fps": 10})
    env = FaceAffectAnalyzer().run(ctx)
    assert env["status"] == "ok", env["error"]
    assert env["data"]["models"]["face_va"] == "landmark-proxy"
    assert env["data"]["summary"]["expressiveness"] is not None
