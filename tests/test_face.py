import numpy as np

from pipeline.face import FaceAnalyzer, FaceTrack, detect_blinks, eye_aspect_ratio, landmark_metrics


def test_eye_aspect_ratio_open_vs_closed():
    open_eye = np.array([[0, 0], [1, 1], [2, 1], [3, 0], [2, -1], [1, -1]], dtype=float)
    closed = open_eye * np.array([1, 0.1])
    assert eye_aspect_ratio(open_eye) > 0.6
    assert eye_aspect_ratio(closed) < 0.1


def test_detect_blinks_counts_dips():
    t = np.arange(0, 10, 0.1)
    ear = np.full_like(t, 0.3)
    for blink_at in (2.0, 5.0, 8.0):
        ear[(t >= blink_at) & (t < blink_at + 0.15)] = 0.1
    assert [round(b, 1) for b in detect_blinks(t, ear)] == [2.0, 5.0, 8.0]


def _fake_landmarks(nose_dx: float) -> np.ndarray:
    lm = np.zeros((478, 2))
    lm[33], lm[263] = [100, 100], [200, 100]  # eye outer corners, IOD = 100 px
    lm[1] = [150 + nose_dx, 140]
    for i, (x, y) in {61: (125, 180), 291: (175, 180), 13: (150, 175), 14: (150, 185)}.items():
        lm[i] = [x, y]
    return lm


def test_yaw_sign_follows_nose_offset():
    assert landmark_metrics(_fake_landmarks(0))["yaw"] == 0
    assert landmark_metrics(_fake_landmarks(15))["yaw"] > 20  # nose to image-right = speaker's left
    assert landmark_metrics(_fake_landmarks(-15))["yaw"] < -20


def test_face_track_interpolates_and_respects_gaps():
    face = {"frames": {"t": [0.0, 0.1, 0.2, 3.0], "bbox": [[0, 0, 10, 10], None, [10, 0, 10, 10], None]}}
    track = FaceTrack.from_face(face)
    assert track.at(0.1)[0] == 5.0
    assert track.at(3.0) is None  # 2.8 s from the last detection


def test_face_analyzer_skips_when_no_face(synthetic_video, make_ctx):
    from pipeline.ingest import IngestAnalyzer

    ctx = make_ctx(synthetic_video)
    IngestAnalyzer().run(ctx)
    env = FaceAnalyzer().run(ctx)
    assert env["status"] == "skipped"


def test_face_analyzer_on_real_fixture(real_fixture, make_ctx):
    from pipeline.ingest import IngestAnalyzer

    ctx = make_ctx(real_fixture)
    IngestAnalyzer().run(ctx)
    env = FaceAnalyzer().run(ctx)
    assert env["status"] == "ok"
    assert env["data"]["summary"]["detection_rate"] > 0.8


def test_looking_down_is_not_a_blink():
    t = np.arange(0, 10, 0.1)
    ear = np.full_like(t, 0.3)
    ear[(t >= 3.0) & (t < 4.5)] = 0.12   # 1.5 s low EAR = looking at notes
    ear[(t >= 7.0) & (t < 7.2)] = 0.1    # a real blink
    assert [round(b, 1) for b in detect_blinks(t, ear)] == [7.0]
