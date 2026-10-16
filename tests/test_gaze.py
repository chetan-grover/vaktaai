import numpy as np

from pipeline.gaze import CENTER, GazeAnalyzer, angles_to_cell, calibrate, grid_summary, mode_filter


def test_angles_map_to_grid_from_speaker_view():
    assert angles_to_cell(0, 0) == CENTER
    assert angles_to_cell(20, 10) == 0      # up and to the speaker's left: back-left
    assert angles_to_cell(-20, -10) == 8    # down and right: front-right
    assert angles_to_cell(70, 0) == -1      # looking away
    assert angles_to_cell(float("nan"), 0) == -1


def test_calibration_uses_camera_window():
    t = np.arange(0, 10, 0.1)
    yaw = np.where(t < 3, 5.0, 25.0)
    pitch = np.full_like(t, -4.0)
    yaw0, pitch0, method = calibrate(t, yaw, pitch, 3.0, yaw_absolute=True)
    assert (yaw0, pitch0, method) == (5.0, -4.0, "camera-window")
    yaw0, _, method = calibrate(t, yaw, pitch, None, yaw_absolute=True)
    assert yaw0 == 0.0 and method == "session-median"


def test_mode_filter_removes_single_frame_flicker():
    assert mode_filter([4, 4, 1, 4, 4, 4, 2, 2, 2, 2]) == [4, 4, 4, 4, 4, 4, 2, 2, 2, 2]


def test_grid_summary_flags_neglected_zones():
    t = [i * 0.1 for i in range(100)]
    cells = [4] * 50 + [5] * 30 + [2] * 20
    s = grid_summary(t, cells, 10.0)
    assert s["eye_contact_ratio"] == 0.5
    assert s["coverage"] == round(3 / 9, 3)
    assert set(s["neglected_cells"]) == {0, 1, 3, 6, 7, 8}


def test_gaze_analyzer_one_sided_speaker(make_ctx, helpers):
    ctx = make_ctx()
    n = 300
    t = [round(i * 0.1, 2) for i in range(n)]
    # First 3 s at the camera, then only to the speaker's right (negative yaw).
    yaw = [0.0 if ti < 3 else -18.0 for ti in t]
    frames = {"t": t, "yaw": yaw, "pitch": [0.0] * n, "gaze_yaw": yaw, "gaze_pitch": [0.0] * n,
              "bbox": [[0, 0, 10, 10]] * n}
    helpers.write_module(ctx, "ingest", {"duration_s": 30.0})
    helpers.write_module(ctx, "face", {"frames": frames, "fps": 10})
    ctx.write_json("meta.json", {"source": "webcam", "calibration_s": 3})
    env = GazeAnalyzer().run(ctx)
    assert env["status"] == "ok", env["error"]
    data = env["data"]
    assert data["calibration"]["method"] == "camera-window"
    assert data["summary"]["dwell"][5] > 0.8  # mid-right
    assert 3 in data["summary"]["neglected_cells"]  # mid-left never visited
    assert any("mid-left" in a["message"] for a in data["alerts"])
    assert len(data["per_second"]) == 30
