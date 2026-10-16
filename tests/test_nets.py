"""Shape checks for the torch modules used by E1, E2 and E3 (skipped without torch)."""
import pytest

torch = pytest.importorskip("torch")

from pipeline.nets import CrossAttentionFusionHead, GazeSmoother, PhysNet  # noqa: E402


def test_physnet_output_length():
    model = PhysNet(frames=32).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 32, 72, 72))
    assert out.shape == (1, 32)


def test_physnet_state_dict_names_match_toolbox():
    keys = set(PhysNet().state_dict())
    assert {"ConvBlock1.0.weight", "upsample.0.weight", "ConvBlock10.weight", "ConvBlock9.1.running_mean"} <= keys


def test_gaze_smoother_shapes():
    out = GazeSmoother()(torch.randn(2, 50, 4))
    assert out.shape == (2, 50, 9)


def test_fusion_head_outputs_and_masks():
    head = CrossAttentionFusionHead(d_audio=1027, d_face=1282, n_emotions=8).eval()
    audio, face = torch.randn(3, 9, 1027), torch.randn(3, 10, 1282)
    face_mask = torch.zeros(3, 10, dtype=torch.bool)
    face_mask[:, 7:] = True
    with torch.no_grad():
        out = head(audio, face, face_mask=face_mask)
    assert out["emotion_logits"].shape == (3, 8)
    assert out["va"].shape == out["audio_va"].shape == out["face_va"].shape == (3, 2)
    assert out["mismatch_logit"].shape == (3,)
    assert sum(p.numel() for p in head.parameters()) < 3_000_000  # "small head, ~2 M params"


def test_fusion_uses_learned_head_when_present(make_ctx, helpers):
    """Save a random E3 checkpoint + embedding streams and check fusion switches to the learned path."""
    import numpy as np

    from pipeline.fusion import FusionAnalyzer, head_path, load_head
    from pipeline.settings import get_settings

    load_head.cache_clear()
    head = CrossAttentionFusionHead(d_audio=12, d_face=10, d_model=16, heads=2, layers=1, n_emotions=8)
    head_path().parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": head.config, "state_dict": head.state_dict(), "mismatch_threshold": 0.0,
                "emotions": ["e%d" % i for i in range(8)], "norm": {}}, head_path())
    ctx = make_ctx()
    t = np.arange(0.25, 10, 0.5)
    np.savez(ctx.path("audio_emb.npz"), t=t, emb=np.random.randn(len(t), 12).astype("float32"), va=np.zeros((len(t), 2)))
    np.savez(ctx.path("face_emb.npz"), t=t, emb=np.random.randn(len(t), 10).astype("float32"), va=np.zeros((len(t), 2)))
    helpers.write_module(ctx, "ingest", {"duration_s": 10.0})
    env = FusionAnalyzer().run(ctx)
    assert env["status"] == "ok", env["error"]
    assert env["data"]["method"] == "learned+rule"
    seg = env["data"]["segments"][0]
    assert seg["learned"]["mismatch"] is True  # threshold 0 -> every segment flagged
    assert len(env["data"]["mismatches"]) == len(env["data"]["segments"])
    assert FusionAnalyzer(force_rule=True).run(ctx)["data"]["method"] == "rule"


def test_gaze_uses_gru_when_present(make_ctx, helpers):
    from pipeline.gaze import GazeAnalyzer, load_smoother
    from pipeline.settings import get_settings

    load_smoother.cache_clear()
    path = get_settings().models_dir / "gaze" / "gaze_gru.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": {"in_features": 4, "hidden": 8, "cells": 9}, "state_dict": GazeSmoother(4, 8, 9).state_dict()}, path)
    ctx = make_ctx()
    n = 50
    frames = {"t": [i * 0.1 for i in range(n)], "yaw": [0.0] * n, "pitch": [0.0] * n,
              "gaze_yaw": [0.0] * n, "gaze_pitch": [0.0] * n, "bbox": [[0, 0, 1, 1]] * n}
    helpers.write_module(ctx, "ingest", {"duration_s": 5.0})
    helpers.write_module(ctx, "face", {"frames": frames, "fps": 10})
    env = GazeAnalyzer().run(ctx)
    assert env["status"] == "ok", env["error"]
    assert env["data"]["models"]["gaze_smoother"] == "gru"


def test_l2cs_loads_safetensors_and_rejects_partial_checkpoints(tmp_path):
    pytest.importorskip("torchvision")
    from safetensors.torch import save_file

    from pipeline.gaze import load_l2cs
    from pipeline.nets import L2CS

    load_l2cs.cache_clear()
    state = {f"module.{k}": v.contiguous() for k, v in L2CS().state_dict().items()}
    good = tmp_path / "l2cs_gaze360_resnet50.safetensors"
    save_file(state, str(good))
    model = load_l2cs(str(good), "cpu")
    with torch.no_grad():
        yaw, pitch = model(torch.rand(1, 3, 224, 224))
    assert yaw.shape == pitch.shape == (1, 90)

    load_l2cs.cache_clear()
    partial = tmp_path / "partial.safetensors"
    save_file({k: v for k, v in state.items() if "fc_" not in k}, str(partial))
    with pytest.raises(RuntimeError, match="missing"):
        load_l2cs(str(partial), "cpu")
    load_l2cs.cache_clear()
