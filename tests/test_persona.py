import numpy as np
import pytest

from analytics.persona.features import FEATURE_NAMES, NORMS, session_vector, to_array
from analytics.persona.infer import RuleMap, persona_trajectory
from analytics.persona.naming import name_profile


def _vector(**overrides):
    v = {n: NORMS[n][0] for n in FEATURE_NAMES}
    v.update(overrides)
    return v


def test_vector_has_24_features_and_imputes_missing():
    assert len(FEATURE_NAMES) == 24
    report = {"voice": {"summary": {"wpm": 200.0}}, "segments": [{"features": {"head_motion": 3.0}}]}
    vec = session_vector(report)
    assert vec["wpm"] == 200.0 and vec["gaze_coverage"] is None
    arr = to_array(vec)
    assert arr.shape == (24,) and arr[FEATURE_NAMES.index("gaze_coverage")] == 0.0
    assert arr[0] == pytest.approx((200 - 145) / 25)


def test_rule_personas():
    assert name_profile({"wpm": 2.0})[0] == "Fast Talker"
    assert name_profile({"pitch_std_st": -1.5, "energy_std_db": -1.5})[0] == "Monotone Presenter"
    assert name_profile({})[0] == "Developing Speaker"


def test_rule_map_trajectory_moves_with_improvement():
    before = _vector(filler_rate_per_min=9.0, eye_contact_ratio=0.1, gaze_coverage=0.15)
    after = _vector(filler_rate_per_min=1.0, eye_contact_ratio=0.5, gaze_coverage=0.7)
    sessions = [{"session_id": "a", "features": before}, {"session_id": "b", "features": after},
                {"session_id": "c", "features": None}]
    out = persona_trajectory(sessions, "/nonexistent")
    assert out["kind"] == "rule-axes" and len(out["points"]) == 2
    assert out["points"][1]["y"] > out["points"][0]["y"]  # more polish
    assert out["points"][0]["persona"]["name"] in {"Filler-Heavy Speaker", "Note Reader"}
    assert any(r["name"] == "Fast Talker" for r in RuleMap().regions())


def test_vae_training_and_map(tmp_path):
    pytest.importorskip("torch")
    from analytics.persona.infer import VaeMap
    from analytics.persona.train import train

    rng = np.random.default_rng(0)
    vectors = []
    for profile in ({"wpm": 200.0}, {"pitch_std_st": 0.8, "energy_std_db": 2.0}, {"filler_rate_per_min": 10.0}):
        for _ in range(20):
            v = _vector(**profile)
            vectors.append({k: val + rng.normal(0, 0.05 * NORMS[k][1]) for k, val in v.items()})
    result = train(vectors, tmp_path / "persona", latent=2, k=3, epochs=300)
    names = {p["name"] for p in result["meta"]["personas"]}
    assert {"Fast Talker", "Monotone Presenter", "Filler-Heavy Speaker"} <= names
    vmap = VaeMap(tmp_path / "persona")
    assert vmap.assign(_vector(wpm=205.0))["name"] == "Fast Talker"
    assert vmap.project([vectors[0]]).shape == (1, 2)
