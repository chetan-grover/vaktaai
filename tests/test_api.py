"""End-to-end through the API: sign up -> upload -> background pipeline -> report -> drills -> history -> delete."""
import math

import pytest
from fastapi.testclient import TestClient

SCRIPT = ("Have you ever wondered why some talks stick? um Today I will, like, share three ideas. "
          "First, start with a story. Then, slow down. Finally, look at everyone in the room. "
          "So remember, practise today.")
PASSWORD = "correct horse battery"


@pytest.fixture
def client(monkeypatch, helpers):
    import pipeline.asr as asr

    def fake_transcribe(path, size, language):
        words = helpers.fake_words(SCRIPT, start=0.3, rate=0.3)
        return {"language": "en", "language_probability": 0.99, "text": SCRIPT, "segments": [], "words": words}

    monkeypatch.setattr(asr, "transcribe", fake_transcribe)
    from app.main import app

    with TestClient(app) as c:
        yield c


def signup(client, name="Asha", email="asha@example.com", password=PASSWORD):
    resp = client.post("/api/auth/register", json={"name": name, "email": email, "password": password})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    return {"Authorization": f"Bearer {body['token']}"}, body


def upload(client, headers, video, title="Demo"):
    with video.open("rb") as fh:
        return client.post("/api/sessions", headers=headers, data={"title": title, "source": "upload"},
                           files={"file": ("talk.mp4", fh, "video/mp4")})


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["models"]["e3_head"] is False


def test_register_login_logout(client):
    headers, body = signup(client, email="Asha@Example.com ")
    assert body["user"]["email"] == "asha@example.com"
    assert client.get("/api/auth/me", headers=headers).json()["name"] == "Asha"

    assert client.post("/api/auth/register", json={"name": "X", "email": "asha@example.com", "password": PASSWORD}).status_code == 409
    assert client.post("/api/auth/register", json={"name": "X", "email": "x@example.com", "password": "short"}).status_code == 422
    assert client.post("/api/auth/register", json={"name": "X", "email": "not-an-email", "password": PASSWORD}).status_code == 422

    assert client.post("/api/auth/login", json={"email": "asha@example.com", "password": "wrong password"}).status_code == 401
    login = client.post("/api/auth/login", json={"email": "ASHA@example.com", "password": PASSWORD})
    assert login.status_code == 200
    second = {"Authorization": f"Bearer {login.json()['token']}"}

    assert client.post("/api/auth/logout", headers=headers).status_code == 204
    assert client.get("/api/auth/me", headers=headers).status_code == 401
    assert client.get("/api/auth/me", headers=second).status_code == 200


def test_routes_require_sign_in(client):
    for method, url in [("get", "/api/sessions"), ("get", "/api/sessions/abc"), ("get", "/api/me/progress"),
                        ("get", "/api/me/persona"), ("delete", "/api/sessions/abc")]:
        assert getattr(client, method)(url).status_code == 401, url
    assert client.get("/api/sessions", headers={"Authorization": "Bearer forged"}).status_code == 401


def test_full_session_lifecycle(client, synthetic_video):
    headers, body = signup(client)
    resp = upload(client, headers, synthetic_video)
    assert resp.status_code == 201, resp.text
    sid = resp.json()["id"]
    assert resp.json()["user_id"] == body["user"]["id"]

    session = client.get(f"/api/sessions/{sid}", headers=headers).json()  # background task already ran in TestClient
    assert session["status"] == "done", session
    stages = session["progress"]["stages"]
    assert stages["ingest"]["status"] == "ok" and stages["report"]["status"] == "ok"
    assert stages["face"]["status"] == "skipped"  # synthetic video has no face; report still builds

    report = client.get(f"/api/sessions/{sid}/report", headers=headers).json()
    assert report["traits"]["confidence"]["score"] is not None
    assert report["content"]["structure"]["opener"]["type"] == "question"
    assert report["voice"]["summary"]["filler_count"] >= 2
    assert "indicative" in report["disclaimer"].lower()
    assert report["reproducibility"]["models"]["asr"].startswith("faster-whisper")
    assert len(report["timeline"]["channels"]["confidence"]) == math.ceil(report["duration_s"])

    plan = report["drills"]["plan"]
    if plan:
        drill = plan[0]["id"]
        assert client.patch(f"/api/sessions/{sid}/drills/{drill}", headers=headers, json={"done": True}).json()["done"] is True
        assert client.get(f"/api/sessions/{sid}/report", headers=headers).json()["drill_progress"][drill] is True

    token = headers["Authorization"].split()[1]
    video = client.get(f"/api/sessions/{sid}/video?token={token}", headers={"Range": "bytes=0-99"})
    assert video.status_code in (200, 206)
    assert client.get(f"/api/sessions/{sid}/video").status_code == 401

    assert len(client.get("/api/sessions", headers=headers).json()) == 1
    progress = client.get("/api/me/progress", headers=headers).json()
    assert progress["points"][0]["confidence"] is not None
    persona = client.get("/api/me/persona", headers=headers).json()
    assert persona["kind"] == "rule-axes" and len(persona["points"]) == 1

    assert client.delete(f"/api/sessions/{sid}/video", headers=headers).json()["video_deleted"] is True
    assert client.get(f"/api/sessions/{sid}/video?token={token}").status_code == 404
    assert client.delete(f"/api/sessions/{sid}", headers=headers).status_code == 204
    assert client.get(f"/api/sessions/{sid}", headers=headers).status_code == 404


def test_users_cannot_see_each_others_sessions(client, synthetic_video):
    asha, _ = signup(client)
    ravi, _ = signup(client, name="Ravi", email="ravi@example.com")
    sid = upload(client, asha, synthetic_video).json()["id"]
    for method, url in [("get", f"/api/sessions/{sid}"), ("get", f"/api/sessions/{sid}/report"),
                        ("get", f"/api/sessions/{sid}/video"), ("delete", f"/api/sessions/{sid}"),
                        ("post", f"/api/sessions/{sid}/reanalyse")]:
        assert getattr(client, method)(url, headers=ravi).status_code == 404, url
    assert client.get("/api/sessions", headers=ravi).json() == []
    assert client.get(f"/api/sessions/{sid}", headers=asha).status_code == 200


def test_signup_claims_legacy_name_only_profile(client):
    from sqlmodel import Session

    from app.db import SpeechSession, User, get_engine

    with Session(get_engine()) as db:
        legacy = User(name="chetan")
        db.add(legacy)
        db.commit()
        db.refresh(legacy)
        db.add(SpeechSession(id="old1", user_id=legacy.id, title="old talk", status="done"))
        db.commit()
        legacy_id = legacy.id
    headers, body = signup(client, name="chetan", email="chetan@example.com")
    assert body["user"]["id"] == legacy_id
    assert [s["id"] for s in client.get("/api/sessions", headers=headers).json()] == ["old1"]


def test_rejects_bad_uploads(client):
    headers, _ = signup(client, name="B", email="b@example.com")
    resp = client.post("/api/sessions", headers=headers, data={"title": "x"}, files={"file": ("notes.txt", b"hi", "text/plain")})
    assert resp.status_code == 415


def test_delete_during_analysis_cancels_the_job(make_ctx, monkeypatch):
    """Deleting a session folder mid-run stops the pipeline at the next stage and leaves no orphan folder."""
    import shutil

    import pipeline.runner as runner
    from worker import jobs

    ctx = make_ctx(session_id="gone")
    original = runner.run_stage

    def delete_after_ingest(cls, c, progress=None):
        if cls.name == "asr":
            shutil.rmtree(c.session_dir)
        return original(cls, c, progress)

    monkeypatch.setattr(runner, "run_stage", delete_after_ingest)
    monkeypatch.setattr(runner.IngestAnalyzer, "run", lambda self, c: {"status": "ok", "elapsed_s": 0})
    jobs._run_session_job("gone")
    assert not ctx.session_dir.exists()
