"""API tests through FastAPI's TestClient. No GPU, no Docker, no Ollama.

Needs:  pip install fastapi httpx pytest      Run:  pytest tests/test_api.py

"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("POSTGRES_URL", None)   # tests use in-memory SQLite, never your real database
os.environ.pop("DATABASE_URL", None)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import note_generator_aftercare as ng  # noqa: E402
from db import Store  # noqa: E402


class FakeNotes:
    adapter_path = "/app/models/llama-2-aftercare-v4"
    model_version = "llama-2-aftercare-v4"
    enabled = True
    demo_mode = False

    def __init__(self, text="Knee incision redness since yesterday.", flags=None):
        self.text, self.flags = text, flags or []

    def disabled_reason(self):
        return None

    def generate_with_flags(self, turns):
        return self.text, list(self.flags)


@pytest.fixture()
def client(monkeypatch):
    fresh = Store("")
    fresh.init_schema()
    monkeypatch.setattr(api, "store", fresh)

    async def fake_llm(*args, **kwargs):
        return "Thank you, a nurse will review this."
    monkeypatch.setattr(api, "generate_llm_response", fake_llm)
    monkeypatch.setattr(api, "note_generator", ng.NoteGenerator(adapter_path=""))  # disabled by default
    with TestClient(api.app) as c:
        yield c


def _check_in(symptoms, vitals=None, pid="P001", comorbidities=None):
    return {"patient": {"patient_id": pid, "age": 68, "comorbidities": comorbidities or []},
            "symptoms": symptoms, "vitals": vitals or {}, "message": "help"}


def test_health_reports_database_and_note_status(client):
    r = client.get("/health").json()
    assert r["status"] == "healthy" and "sqlite" in r["database"] and r["note_generation"].startswith("disabled")


def test_emergency_check_in_creates_alert_and_is_stored(client):
    body = _check_in([{"name": "chest_pain", "severity": 9}, {"name": "dyspnea", "severity": 8}])
    r = client.post("/check-in", json=body).json()
    assert r["urgency_level"] == "EMERGENCY" and r["nurse_alert"] is True
    assert client.get(f"/check-in/{r['check_in_id']}").json()["llm_response"] == "Thank you, a nurse will review this."
    assert client.get("/alerts/pending").json()["count"] == 1
    assert client.get("/stats").json()["total_alerts"] == 1


def test_routine_check_in_has_no_alert(client):
    r = client.post("/check-in", json=_check_in([{"name": "mild_headache", "severity": 2}], {"temp": 98.6})).json()
    assert r["nurse_alert"] is False
    assert client.get("/alerts/pending").json()["count"] == 0


@pytest.mark.xfail(reason="Known gap: the scorer matches exact names such as chest_pain; 'chest pain' scores ROUTINE.",
                   strict=False)
def test_unrecognised_symptom_name_should_not_score_routine(client):
    r = client.post("/check-in", json=_check_in([{"name": "chest pain", "severity": 9}])).json()
    assert r["urgency_level"] != "ROUTINE"


def test_acknowledge_alert(client):
    body = _check_in([{"name": "dyspnea", "severity": 8}])
    client.post("/check-in", json=body)
    aid = client.get("/alerts/pending").json()["alerts"][0]["alert_id"]
    assert client.post(f"/alerts/{aid}/acknowledge", params={"reviewed_by": "Nurse Kim"}).status_code == 200
    assert client.get("/alerts/pending").json()["count"] == 0
    assert client.post("/alerts/nope/acknowledge", params={"reviewed_by": "x"}).status_code == 404


def _conversation(client):
    sid = client.post("/conversation/start", json={"patient_id": "P001", "message": "My knee incision is red."}).json()["session_id"]
    client.post(f"/conversation/{sid}/message", json={"patient_id": "P001", "message": "It started yesterday."})
    return sid


def test_conversation_keeps_history(client):
    sid = _conversation(client)
    h = client.get(f"/conversation/{sid}").json()
    assert h["message_count"] == 4 and [m["role"] for m in h["history"]] == ["patient", "assistant", "patient", "assistant"]
    assert client.get("/conversation/nope").status_code == 404


def test_note_returns_503_when_model_unavailable(client):
    sid = _conversation(client)
    r = client.post(f"/conversation/{sid}/note")
    assert r.status_code == 503 and "NOTE_ADAPTER_PATH" in r.json()["detail"]


def test_note_draft_review_and_feedback_flow(client, monkeypatch):
    monkeypatch.setattr(api, "note_generator", FakeNotes())
    sid = _conversation(client)
    note = client.post(f"/conversation/{sid}/note").json()
    assert note["status"] == "draft" and note["model_version"] == "llama-2-aftercare-v4" and note["low_confidence"] is False
    assert client.get("/notes/pending").json()["count"] == 1
    r = client.post(f"/notes/{note['note_id']}/review",
                    json={"reviewed_by": "Nurse Kim", "final_text": "Knee incision redness for one day.", "approved": True})
    assert r.json()["status"] == "approved"
    assert client.post(f"/notes/{note['note_id']}/review",
                       json={"reviewed_by": "x", "final_text": "y"}).status_code == 409
    fb = client.get("/feedback/notes").json()
    assert fb["count"] == 1 and fb["records"][0]["edited"] is True and len(fb["records"][0]["dialogue_history"]) == 4


def test_guard_flags_mark_draft_low_confidence(client, monkeypatch):
    monkeypatch.setattr(api, "note_generator", FakeNotes("Lisinopril.", ["very_short"]))
    sid = _conversation(client)
    note = client.post(f"/conversation/{sid}/note").json()
    assert note["low_confidence"] is True and note["quality_flags"] == ["very_short"]


def test_unknown_note_is_404(client):
    assert client.get("/notes/nope").status_code == 404
    assert client.post("/notes/nope/review", json={"reviewed_by": "x", "final_text": "y"}).status_code == 404


def test_import_conversation_then_note(client):
    r = client.post("/conversation/import", json={"patient_id": "P9", "turns": [
        {"role": "doctor", "content": "How is the knee?"}, {"role": "patient", "content": "Better."}, {"role": "doctor", "content": " "}]})
    assert r.status_code == 200
    body = r.json()
    assert body["message_count"] == 2
    assert [t["role"] for t in body["conversation_history"]] == ["assistant", "patient"]
    assert client.post(f"/conversation/{body['session_id']}/note").status_code == 200
    assert client.post("/conversation/import", json={"patient_id": "P9", "turns": []}).status_code == 422
