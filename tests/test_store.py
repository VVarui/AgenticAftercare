"""Store tests: no GPU, no Docker. Run:  pytest tests/test_store.py"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import Store  # noqa: E402


@pytest.fixture()
def store():
    s = Store("")  # in-memory SQLite
    s.init_schema()
    return s


def _check_in(cid="c1", pid="P001", level="URGENT", alert=True):
    record = {
        "check_in_id": cid, "patient_id": pid, "timestamp": "2026-11-11T10:00:00", "symptoms": ["chest pain"],
        "vitals": {"heart_rate": 118}, "message": "hurts",
        "triage_result": {"urgency_level": level, "score": 3, "confidence": 0.85, "reasoning": "r",
                          "red_flags_detected": ["x"], "nurse_alert": alert, "recommended_action": "call"},
    }
    patient = {"patient_id": pid, "age": 68, "comorbidities": ["HF"], "current_meds": [], "allergies": []}
    a = ({"alert_id": "a-" + cid, "check_in_id": cid, "patient_id": pid, "timestamp": record["timestamp"],
          "urgency": level, "symptoms": ["chest pain"], "reasoning": "r", "recommended_action": "call"}
         if alert else None)
    return record, patient, a


def test_seed_patients_and_idempotent_init(store):
    store.init_schema()  # second call must not fail or duplicate
    with store._tx() as cur:
        n = store._row(cur, "SELECT COUNT(*) AS n FROM patients")["n"]
    assert n == 2


def test_check_in_roundtrip_and_llm_response(store):
    rec, pat, alert = _check_in()
    store.save_check_in(rec, pat, alert)
    store.set_llm_response("c1", "We are here to help.")
    got = store.get_check_in("c1")
    assert got["triage_result"]["urgency_level"] == "URGENT"
    assert got["triage_result"]["nurse_alert"] is True
    assert got["symptoms"] == ["chest pain"] and got["vitals"] == {"heart_rate": 118}
    assert got["llm_response"] == "We are here to help."
    assert store.get_check_in("nope") is None


def test_alert_lifecycle(store):
    rec, pat, alert = _check_in()
    store.save_check_in(rec, pat, alert)
    rec2, pat2, _ = _check_in("c2", level="ROUTINE", alert=False)
    store.save_check_in(rec2, pat2, None)
    assert [a["alert_id"] for a in store.pending_alerts()] == ["a-c1"]
    assert store.acknowledge_alert("a-c1", "Nurse Kim") is True
    assert store.pending_alerts() == []
    assert store.get_alert("a-c1")["reviewed_by"] == "Nurse Kim"
    assert store.acknowledge_alert("missing", "x") is False


def test_history_and_stats(store):
    for i, level in enumerate(["URGENT", "ROUTINE", "EMERGENCY"]):
        rec, pat, alert = _check_in(f"c{i}", level=level, alert=level != "ROUTINE")
        rec["timestamp"] = f"2026-11-11T10:0{i}:00"
        store.save_check_in(rec, pat, alert)
    total, rows = store.patient_history("P001", limit=2)
    assert total == 3 and [r["check_in_id"] for r in rows] == ["c2", "c1"]  # newest first
    s = store.stats()
    assert (s["total_check_ins"], s["total_alerts"], s["urgent_cases"], s["pending_alerts"]) == (3, 2, 2, 2)


def test_conversation_order_and_restart(store):
    store.create_session("s1", "P001", {"age": 60}, "2026-11-11T10:00:00")
    store.add_message("s1", "patient", "hello", "t1")
    store.add_message("s1", "assistant", "hi", "t2")
    store.add_message("s1", "patient", "knee hurts", "t3")
    s = store.get_session("s1")
    assert [m["content"] for m in s["history"]] == ["hello", "hi", "knee hurts"]
    assert s["context"] == {"age": 60}
    store.create_session("s1", "P001", {}, "2026-11-11T11:00:00")  # restart clears messages
    assert store.get_session("s1")["history"] == []
    assert store.get_session("missing") is None


def _draft(store, note_id="n1"):
    store.create_session("s1", "P001", {}, "t0")
    store.add_message("s1", "patient", "my knee hurts", "t1")
    store.save_draft({"note_id": note_id, "session_id": "s1", "patient_id": "P001", "created_at": "t2",
                      "draft_text": "Knee pain.", "quality_flags": ["very_short"], "low_confidence": True,
                      "model_version": "llama-2-aftercare-v4", "disclaimer": "AI draft"})


def test_draft_review_and_feedback_log(store):
    _draft(store)
    d = store.get_draft("n1")
    assert d["status"] == "draft" and d["low_confidence"] is True and d["quality_flags"] == ["very_short"]
    assert [n["note_id"] for n in store.pending_drafts()] == ["n1"]
    status, note = store.review_draft("n1", "Nurse Kim", "Knee pain for two days.", True)
    assert status == "ok" and note["status"] == "approved" and note["final_text"] == "Knee pain for two days."
    assert store.pending_drafts() == []
    fb = store.feedback_records()
    assert len(fb) == 1 and fb[0]["edited"] is True and fb[0]["approved"] is True
    assert fb[0]["model_draft"] == "Knee pain." and fb[0]["dialogue_history"][0]["content"] == "my knee hurts"
    assert fb[0]["model_version"] == "llama-2-aftercare-v4"


def test_review_conflict_missing_and_reject(store):
    _draft(store)
    assert store.review_draft("zzz", "N", "t", True) == ("missing", None)
    assert store.review_draft("n1", "N", "Knee pain.", False)[0] == "ok"
    assert store.review_draft("n1", "N", "again", True) == ("conflict", "rejected")
    fb = store.feedback_records()
    assert len(fb) == 1 and fb[0]["approved"] is False and fb[0]["edited"] is False


def test_audit_trail(store):
    rec, pat, alert = _check_in()
    store.save_check_in(rec, pat, alert)
    store.acknowledge_alert("a-c1", "Nurse Kim")
    actions = {e["action"] for e in store.audit_events()}
    assert {"check_in_created", "alert_created", "alert_acknowledged"} <= actions


def test_data_survives_reopen(tmp_path):
    url = f"sqlite:///{tmp_path / 'aftercare.db'}"
    a = Store(url)
    a.init_schema()
    rec, pat, alert = _check_in()
    a.save_check_in(rec, pat, alert)
    b = Store(url)  # new object = what an API restart does
    b.init_schema()
    assert b.persistent is True
    assert b.get_check_in("c1") is not None and len(b.pending_alerts()) == 1
