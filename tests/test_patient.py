"""Tests for the patient page (patient.py): helpers and page logic against a fake Streamlit. No network, no GPU.

Run:  python -m pytest -q tests/test_patient.py
"""
import contextlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import patient
from triage_scorer import TriageScorer


# ---------------------------------------------------------------- helpers
def test_every_patient_symptom_maps_to_a_rule_the_scorer_knows():
    known = TriageScorer().known_symptom_names()
    names = [rule for _, rule in patient.PATIENT_SYMPTOMS]
    assert len(names) == len(set(names)) >= 10
    assert [n for n in names if n not in known] == []


def test_every_triage_rule_symptom_is_offered_to_patients():
    scorer = TriageScorer()
    rules = set(scorer.red_flags_emergency) | set(scorer.red_flags_urgent) | set(scorer.yellow_flags)
    offered = {rule for _, rule in patient.PATIENT_SYMPTOMS}
    assert rules - offered == set()


def test_payload_maps_labels_and_conditions():
    body = patient.build_patient_payload(
        " P7 ", 70, ["Heart disease", "COPD (lung disease)", "nonsense"],
        {"Chest pain": 9, "Cough": 4, patient.OTHER_LABEL: 3}, 5,
        {"temp": 0.0, "heart_rate": 118, "oxygen_sat": 0}, "  it hurts ")
    assert body["patient"]["patient_id"] == "P7" and body["patient"]["comorbidities"] == ["CAD", "COPD"]
    assert {s["name"]: s["severity"] for s in body["symptoms"]} == {"chest_pain": 9, "cough": 4, patient.OTHER_NAME: 3}
    assert all(s["duration_hours"] == 5 for s in body["symptoms"])
    assert body["vitals"] == {"heart_rate": 118}          # zero means "not measured"
    assert body["message"] == "it hurts"


@pytest.mark.parametrize("level,kind", [("EMERGENCY", "error"), ("URGENT", "warning"),
                                         ("MONITOR", "info"), ("ROUTINE", "success"), (None, "success")])
def test_guidance_kind(level, kind):
    assert patient.guidance(level)[0] == kind


def test_guidance_never_leaks_scores_or_rule_details():
    for level in ("EMERGENCY", "URGENT", "MONITOR", "ROUTINE"):
        text = patient.guidance(level)[1].lower()
        for leak in ("score", "red flag", "confidence", "triage", "rule", "severity"):
            assert leak not in text, (level, leak)
    assert "911" in patient.guidance("EMERGENCY")[1] and "911" in patient.EMERGENCY_BANNER


def test_clinic_phone_is_used_when_set(monkeypatch):
    monkeypatch.setattr(patient, "CLINIC_PHONE", "555-0100")
    assert "555-0100" in patient.guidance("URGENT")[1]


# ---------------------------------------------------------------- page logic with a fake Streamlit
class Fake:
    submit = True

    def __init__(self, log, state=None, chat=None, fields=None):
        self.log, self.session_state = log, state if state is not None else {}
        self._chat, self.fields = chat, fields or {}

    def _ctx(self):
        return contextlib.nullcontext(self)

    def __getattr__(self, name):
        if name in ("error", "warning", "info", "success", "write", "caption", "markdown", "subheader", "title",
                    "set_page_config"):
            return lambda *a, **k: self.log.append((name, a[0] if a else ""))
        raise AttributeError(name)

    def columns(self, n, *a, **k):
        return [Fake(self.log, self.session_state, self._chat, self.fields) for _ in range(n if isinstance(n, int) else len(n))]

    def form(self, key):
        return contextlib.nullcontext()

    def spinner(self, *a):
        return contextlib.nullcontext()

    def chat_message(self, *a):
        return contextlib.nullcontext()

    def text_input(self, label, value="", **k):
        return self.fields.get("patient_id", value)

    def text_area(self, label, value="", **k):
        return self.fields.get("message", value)

    def number_input(self, label, mn=0, mx=0, value=0, **k):
        return value

    def multiselect(self, label, options, default=None, **k):
        return self.fields.get("symptoms", []) if "symptoms" in label else []

    def slider(self, label, mn, mx, value, **k):
        return self.fields.get("severity", value)

    def chat_input(self, *a, **k):
        return self._chat

    def form_submit_button(self, *a, **k):
        return Fake.submit

    def rerun(self):
        raise RuntimeError("rerun")


@pytest.fixture()
def api(monkeypatch):
    calls = []
    responses = {"/check-in": (True, {"urgency_level": "URGENT", "urgency_score": 3, "confidence": .85,
                                      "reasoning": "secret reasoning", "red_flags": ["chest_pain"],
                                      "nurse_alert": True, "recommended_action": "x", "llm_response": "Hello, a nurse will look."}, 200),
                 "/conversation/start": (True, {"session_id": "s1", "conversation_history": [
                     {"role": "patient", "content": "hi"}, {"role": "assistant", "content": "hello"}]}, 200)}

    def fake(method, path, timeout=0, **kw):
        calls.append((method, path, kw))
        if path.endswith("/message"):
            return True, {"session_id": "s1", "conversation_history": [{"role": "patient", "content": "again"}]}, 200
        return responses[path]
    monkeypatch.setattr(patient, "api_call", fake)
    fake.calls, fake.responses = calls, responses
    return fake


def run_check_in(fields, state=None):
    log = []
    st = Fake(log, state, fields=fields)
    patient.section_check_in(st)
    return st, log


def test_check_in_submit_stores_only_patient_safe_text(api):
    st, log = run_check_in({"symptoms": ["Chest pain"], "severity": 7})
    sent = [c for c in api.calls if c[1] == "/check-in"][0][2]["json"]
    assert sent["symptoms"][0]["name"] == "chest_pain"
    kind, text = st.session_state["patient_guidance"]
    assert kind == "warning" and "secret reasoning" not in text
    shown = Fake([], st.session_state)
    patient.section_result(shown)
    flat = " ".join(str(v) for _, v in shown.log)
    assert "Hello, a nurse will look." in flat and "secret reasoning" not in flat and "chest_pain" not in flat


def test_check_in_requires_patient_id_and_a_symptom(api):
    _, log = run_check_in({"patient_id": "  ", "symptoms": ["Cough"]})
    assert any("patient ID" in v for k, v in log if k == "warning")
    _, log = run_check_in({"symptoms": []})
    assert any("at least one symptom" in v for k, v in log if k == "warning")
    assert not [c for c in api.calls if c[1] == "/check-in"]


def test_check_in_failure_is_plain_language(api):
    api.responses["/check-in"] = (False, "Traceback blah 500", 500)
    st, log = run_check_in({"symptoms": ["Cough"]})
    errors = " ".join(v for k, v in log if k == "error")
    assert "could not send" in errors and "Traceback" not in errors
    assert "patient_guidance" not in st.session_state


def test_other_symptom_is_sent_as_unscored_name(api):
    run_check_in({"symptoms": [patient.OTHER_LABEL], "message": "my wound smells bad"})
    sent = [c for c in api.calls if c[1] == "/check-in"][0][2]["json"]
    assert sent["symptoms"][0]["name"] == patient.OTHER_NAME and sent["message"] == "my wound smells bad"


def test_changing_patient_id_clears_the_old_chat(api):
    state = {"patient_id": "P001", "patient_session_id": "old", "patient_history": [{"role": "patient", "content": "x"}]}
    st, _ = run_check_in({"patient_id": "P002", "symptoms": ["Cough"]}, state)
    assert st.session_state["patient_id"] == "P002" and "patient_session_id" not in st.session_state


def test_chat_starts_then_continues(api):
    state = {"patient_id": "P001"}
    with pytest.raises(RuntimeError):                       # st.rerun() after a successful send
        patient.section_chat(Fake([], state, chat="hi"))
    assert state["patient_session_id"] == "s1" and len(state["patient_history"]) == 2
    assert [c[1] for c in api.calls] == ["/conversation/start"]
    with pytest.raises(RuntimeError):
        patient.section_chat(Fake([], state, chat="more"))
    assert api.calls[-1][1] == "/conversation/s1/message"


def test_chat_hidden_until_a_check_in_exists(api):
    log = []
    patient.section_chat(Fake(log, {}, chat="hi"))
    assert log == [] and api.calls == []
