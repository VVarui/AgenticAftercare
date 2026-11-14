import sys, types, contextlib
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import demo

# ---------- pure helpers ----------
p = demo.build_check_in_payload(" P1 ", 0, ["HTN"], [], {"chest_pain": 9}, 2, {"temp": 0.0, "heart_rate": 118, "oxygen_sat": 0}, " ")
assert p["patient"]["patient_id"] == "P1" and p["patient"]["age"] is None
assert p["vitals"] == {"heart_rate": 118} and p["message"] is None
assert p["symptoms"] == [{"name": "chest_pain", "severity": 9, "duration_hours": 2}]
assert demo.split_list("HTN, COPD ,, ") == ["HTN", "COPD"]
assert demo.urgency_kind("EMERGENCY") == "error" and demo.urgency_kind("routine") == "success"
assert len(demo.flag_messages(["very_short", "weird"])) == 2 and "weird" in demo.flag_messages(["weird"])[0]
assert demo.transcript_text([{"role": "patient", "content": "a"}, {"role": "assistant", "content": "b"}]) == "Patient: a\nAssistant: b"
for name in demo.SYMPTOM_CHOICES: assert name == name.lower() and " " not in name
assert demo.is_demo_version("DEMO: recorded v4 output") and not demo.is_demo_version("llama-2-aftercare-v4") and not demo.is_demo_version(None)
tt = demo.parse_transcript("Doctor: How are you?\nPatient: Not great,\nmy knee hurts.\n[doctor] Any fever?\n\nPATIENT: No.\nrandom text")
assert tt == [{"role": "doctor", "content": "How are you?"}, {"role": "patient", "content": "Not great, my knee hurts."},
              {"role": "doctor", "content": "Any fever?"}, {"role": "patient", "content": "No. random text"}], tt
assert demo.parse_transcript("hello there") == [] and demo.parse_transcript("Doctor:") == []
print("helpers OK")

# ---------- fake streamlit to run every tab's code path ----------
class Fake:
    def __init__(self, log): self.log = log; self.session_state = {}; self.sidebar = self._ctx()
    def _ctx(self): return contextlib.nullcontext(self)
    def __getattr__(self, name):
        if name in ("error","warning","info","success","write","caption","markdown","text","subheader","title","divider","dataframe","metric","set_page_config"):
            return lambda *a, **k: self.log.append((name, a[0] if a else ""))
        raise AttributeError(name)
    def columns(self, n, *a, **k): n = n if isinstance(n, int) else len(n); return [self._ctx() and Fake(self.log) for _ in range(n)]
    def tabs(self, labels): return [contextlib.nullcontext() for _ in labels]
    def form(self, key): return contextlib.nullcontext()
    def container(self, **k): return contextlib.nullcontext()
    def expander(self, *a, **k): return contextlib.nullcontext()
    def spinner(self, *a): return contextlib.nullcontext()
    def chat_message(self, *a): return contextlib.nullcontext()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    # widgets return defaults
    def text_input(self, label, value="", **k): return value
    def text_area(self, label, value="", **k): return value
    def number_input(self, label, mn=0, mx=0, value=0, **k): return value
    def multiselect(self, label, opts, default=None, **k): return default or []
    def slider(self, label, mn, mx, value, **k): return value
    def chat_input(self, *a, **k): return None
    def rerun(self): raise RuntimeError("rerun")
    submit = False
    only = None
    def form_submit_button(self, label='', *a, **k): return Fake.submit if Fake.only is None else label == Fake.only
    pressed = set()
    def button(self, label, key=None, **k): return (not k.get('disabled')) and (key in Fake.pressed or label in Fake.pressed)

NOTE = {"note_id": "n1", "session_id": "s1", "patient_id": "P001", "created_at": "2026-11-13T13:00:00", "draft_text": "Lisinopril 10 mg daily.",
        "quality_flags": ["very_short"], "low_confidence": True, "model_version": "llama-2-aftercare-v4"}
calls = []
def fake_api(method, path, timeout=0, **kw):
    calls.append((method, path, kw))
    if path == "/health": return True, {"database": "postgres", "note_generation": "disabled: no GPU"}, 200
    if path == "/check-in": return True, {"urgency_level": "EMERGENCY", "urgency_score": 4, "confidence": .95, "reasoning": "r", "red_flags": ["chest_pain"], "nurse_alert": True, "recommended_action": "CALL 911", "llm_response": "hi"}, 200
    if path == "/alerts/pending": return True, {"count": 1, "alerts": [{"alert_id": "a1", "patient_id": "P001", "urgency": "EMERGENCY", "timestamp": "2026-11-13T13:00:00", "symptoms": ["chest_pain"], "reasoning": "r", "recommended_action": "x"}]}, 200
    if path.startswith("/alerts/a1/acknowledge"): return True, {"status": "acknowledged"}, 200
    if path == "/notes/pending": return True, {"count": 1, "notes": [NOTE]}, 200
    if path == "/conversation/s1": return True, {"history": [{"role": "patient", "content": "hi"}]}, 200
    if path == "/conversation/import": return True, {"session_id": "s1", "conversation_history": [{"role": "assistant", "content": "How are you?"}, {"role": "patient", "content": "hi"}]}, 200
    if path == "/conversation/start": return True, {"session_id": "s1", "conversation_history": [{"role": "patient", "content": "hi"}]}, 200
    if path.endswith("/note"): return False, "NOTE_ADAPTER_PATH is not set", 503
    if path == "/notes/n1/review": return True, {"status": "approved"}, 200
    if path == "/stats": return True, {"total_check_ins": 1, "pending_alerts": 1, "draft_notes_pending": 1, "feedback_records": 0}, 200
    if path == "/feedback/notes": return True, {"count": 1, "records": [{"note_id": "n1", "approved": True, "edited": True}]}, 200
    raise AssertionError(path)
demo.api_call = fake_api

log = []; st = Fake(log)
Fake.submit = True; demo.tab_check_in(st, "Nurse A"); Fake.submit = False
assert any(k == "error" and "EMERGENCY" in v for k, v in log), "urgent box missing"
assert ("POST", "/check-in") in [(m, p) for m, p, _ in calls]
demo.tab_alerts(st, "Nurse A")
Fake.pressed = {"ack_a1"}
try: demo.tab_alerts(st, "Nurse A")
except RuntimeError: pass
assert any(p.startswith("/alerts/a1/acknowledge") and kw["params"] == {"reviewed_by": "Nurse A"} for _, p, kw in calls)
Fake.pressed = set()
demo.tab_review(st, "Nurse A")
assert any(k == "warning" and "very short" in v for k, v in log), "flag warning not shown"
Fake.pressed = {"approve_n1"}
try: demo.tab_review(st, "Nurse A")
except RuntimeError: pass
rv = [kw for _, p, kw in calls if p == "/notes/n1/review"][-1]["json"]
assert rv == {"reviewed_by": "Nurse A", "final_text": "Lisinopril 10 mg daily.", "approved": True}, rv
Fake.pressed = {"reject_n1"}
try: demo.tab_review(st, "Nurse A")
except RuntimeError: pass
assert [kw for _, p, kw in calls if p == "/notes/n1/review"][-1]["json"]["approved"] is False
n_before = len([c for c in calls if c[1] == "/notes/n1/review"])
Fake.pressed = {"approve_n1"}; demo.tab_review(st, "")
assert len([c for c in calls if c[1] == "/notes/n1/review"]) == n_before, "review sent without a nurse name"
Fake.pressed = set()
Fake.submit = True
try: demo.tab_conversation(st, "Nurse A")
except RuntimeError: pass
Fake.submit = False
assert "session_id" in st.session_state
st.session_state.clear()
Fake.submit = True; Fake.only = "Load conversation"
_orig_ta = Fake.text_area
Fake.text_area = lambda self, label, value="", **k: "Doctor: How are you?\nPatient: hi" if k.get("key") == "import_text" else value
try: demo.tab_conversation(st, "Nurse A")
except RuntimeError: pass
Fake.text_area = _orig_ta; Fake.submit = False; Fake.only = None
imp = [kw for _, p, kw in calls if p == "/conversation/import"][-1]["json"]
assert imp["turns"] == [{"role": "doctor", "content": "How are you?"}, {"role": "patient", "content": "hi"}], imp
assert "session_id" in st.session_state
Fake.pressed = {"Generate draft note"}; demo.tab_conversation(st, "Nurse A")
assert any(k == "info" and "not available" in v for k, v in log), "503 message missing"
Fake.pressed = set()
demo.tab_system(st, "Nurse A")
print("tab code paths OK;", len(calls), "API calls simulated")
