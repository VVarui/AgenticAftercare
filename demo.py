"""
AI Aftercare: Streamlit frontend (nurse dashboard).

Talks to the API only (API_URL, default http://localhost:8000). Tabs:
  1. Check-in & triage      -> POST /check-in
  2. Nurse alerts           -> GET /alerts/pending, POST /alerts/{id}/acknowledge
  3. Conversation & note    -> POST /conversation/start|message, POST /conversation/{id}/note
  4. Draft note review      -> GET /notes/pending, POST /notes/{id}/review   (nurse edits, approves or rejects)
  5. System                 -> GET /stats, GET /feedback/notes

Run locally:  streamlit run demo.py       (needs: pip install streamlit requests)
"""
import os
from typing import Dict, List, Optional, Tuple

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# Symptom names as triage_scorer.py knows them. Case, spaces and hyphens are tolerated ("Chest pain" works), but a name
# the rules do not know is not scored: it is reported back and the check-in is held at MONITOR for a nurse to read.
SYMPTOM_CHOICES = [
    "chest_pain", "dyspnea", "difficulty_breathing", "loss_of_consciousness", "severe_bleeding",
    "severe_allergic_reaction", "altered_mental_status", "severe_headache", "abdominal_pain",
    "fever", "cough", "mild_fever", "mild_cough", "mild_headache", "nausea_vomiting",
]

FLAG_TEXT = {
    "very_short": "The draft is very short, so it is probably incomplete.",
    "repetition": "The model repeated itself; repeated sentences were removed. Check the note carefully.",
    "truncated": "The draft may be cut off mid-sentence.",
    "continued_dialogue": "The model kept inventing dialogue after the note; the extra text was removed.",
}

URGENCY_ORDER = ["ROUTINE", "MONITOR", "URGENT", "EMERGENCY"]


# ----------------------------------------------------------------------------
# Helpers (no Streamlit here, so they can be unit-tested)
# ----------------------------------------------------------------------------

def api_call(method: str, path: str, timeout: float = 120.0, **kwargs) -> Tuple[bool, object, int]:
    """-> (ok, json_or_error_text, status_code). Never raises."""
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        return False, f"Cannot reach the API at {API_URL}: {exc}", 0
    try:
        body = resp.json()
    except ValueError:
        body = resp.text
    if resp.ok:
        return True, body, resp.status_code
    detail = body.get("detail") if isinstance(body, dict) else body
    return False, str(detail), resp.status_code


def build_check_in_payload(patient_id: str, age: Optional[int], comorbidities: List[str], meds: List[str],
                           symptoms: Dict[str, int], duration_hours: Optional[int], vitals: Dict[str, object],
                           message: str) -> Dict:
    """Request body for POST /check-in. Empty vitals are left out so the triage rules ignore them."""
    return {
        "patient": {
            "patient_id": patient_id.strip(),
            "age": age or None,
            "comorbidities": comorbidities,
            "current_meds": meds,
            "allergies": [],
        },
        "symptoms": [
            {"name": name, "severity": int(sev), "duration_hours": duration_hours}
            for name, sev in symptoms.items()
        ],
        "vitals": {k: v for k, v in vitals.items() if v not in (None, 0, 0.0)},
        "message": message.strip() or None,
    }


def split_list(text: str) -> List[str]:
    """'HTN, COPD' -> ['HTN', 'COPD']"""
    return [part.strip() for part in (text or "").split(",") if part.strip()]


def flag_messages(flags: List[str]) -> List[str]:
    return [FLAG_TEXT.get(flag, f"Output check raised: {flag}") for flag in flags or []]


def urgency_kind(level: str) -> str:
    """Which Streamlit box to use for an urgency level."""
    return {"EMERGENCY": "error", "URGENT": "warning", "MONITOR": "info"}.get((level or "").upper(), "success")


def is_demo_version(model_version: Optional[str]) -> bool:
    """True for drafts made by the API's demo mode (recorded notes, not generated from the conversation)."""
    return str(model_version or "").upper().startswith("DEMO")


def parse_transcript(text: str) -> List[Dict[str, str]]:
    """
    'Doctor: hi' / 'Patient: my knee hurts' lines (also '[doctor] hi', 'Nurse: ...') -> turns for /conversation/import.
    A line without a speaker label continues the previous turn.
    """
    import re
    turns: List[Dict[str, str]] = []
    speaker = r"(doctor|dr|physician|nurse|clinician|patient)"
    label = re.compile(rf"^\s*(?:\[{speaker}\]|{speaker}\s*:)\s*(.*)$", re.IGNORECASE)
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        m = label.match(line)
        if m:
            who = (m.group(1) or m.group(2)).lower()
            turns.append({"role": "patient" if who == "patient" else "doctor", "content": m.group(3).strip()})
        elif turns:
            turns[-1]["content"] += " " + line.strip()
    return [t for t in turns if t["content"]]


def transcript_text(history: List[Dict]) -> str:
    lines = []
    for turn in history or []:
        who = "Patient" if turn.get("role") == "patient" else "Assistant"
        lines.append(f"{who}: {turn.get('content', '')}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

def _box(st, kind: str, text: str):
    {"error": st.error, "warning": st.warning, "info": st.info, "success": st.success}[kind](text)


def tab_check_in(st, nurse: str):
    st.subheader("Patient check-in and triage")
    st.caption("The triage score comes from fixed, explainable rules, not from the language model.")
    with st.form("check_in_form"):
        c1, c2 = st.columns(2)
        patient_id = c1.text_input("Patient ID", "P001")
        age = c2.number_input("Age", 0, 120, 62)
        comorb = st.text_input("Conditions (comma separated)", "HTN")
        meds = st.text_input("Current medications (comma separated)", "")
        chosen = st.multiselect("Symptoms", SYMPTOM_CHOICES, default=["mild_headache"])
        sev_cols = st.columns(max(len(chosen), 1))
        severities = {name: sev_cols[i].slider(f"{name} (1-10)", 1, 10, 3) for i, name in enumerate(chosen)}
        duration = st.number_input("Duration in hours (applies to all symptoms)", 0, 500, 6)
        v1, v2, v3, v4 = st.columns(4)
        vitals = {
            "temp": v1.number_input("Temp (F)", 0.0, 110.0, 0.0, step=0.1),
            "heart_rate": v2.number_input("Heart rate", 0, 250, 0),
            "oxygen_sat": v3.number_input("SpO2 (%)", 0, 100, 0),
            "systolic": v4.number_input("Systolic BP", 0, 300, 0),
        }
        st.caption("Leave a vital at 0 to leave it out.")
        message = st.text_area("Patient message", "I have had a headache since this morning.")
        submitted = st.form_submit_button("Submit check-in")
    if not submitted:
        return
    if not chosen:
        st.warning("Choose at least one symptom.")
        return
    payload = build_check_in_payload(patient_id, int(age), split_list(comorb), split_list(meds), severities,
                                     int(duration) or None, vitals, message)
    with st.spinner("Scoring..."):
        ok, data, status = api_call("POST", "/check-in", json=payload)
    if not ok:
        st.error(f"Check-in failed ({status}): {data}")
        return
    _box(st, urgency_kind(data["urgency_level"]),
         f"**{data['urgency_level']}** (score {data['urgency_score']}, confidence {data['confidence']:.2f})")
    st.write("**Recommended action:**", data["recommended_action"])
    st.write("**Why:**", data["reasoning"])
    if data.get("red_flags"):
        st.write("**Red flags:**", ", ".join(data["red_flags"]))
    if data["nurse_alert"]:
        st.warning("A nurse alert was created. See the Nurse alerts tab.")
    if data.get("llm_response"):
        st.write("**Reply to the patient:**", data["llm_response"])


def tab_alerts(st, nurse: str):
    st.subheader("Pending nurse alerts")
    if st.button("Refresh", key="refresh_alerts"):
        st.rerun()
    ok, data, status = api_call("GET", "/alerts/pending")
    if not ok:
        st.error(f"Could not load alerts ({status}): {data}")
        return
    alerts = data["alerts"]
    if not alerts:
        st.success("No pending alerts.")
        return
    for alert in alerts:
        with st.container(border=True):
            _box(st, urgency_kind(alert["urgency"]),
                 f"**{alert['urgency']}**: patient {alert['patient_id']} at {alert['timestamp'][:19].replace('T', ' ')}")
            st.write("Symptoms:", ", ".join(alert["symptoms"]))
            st.write("Why:", alert["reasoning"])
            st.write("Recommended:", alert["recommended_action"])
            if not nurse.strip():
                st.info("Enter your name in the sidebar to acknowledge alerts.")
            if st.button("Acknowledge", key=f"ack_{alert['alert_id']}", disabled=not nurse.strip()):
                ok, res, status = api_call("POST", f"/alerts/{alert['alert_id']}/acknowledge",
                                           params={"reviewed_by": nurse.strip()})
                if ok:
                    st.rerun()
                else:
                    st.error(f"Could not acknowledge ({status}): {res}")


def tab_conversation(st, nurse: str):
    st.subheader("Conversation and draft note")
    st.caption("Chat replies come from the Ollama model. The clinical note comes from the fine-tuned v4 model "
               "and needs an NVIDIA GPU; without one the note button explains why it is unavailable. "
               "For the best note, load a full doctor-patient conversation instead of chatting.")
    state = st.session_state
    if "session_id" not in state:
        with st.form("start_conv"):
            pid = st.text_input("Patient ID", "P001")
            first = st.text_area("Patient's first message", "I had knee surgery last week and the incision looks red.")
            if st.form_submit_button("Start conversation"):
                ok, data, status = api_call("POST", "/conversation/start", json={"patient_id": pid, "message": first})
                if ok:
                    state["session_id"], state["patient_id"], state["history"] = (
                        data["session_id"], pid, data["conversation_history"])
                    st.rerun()
                else:
                    st.error(f"Could not start ({status}): {data}")
        with st.form("import_conv"):
            st.markdown("**Or load a doctor-patient conversation** (the note model was trained on these)")
            pid2 = st.text_input("Patient ID", "P001", key="import_pid")
            pasted = st.text_area("Paste the conversation, one line per turn, starting each with Doctor: or Patient:",
                                  height=220, key="import_text")
            if st.form_submit_button("Load conversation"):
                turns = parse_transcript(pasted)
                if not turns:
                    st.warning("Nothing to load. Start each line with 'Doctor:' or 'Patient:'.")
                else:
                    ok, data, status = api_call("POST", "/conversation/import", json={"patient_id": pid2, "turns": turns})
                    if ok:
                        state["session_id"], state["patient_id"], state["history"] = (
                            data["session_id"], pid2, data["conversation_history"])
                        st.rerun()
                    else:
                        st.error(f"Could not load ({status}): {data}")
        return

    for turn in state["history"]:
        with st.chat_message("user" if turn["role"] == "patient" else "assistant"):
            st.write(turn["content"])
    reply = st.chat_input("Patient message")
    if reply:
        ok, data, status = api_call("POST", f"/conversation/{state['session_id']}/message",
                                    json={"patient_id": state["patient_id"], "message": reply})
        if ok:
            state["history"] = data["conversation_history"]
            st.rerun()
        else:
            st.error(f"Message failed ({status}): {data}")

    c1, c2 = st.columns(2)
    if c1.button("Generate draft note"):
        with st.spinner("Generating (the first call loads the model and can take a few minutes)..."):
            ok, data, status = api_call("POST", f"/conversation/{state['session_id']}/note", timeout=900)
        if ok:
            st.success("Draft created. Review it in the Draft note review tab.")
            if is_demo_version(data.get("model_version")):
                st.info("DEMO mode: this is a recorded v4 note, not generated from the conversation above.")
            st.text_area("Draft", data["draft_text"], height=200, disabled=True)
            for msg in flag_messages(data.get("quality_flags")):
                st.warning(msg)
        elif status == 503:
            st.info(f"Note generation is not available here: {data}")
        else:
            st.error(f"Note generation failed ({status}): {data}")
    if c2.button("New conversation"):
        for key in ("session_id", "patient_id", "history"):
            state.pop(key, None)
        st.rerun()


def tab_review(st, nurse: str):
    st.subheader("Draft notes awaiting review")
    st.warning("AI-generated drafts are not clinically validated. The model can invent details or assume "
               "the patient's sex. Compare every statement with the conversation before approving.")
    if st.button("Refresh", key="refresh_notes"):
        st.rerun()
    ok, data, status = api_call("GET", "/notes/pending")
    if not ok:
        st.error(f"Could not load drafts ({status}): {data}")
        return
    notes = data["notes"]
    if not notes:
        st.success("No drafts waiting.")
        return
    for note in notes:
        nid = note["note_id"]
        title = f"Patient {note['patient_id']} · {note['created_at'][:19].replace('T', ' ')}"
        if note.get("low_confidence"):
            title += " · CHECK CLOSELY"
        with st.expander(title, expanded=True):
            if is_demo_version(note.get("model_version")):
                st.info("DEMO draft: a note the v4 model wrote earlier on a GPU. It is NOT generated from this "
                        "conversation, so it will not match what was said. Use it to try the review flow.")
            for msg in flag_messages(note.get("quality_flags")):
                st.warning(msg)
            left, right = st.columns(2)
            with left:
                st.markdown("**Conversation**")
                okc, conv, _ = api_call("GET", f"/conversation/{note['session_id']}")
                st.text(transcript_text(conv["history"]) if okc else "(conversation not available)")
            with right:
                st.markdown(f"**Draft** ({note.get('model_version') or 'model'})")
                final = st.text_area("Edit the note", note["draft_text"], height=260, key=f"edit_{nid}")
            b1, b2, _ = st.columns([1, 1, 3])
            if not nurse.strip():
                st.info("Enter your name in the sidebar to approve or reject.")
            if b1.button("Approve", key=f"approve_{nid}", disabled=not nurse.strip(), type="primary"):
                _review(st, nid, nurse, final, True)
            if b2.button("Reject", key=f"reject_{nid}", disabled=not nurse.strip()):
                _review(st, nid, nurse, final, False)


def _review(st, note_id: str, nurse: str, final_text: str, approved: bool):
    ok, data, status = api_call("POST", f"/notes/{note_id}/review",
                                json={"reviewed_by": nurse.strip(), "final_text": final_text, "approved": approved})
    if ok:
        st.rerun()
    else:
        st.error(f"Review failed ({status}): {data}")


def tab_system(st, nurse: str):
    st.subheader("System")
    ok, health, _ = api_call("GET", "/health")
    if ok:
        st.write("**Database:**", health.get("database", "unknown"))
        st.write("**Note model:**", health.get("note_generation", "unknown"))
    ok, stats, _ = api_call("GET", "/stats")
    if ok:
        cols = st.columns(4)
        cols[0].metric("Check-ins", stats["total_check_ins"])
        cols[1].metric("Pending alerts", stats["pending_alerts"])
        cols[2].metric("Drafts pending", stats.get("draft_notes_pending", 0))
        cols[3].metric("Feedback records", stats.get("feedback_records", 0))
    ok, fb, _ = api_call("GET", "/feedback/notes", params={"limit": 20})
    if ok and fb.get("records"):
        st.markdown("**Recent nurse feedback (draft vs final)**")
        st.dataframe([{k: r.get(k) for k in ("note_id", "approved", "edited")} for r in fb["records"]])


def main():
    import streamlit as st

    st.set_page_config(page_title="AI Aftercare", page_icon="🩺", layout="wide")
    st.title("AI Aftercare: nurse dashboard")
    with st.sidebar:
        nurse = st.text_input("Your name (nurse)", key="nurse_name")
        ok, health, _ = api_call("GET", "/health", timeout=5)
        if ok:
            st.success("API connected")
            st.caption(f"Database: {health.get('database', '?')}")
            st.caption(f"Note model: {health.get('note_generation', '?')}")
        else:
            st.error(str(health))
        st.caption("Prototype: not for real patient data.")
    tabs = st.tabs(["Check-in", "Nurse alerts", "Conversation & note", "Draft note review", "System"])
    for tab, fn in zip(tabs, (tab_check_in, tab_alerts, tab_conversation, tab_review, tab_system)):
        with tab:
            fn(st, nurse)


if __name__ == "__main__":
    main()
