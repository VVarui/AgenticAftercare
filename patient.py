"""
AI Aftercare: patient check-in page (a separate Streamlit app from the nurse dashboard).

Run locally:   streamlit run patient.py --server.port 8502      (docker compose starts it on port 8502)

What the patient sees: plain-language questions, one clear safety message, an automated reply and a follow-up chat.
What the patient does NOT see: triage scores, red flags, the rules' reasoning, other patients, nurse alerts or notes.
Those stay on the nurse dashboard (demo.py).

Prototype limits: no patient login (anyone can type any Patient ID), no text-message or phone channel, no privacy
review. Use synthetic data only.
"""
import os
from typing import Dict, List, Tuple

from demo import api_call, build_check_in_payload  # same API helpers as the nurse dashboard

# Optional: shown in the safety messages when set (for example CLINIC_PHONE="555-0100").
CLINIC_PHONE = os.getenv("CLINIC_PHONE", "").strip()

# Plain-language label -> the symptom name the triage rules understand. Order is the order shown.
PATIENT_SYMPTOMS: List[Tuple[str, str]] = [
    ("Chest pain", "chest_pain"),
    ("Shortness of breath", "dyspnea"),
    ("Struggling to breathe", "difficulty_breathing"),
    ("Fainting or passing out", "loss_of_consciousness"),
    ("Heavy bleeding that will not stop", "severe_bleeding"),
    ("Allergic reaction (swelling, hives, trouble breathing)", "severe_allergic_reaction"),
    ("Confusion or unusual drowsiness", "altered_mental_status"),
    ("Severe headache", "severe_headache"),
    ("Belly pain", "abdominal_pain"),
    ("Fever", "fever"),
    ("Slight fever", "mild_fever"),
    ("Cough", "cough"),
    ("Mild cough", "mild_cough"),
    ("Mild headache", "mild_headache"),
    ("Nausea or vomiting", "nausea_vomiting"),
]

# Anything the rules cannot score is sent under this name. The API holds such check-ins at MONITOR and tells the
# nurse to read the patient's message, so "something else" is never reported as stable.
OTHER_LABEL = "Something else (describe it below)"
OTHER_NAME = "other_see_message"

CONDITION_CHOICES: Dict[str, str] = {
    "High blood pressure": "HTN",
    "Heart disease": "CAD",
    "COPD (lung disease)": "COPD",
    "Diabetes": "diabetes",
    "Asthma": "asthma",
}

EMERGENCY_BANNER = ("If you think this is an emergency, call 911 now. "
                    "This page is not monitored in real time and cannot send help.")


def _phone_hint() -> str:
    return f" Or call your clinic at {CLINIC_PHONE}." if CLINIC_PHONE else " Or call your clinic."


def guidance(urgency_level: str) -> Tuple[str, str]:
    """-> (box kind, message for the patient). Plain advice only: no score, no rule names, no medical reasoning."""
    level = (urgency_level or "").upper()
    if level == "EMERGENCY":
        return ("error", "Please call 911 or go to the nearest emergency room now. Do not wait for a call back. "
                         "Your care team has also been told.")
    if level == "URGENT":
        return ("warning", "A nurse should look at your check-in today and will contact you. "
                           "If you feel worse, or have trouble breathing, call 911." + _phone_hint())
    if level == "MONITOR":
        return ("info", "Thank you. Your care team has your check-in and will follow up within a day or two. "
                        "If you feel worse, send another check-in." + _phone_hint())
    return ("success", "Thank you. Your check-in was received. If anything changes, send another check-in.")


def build_patient_payload(patient_id: str, age: int, condition_labels: List[str], symptom_severity: Dict[str, int],
                          duration_hours: int, vitals: Dict[str, object], message: str) -> Dict:
    """Patient-form answers -> body for POST /check-in. `symptom_severity` is keyed by the plain-language label."""
    to_rule = dict(PATIENT_SYMPTOMS)
    to_rule[OTHER_LABEL] = OTHER_NAME
    symptoms = {to_rule[label]: sev for label, sev in symptom_severity.items() if label in to_rule}
    conditions = [CONDITION_CHOICES[c] for c in condition_labels if c in CONDITION_CHOICES]
    return build_check_in_payload(patient_id, age, conditions, [], symptoms, duration_hours or None, vitals, message)


def _box(st, kind: str, text: str):
    {"error": st.error, "warning": st.warning, "info": st.info, "success": st.success}[kind](text)


def section_check_in(st):
    st.subheader("How are you feeling today?")
    with st.form("patient_check_in"):
        c1, c2 = st.columns(2)
        patient_id = c1.text_input("Your patient ID (from your discharge papers)", "P001")
        age = c2.number_input("Your age", 0, 120, 62)
        conditions = st.multiselect("Do you have any of these? (optional)", list(CONDITION_CHOICES))
        chosen = st.multiselect("What symptoms are you having?", [label for label, _ in PATIENT_SYMPTOMS] + [OTHER_LABEL])
        sev_cols = st.columns(max(len(chosen), 1))
        severities = {label: sev_cols[i].slider(f"How bad is: {label}? (1 = very mild, 10 = the worst)", 1, 10, 3)
                      for i, label in enumerate(chosen)}
        duration = st.number_input("For how many hours have you had these symptoms?", 0, 500, 6)
        st.markdown("**If you measured any of these at home (optional, leave 0 if not):**")
        v1, v2, v3 = st.columns(3)
        vitals = {
            "temp": v1.number_input("Temperature (F)", 0.0, 110.0, 0.0, step=0.1),
            "heart_rate": v2.number_input("Pulse (beats per minute)", 0, 250, 0),
            "oxygen_sat": v3.number_input("Oxygen level (%)", 0, 100, 0),
        }
        message = st.text_area("Anything else you want your care team to know?", "")
        submitted = st.form_submit_button("Send my check-in")
    if not submitted:
        return
    if not patient_id.strip():
        st.warning("Please enter your patient ID.")
        return
    if not chosen:
        st.warning("Please choose at least one symptom. If none fit, choose 'Something else' and describe it.")
        return

    payload = build_patient_payload(patient_id, int(age), conditions, severities, int(duration), vitals, message)
    with st.spinner("Sending..."):
        ok, data, status = api_call("POST", "/check-in", json=payload)
    if not ok:
        st.error("We could not send your check-in. Please try again in a moment." + _phone_hint())
        return

    if st.session_state.get("patient_id") != patient_id.strip():   # a different patient: start with a clean chat
        for key in ("patient_session_id", "patient_history"):
            st.session_state.pop(key, None)
    st.session_state["patient_id"] = patient_id.strip()
    st.session_state["patient_reply"] = data.get("llm_response")
    kind, text = guidance(data.get("urgency_level"))
    st.session_state["patient_guidance"] = (kind, text)


def section_result(st):
    shown = st.session_state.get("patient_guidance")
    if not shown:
        return
    _box(st, *shown)
    reply = st.session_state.get("patient_reply")
    if reply:
        st.markdown("**Automated reply**")
        st.write(reply)
        st.caption("This reply is written by a computer program. It cannot diagnose you or give medical advice.")


def section_chat(st):
    pid = st.session_state.get("patient_id")
    if not pid:
        return
    st.subheader("Questions for your care team's assistant")
    st.caption("An automated assistant answers. It cannot diagnose or give medical advice, and a nurse may read "
               "this conversation.")
    history = st.session_state.get("patient_history", [])
    for turn in history:
        with st.chat_message("user" if turn["role"] == "patient" else "assistant"):
            st.write(turn["content"])
    text = st.chat_input("Type your message")
    if not text:
        return
    session_id = st.session_state.get("patient_session_id")
    if session_id:
        ok, data, status = api_call("POST", f"/conversation/{session_id}/message",
                                    json={"patient_id": pid, "message": text})
    else:
        ok, data, status = api_call("POST", "/conversation/start", json={"patient_id": pid, "message": text})
    if not ok:
        st.error("We could not send your message. Please try again." + _phone_hint())
        return
    st.session_state["patient_session_id"] = data["session_id"]
    st.session_state["patient_history"] = data["conversation_history"]
    st.rerun()


def main():
    import streamlit as st

    st.set_page_config(page_title="AI Aftercare: check-in", page_icon="🩺", layout="centered")
    st.title("AI Aftercare: patient check-in")
    st.error(EMERGENCY_BANNER)
    st.caption("Prototype for a course project. Use made-up information only.")
    section_check_in(st)
    section_result(st)
    section_chat(st)


if __name__ == "__main__":
    main()
