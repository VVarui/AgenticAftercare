"""Unit tests for the rule-based triage scorer (no GPU, no network, no extra packages).

Run:  python -m pytest -q tests/test_triage.py
These tests pin down what the rules DO. They are not a clinical validation: the thresholds have not been
reviewed by a clinician.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from triage_scorer import Symptom, TriageScorer, UrgencyLevel

scorer = TriageScorer()


def score(symptoms, vitals=None, comorbidities=None):
    syms = [Symptom(name=n, severity=sev, duration_hours=dur) for n, sev, dur in
            [(s + (None,))[:3] if len(s) == 2 else s for s in symptoms]]
    return scorer.score_check_in(syms, vitals or {}, {"comorbidities": comorbidities or []})


def level(result):
    return result["urgency_level"]


# ---------------------------------------------------------------- result shape
def test_result_has_all_keys():
    r = score([("mild_headache", 2)])
    for key in ("urgency_level", "score", "confidence", "reasoning", "red_flags_detected", "nurse_alert",
                "recommended_action", "unrecognized_symptoms"):
        assert key in r
    assert isinstance(r["urgency_level"], UrgencyLevel)
    assert r["score"] == r["urgency_level"].value


def test_nobody_reported_is_routine():
    r = score([])
    assert level(r) == UrgencyLevel.ROUTINE and r["nurse_alert"] is False and r["red_flags_detected"] == []


# ---------------------------------------------------------------- emergency
@pytest.mark.parametrize("name", ["loss_of_consciousness", "severe_bleeding", "severe_allergic_reaction"])
def test_any_severity_emergency_symptoms(name):
    r = score([(name, 1)])
    assert level(r) == UrgencyLevel.EMERGENCY and r["nurse_alert"] is True
    assert "CALL 911" in r["recommended_action"]


@pytest.mark.parametrize("name", ["chest_pain", "difficulty_breathing"])
def test_severity_8_is_emergency(name):
    assert level(score([(name, 8)])) == UrgencyLevel.EMERGENCY
    assert level(score([(name, 10)])) == UrgencyLevel.EMERGENCY


@pytest.mark.parametrize("vitals", [{"systolic": 181}, {"heart_rate": 131}, {"oxygen_sat": 84}])
def test_vital_sign_emergencies(vitals):
    assert level(score([], vitals)) == UrgencyLevel.EMERGENCY


@pytest.mark.parametrize("vitals", [{"systolic": 180}, {"heart_rate": 130}, {"oxygen_sat": 85}])
def test_vital_boundaries_are_not_emergencies(vitals):
    assert level(score([], vitals)) != UrgencyLevel.EMERGENCY


def test_emergency_wins_over_everything_else():
    r = score([("mild_headache", 2), ("chest_pain", 9)], {"temp": 104})
    assert level(r) == UrgencyLevel.EMERGENCY


# ---------------------------------------------------------------- urgent
def test_chest_pain_severity_5_is_urgent_and_names_history():
    r = score([("chest_pain", 5)], comorbidities=["CAD"])
    assert level(r) == UrgencyLevel.URGENT and r["nurse_alert"] is True
    assert any("history" in f for f in r["red_flags_detected"])
    assert "CAD" in r["reasoning"]


def test_chest_pain_severity_4_is_not_urgent():
    assert level(score([("chest_pain", 4)])) != UrgencyLevel.URGENT


@pytest.mark.parametrize("name,sev", [("dyspnea", 7), ("altered_mental_status", 5), ("severe_headache", 8),
                                       ("abdominal_pain", 7), ("difficulty_breathing", 7)])
def test_urgent_thresholds(name, sev):
    assert level(score([(name, sev)])) == UrgencyLevel.URGENT
    assert level(score([(name, sev - 1)])) != UrgencyLevel.URGENT


def test_difficulty_breathing_severity_7_was_routine_before_the_fix():
    # Regression: the rule table only had difficulty_breathing at 8+, so severity 7 scored ROUTINE.
    assert level(score([("difficulty_breathing", 7)])) == UrgencyLevel.URGENT


def test_pneumonia_pattern_needs_temperature_above_101():
    syms = [("fever", 4), ("cough", 4), ("dyspnea", 4)]
    assert level(score(syms, {"temp": 102})) == UrgencyLevel.URGENT
    assert "Pneumonia pattern" in score(syms, {"temp": 102})["reasoning"]
    assert level(score(syms, {"temp": 100.5})) != UrgencyLevel.URGENT


def test_sepsis_pattern():
    syms = [("fever", 4), ("confusion", 4)]
    r = score(syms, {"temp": 103.2, "heart_rate": 110})
    assert level(r) == UrgencyLevel.URGENT and "sepsis" in r["reasoning"].lower()


def test_very_high_temperature_alone_is_urgent():
    # Regression: temp 104 with no symptoms listed used to score ROUTINE ("Patient stable").
    r = score([], {"temp": 104})
    assert level(r) == UrgencyLevel.URGENT and r["nurse_alert"] is True


def test_temperature_boundary_103_5_is_not_urgent_by_itself():
    assert level(score([], {"temp": 103.5})) == UrgencyLevel.MONITOR


# ---------------------------------------------------------------- monitor
def test_monitor_symptom_by_severity_does_not_alert():
    r = score([("mild_headache", 5)])
    assert level(r) == UrgencyLevel.MONITOR and r["nurse_alert"] is False


def test_mild_cough_monitored_after_48_hours():
    # Regression: the 48 h rule was declared but never applied.
    assert level(score([("mild_cough", 3, 72)])) == UrgencyLevel.MONITOR
    assert level(score([("mild_cough", 3, 24)])) == UrgencyLevel.ROUTINE
    assert level(score([("mild_cough", 3)])) == UrgencyLevel.ROUTINE      # duration unknown: stays routine


def test_temperature_bands():
    assert level(score([], {"temp": 99.9})) == UrgencyLevel.ROUTINE
    assert level(score([], {"temp": 100.5})) == UrgencyLevel.MONITOR      # low-grade
    assert level(score([], {"temp": 101})) == UrgencyLevel.MONITOR        # mild_fever threshold from the rule table
    assert level(score([], {"temp": 102.5})) == UrgencyLevel.MONITOR      # was ROUTINE before the fix
    assert level(score([], {"temp": 103.6})) == UrgencyLevel.URGENT


# ---------------------------------------------------------------- routine
def test_routine_example():
    r = score([("mild_cough", 3, 24)], {"temp": 98.6, "heart_rate": 72})
    assert level(r) == UrgencyLevel.ROUTINE and r["recommended_action"].startswith("Routine")


# ---------------------------------------------------------------- names
@pytest.mark.parametrize("raw", ["Chest Pain", "chest-pain", "  chest_pain ", "CHEST_PAIN", "chest  pain"])
def test_names_are_normalised(raw):
    r = score([(raw, 9)])
    assert level(r) == UrgencyLevel.EMERGENCY and r["unrecognized_symptoms"] == []


def test_unrecognised_symptom_is_surfaced_not_silently_routine():
    r = score([("toe_tingling", 9)])
    assert r["unrecognized_symptoms"] == ["toe_tingling"]
    assert level(r) == UrgencyLevel.MONITOR and r["nurse_alert"] is False
    assert "toe_tingling" in r["reasoning"] and "review" in r["reasoning"].lower()


def test_unrecognised_symptom_does_not_lower_a_higher_score():
    r = score([("chest_pain", 9), ("toe_tingling", 9)])
    assert level(r) == UrgencyLevel.EMERGENCY and r["unrecognized_symptoms"] == ["toe_tingling"]


def test_every_dashboard_symptom_is_recognised():
    # demo.py offers these names in its drop-down; none of them may be treated as unknown.
    import re
    src = open(os.path.join(os.path.dirname(__file__), "..", "demo.py"), encoding="utf-8").read()
    block = src.split("SYMPTOM_CHOICES", 1)[1].split("]", 1)[0]
    names = re.findall(r'"([a-z_]+)"', block)
    assert len(names) >= 10
    unknown = [n for n in names if n not in scorer.known_symptom_names()]
    assert unknown == []


# ---------------------------------------------------------------- known gaps (documented, not fixed)
@pytest.mark.xfail(reason="Known gap: SpO2 85-91 and HR 110-130 are not scored by themselves; needs clinician input "
                          "on thresholds before changing.", strict=True)
def test_low_but_not_critical_oxygen_is_flagged():
    assert level(score([], {"oxygen_sat": 88})) != UrgencyLevel.ROUTINE
