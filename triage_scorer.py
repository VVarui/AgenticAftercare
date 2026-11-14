"""
AI Aftercare: Triage Scoring Module
Deterministic, rule-based urgency assessment (no ML required)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class UrgencyLevel(Enum):
    ROUTINE = 1           # Regular follow-up
    MONITOR = 2          # Watch closely, follow up soon
    URGENT = 3           # Requires nurse review today
    EMERGENCY = 4        # Call 911 / immediate intervention

@dataclass
class Symptom:
    """Structured symptom representation"""
    name: str                    # "cough", "chest_pain", "dyspnea"
    severity: int                # 1-10
    duration_hours: Optional[int] = None
    onset: Optional[str] = None  # "sudden", "gradual", "recurring"
    
class TriageScorer:
    """
    Deterministic triage scorer for patient check-ins.
    Uses clinical rules to assign urgency levels.
    
    Rules are transparent, explainable, and safe for production.
    """
    
    def __init__(self):
        # Red flags that immediately escalate to EMERGENCY (4)
        self.red_flags_emergency = {
            "chest_pain": {"severity_threshold": 8, "conditions": ["COPD", "CAD", "HTN"]},
            "difficulty_breathing": {"severity_threshold": 8},
            "loss_of_consciousness": {"severity_threshold": 1},  # Any severity
            "severe_bleeding": {"severity_threshold": 1},
            "severe_allergic_reaction": {"severity_threshold": 1},
        }
        
        # Urgent flags (escalate to URGENT=3)
        self.red_flags_urgent = {
            "fever": {"threshold": 103.5, "combos": ["cough", "dyspnea"]},  # pneumonia pattern
            "dyspnea": {"severity_threshold": 7},
            "altered_mental_status": {"severity_threshold": 5},
            "severe_headache": {"severity_threshold": 8},
            "abdominal_pain": {"severity_threshold": 7, "type": "acute"},
            "chest_pain": {"severity_threshold": 5, "conditions": ["COPD", "CAD"]},
        }
        
        # Monitor flags (MONITOR=2)
        self.yellow_flags = {
            "mild_fever": {"threshold": 101},  # 101-103.5
            "mild_cough": {"duration_hours": 48},
            "mild_headache": {"severity_threshold": 5},
            "nausea_vomiting": {"severity_threshold": 5, "duration_hours": 4},
        }
    
    def score_check_in(
        self,
        symptoms: List[Symptom],
        vitals: Dict[str, float],
        patient_history: Dict[str, any],
    ) -> Dict:
        """
        Main scoring function.
        
        Returns:
        {
            urgency_level: UrgencyLevel,
            score: 1-4,
            confidence: 0.0-1.0,
            reasoning: str,
            red_flags_detected: [str],
            nurse_alert: bool,
            recommended_action: str
        }
        """
        
        # Step 1: Check for immediate emergencies
        emergency_flags = self._check_emergency_flags(symptoms, vitals, patient_history)
        if emergency_flags:
            return {
                "urgency_level": UrgencyLevel.EMERGENCY,
                "score": 4,
                "confidence": 0.95,
                "reasoning": f"EMERGENCY: {'; '.join(emergency_flags)}",
                "red_flags_detected": emergency_flags,
                "nurse_alert": True,
                "recommended_action": "CALL 911 or refer to ED immediately",
            }
        
        # Step 2: Check for urgent flags
        urgent_flags, urgent_combos = self._check_urgent_flags(symptoms, vitals, patient_history)
        if urgent_flags or urgent_combos:
            reasoning = self._build_reasoning(urgent_flags, urgent_combos, patient_history)
            return {
                "urgency_level": UrgencyLevel.URGENT,
                "score": 3,
                "confidence": 0.85,
                "reasoning": reasoning,
                "red_flags_detected": urgent_flags + urgent_combos,
                "nurse_alert": True,
                "recommended_action": "Nurse should call patient today for assessment",
            }
        
        # Step 3: Check for monitor flags
        monitor_flags = self._check_monitor_flags(symptoms, vitals, patient_history)
        if monitor_flags:
            return {
                "urgency_level": UrgencyLevel.MONITOR,
                "score": 2,
                "confidence": 0.80,
                "reasoning": f"Monitor condition(s): {'; '.join(monitor_flags)}",
                "red_flags_detected": monitor_flags,
                "nurse_alert": False,
                "recommended_action": "Follow up in 24-48 hours, monitor symptoms",
            }
        
        # Step 4: Routine check-in
        return {
            "urgency_level": UrgencyLevel.ROUTINE,
            "score": 1,
            "confidence": 0.90,
            "reasoning": "Patient stable, symptoms within expected range",
            "red_flags_detected": [],
            "nurse_alert": False,
            "recommended_action": "Routine follow-up as scheduled",
        }
    
    def _check_emergency_flags(self, symptoms, vitals, history):
        """Check for life-threatening conditions"""
        flags = []
        
        for symptom in symptoms:
            if symptom.name in self.red_flags_emergency:
                rule = self.red_flags_emergency[symptom.name]
                
                # Any-severity flags
                if rule.get("severity_threshold") == 1:
                    flags.append(f"{symptom.name} (immediate threat)")
                # Severity-based flags
                elif symptom.severity >= rule.get("severity_threshold", 10):
                    flags.append(f"{symptom.name} (severity {symptom.severity})")
        
        # Vital sign emergencies
        if vitals.get("systolic") and vitals["systolic"] > 180:
            flags.append("Severe hypertension (SBP > 180)")
        if vitals.get("heart_rate") and vitals["heart_rate"] > 130:
            flags.append("Severe tachycardia (HR > 130)")
        if vitals.get("oxygen_sat") and vitals["oxygen_sat"] < 85:
            flags.append("Critical hypoxia (SpO2 < 85%)")
        
        return flags
    
    def _check_urgent_flags(self, symptoms, vitals, history):
        """Check for conditions requiring same-day nurse review"""
        flags = []
        combos = []
        
        for symptom in symptoms:
            if symptom.name in self.red_flags_urgent:
                rule = self.red_flags_urgent[symptom.name]
                
                if symptom.severity >= rule.get("severity_threshold", 10):
                    # Check comorbidities that increase risk
                    if "conditions" in rule:
                        patient_conditions = history.get("comorbidities", [])
                        if any(c in patient_conditions for c in rule["conditions"]):
                            flags.append(f"{symptom.name} + {rule['conditions']} history")
                        else:
                            flags.append(f"{symptom.name} (severity {symptom.severity})")
                    else:
                        flags.append(f"{symptom.name} (severity {symptom.severity})")
        
        # Symptom combinations (e.g., fever + cough + dyspnea = pneumonia pattern)
        symptom_names = {s.name for s in symptoms}
        
        # Pneumonia pattern
        if {"fever", "cough", "dyspnea"}.issubset(symptom_names):
            fever_symptom = next(s for s in symptoms if s.name == "fever")
            if vitals.get("temp", 98.6) > 101:
                combos.append("Pneumonia pattern (fever + cough + dyspnea)")
        
        # Sepsis pattern
        if fever_symptom := next((s for s in symptoms if s.name == "fever"), None):
            if vitals.get("temp", 98.6) > 103 and vitals.get("heart_rate", 60) > 100:
                if any(s in symptom_names for s in ["malaise", "confusion", "weakness"]):
                    combos.append("Possible sepsis (high fever + tachycardia + systemic symptoms)")
        
        return flags, combos
    
    def _check_monitor_flags(self, symptoms, vitals, history):
        """Check for conditions to monitor closely"""
        flags = []
        
        for symptom in symptoms:
            if symptom.name in self.yellow_flags:
                rule = self.yellow_flags[symptom.name]
                
                if symptom.severity >= rule.get("severity_threshold", 10):
                    flags.append(f"{symptom.name} (monitoring)")
        
        # Vital sign warnings
        if vitals.get("temp") and 100 < vitals["temp"] < 101:
            flags.append("Low-grade fever - monitor")
        
        return flags
    
    def _build_reasoning(self, flags, combos, history):
        """Build human-readable explanation for nurse"""
        reason = "Clinical reasoning: "
        
        if combos:
            reason += f"Symptom pattern suggests: {'; '.join(combos)}. "
        
        reason += f"Patient flags: {'; '.join(flags)}. "
        
        if history.get("comorbidities"):
            reason += f"Relevant history: {', '.join(history['comorbidities'])}. "
        
        reason += "Nurse review recommended to assess for escalation."
        
        return reason


# Example usage
if __name__ == "__main__":
    scorer = TriageScorer()
    
    # Example 1: Routine check-in
    symptoms_routine = [
        Symptom(name="mild_cough", severity=3, duration_hours=24),
    ]
    vitals_routine = {"temp": 98.6, "heart_rate": 72}
    history_routine = {"comorbidities": []}
    
    result1 = scorer.score_check_in(symptoms_routine, vitals_routine, history_routine)
    print("=== ROUTINE CHECK-IN ===")
    print(f"Urgency: {result1['urgency_level'].name}")
    print(f"Reasoning: {result1['reasoning']}\n")
    
    # Example 2: Urgent check-in
    symptoms_urgent = [
        Symptom(name="fever", severity=7),
        Symptom(name="cough", severity=6),
        Symptom(name="dyspnea", severity=7),
    ]
    vitals_urgent = {"temp": 102.5, "heart_rate": 105, "oxygen_sat": 92}
    history_urgent = {"comorbidities": ["COPD", "hypertension"]}
    
    result2 = scorer.score_check_in(symptoms_urgent, vitals_urgent, history_urgent)
    print("=== URGENT CHECK-IN ===")
    print(f"Urgency: {result2['urgency_level'].name}")
    print(f"Reasoning: {result2['reasoning']}\n")
    
    # Example 3: Emergency
    symptoms_emergency = [
        Symptom(name="chest_pain", severity=9),
        Symptom(name="difficulty_breathing", severity=9),
    ]
    vitals_emergency = {"temp": 98.6, "heart_rate": 130, "systolic": 185, "oxygen_sat": 83}
    history_emergency = {"comorbidities": ["CAD"]}
    
    result3 = scorer.score_check_in(symptoms_emergency, vitals_emergency, history_emergency)
    print("=== EMERGENCY CHECK-IN ===")
    print(f"Urgency: {result3['urgency_level'].name}")
    print(f"Reasoning: {result3['reasoning']}")
    print(f"Recommended action: {result3['recommended_action']}")
