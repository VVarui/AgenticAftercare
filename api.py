"""
AI Aftercare: FastAPI REST API
Real-time patient check-in endpoint with triage scoring
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import uuid
from datetime import datetime
from triage_scorer import TriageScorer, Symptom, UrgencyLevel
import asyncio
import httpx
import os
from note_generator_aftercare import note_generator
from db import store

app = FastAPI(
    title="AI Aftercare API",
    description="Real-time conversational nurse assistant with triage",
    version="0.1.0",
)

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistence: PostgreSQL when POSTGRES_URL is set (docker compose), see db.py.
# Without POSTGRES_URL it falls back to an in-memory SQLite database (local dev only, not persistent).

scorer = TriageScorer()

# Ollama configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = "llama2"

# ============================================================================
# Pydantic Models
# ============================================================================

class SymptomInput(BaseModel):
    name: str
    severity: int  # 1-10
    duration_hours: Optional[int] = None
    onset: Optional[str] = None

class VitalsInput(BaseModel):
    temp: Optional[float] = None
    heart_rate: Optional[int] = None
    oxygen_sat: Optional[float] = None  # 0-100%
    systolic: Optional[int] = None
    diastolic: Optional[int] = None
    respiratory_rate: Optional[int] = None

class PatientHistoryInput(BaseModel):
    patient_id: str
    age: Optional[int] = None
    comorbidities: List[str] = []
    current_meds: List[str] = []
    allergies: List[str] = []

class CheckInRequest(BaseModel):
    patient: PatientHistoryInput
    symptoms: List[SymptomInput]
    vitals: VitalsInput
    message: Optional[str] = None  # Patient's free-text message

class ConversationMessage(BaseModel):
    role: str  # "patient" or "assistant"
    content: str
    timestamp: Optional[str] = None

class LiveConversationRequest(BaseModel):
    patient_id: str
    session_id: Optional[str] = None  # For multi-turn conversations
    message: str
    context: Optional[Dict] = None  # Patient context if available

class ImportTurn(BaseModel):
    role: str       # "patient", or "doctor" / "nurse" / "assistant" for the clinician side
    content: str

class ConversationImportRequest(BaseModel):
    patient_id: str
    turns: List[ImportTurn]

class LiveConversationResponse(BaseModel):
    session_id: str
    message: str
    role: str = "assistant"
    timestamp: str
    conversation_history: List[ConversationMessage]

class CheckInResponse(BaseModel):
    check_in_id: str
    timestamp: str
    urgency_level: str
    urgency_score: int
    confidence: float
    reasoning: str
    red_flags: List[str]
    nurse_alert: bool
    recommended_action: str
    llm_response: Optional[str] = None  # Will be filled in by LLM layer

class NurseAlertResponse(BaseModel):
    alert_id: str
    check_in_id: str
    patient_id: str
    urgency_level: str
    symptoms_summary: str
    reasoning: str
    timestamp: str
    status: str = "pending"  # pending, reviewed, resolved

class NoteReviewRequest(BaseModel):
    reviewed_by: str
    final_text: str          # the note after the nurse edited it
    approved: bool = True

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check for deployment monitoring"""
    db_ok = store.ping()
    return {
        "status": "healthy" if db_ok else "degraded",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Aftercare",
        "database": store.describe() if db_ok else "unreachable",
        "note_generation": ("DEMO mode (recorded v4 notes, not generated live)" if note_generator.demo_mode
                            else "enabled" if note_generator.enabled
                            else f"disabled: {note_generator.disabled_reason()}"),
    }

async def generate_llm_response(symptoms_text: str, triage_level: str, patient_history: Dict) -> str:
    """
    Generate conversational response using Ollama + Llama2.
    """
    try:
        prompt = f"""You are a helpful healthcare nurse assistant. A patient has reported the following:

Symptoms: {symptoms_text}
Urgency Level: {triage_level}
Patient Age: {patient_history.get('age', 'Unknown')}
Comorbidities: {', '.join(patient_history.get('comorbidities', []))}

Provide a brief, empathetic response acknowledging their symptoms. Do NOT diagnose. Keep it under 100 words."""
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "Unable to generate response").strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return f"I understand you're experiencing {symptoms_text}. Please wait for a nurse to review your case."


@app.post("/check-in", response_model=CheckInResponse)
async def patient_check_in(request: CheckInRequest):
    """
    Process patient check-in with real-time triage scoring.
    
    Flow:
    1. Parse symptoms/vitals
    2. Score with deterministic triage
    3. Generate LLM response (via Ollama)
    4. Generate nurse alert if urgent
    
    Returns: Triage score + LLM response + alert status
    """
    
    try:
        # Step 1: Convert to internal format
        symptoms = [
            Symptom(
                name=s.name,
                severity=s.severity,
                duration_hours=s.duration_hours,
                onset=s.onset,
            )
            for s in request.symptoms
        ]
        
        vitals_dict = request.vitals.dict(exclude_none=True)
        
        patient_history = {
            "patient_id": request.patient.patient_id,
            "age": request.patient.age,
            "comorbidities": request.patient.comorbidities,
            "current_meds": request.patient.current_meds,
            "allergies": request.patient.allergies,
        }
        
        # Step 2: Score with triage module
        triage_result = scorer.score_check_in(symptoms, vitals_dict, patient_history)
        
        # Step 3: Generate check-in record
        check_in_id = str(uuid.uuid4())
        check_in_record = {
            "check_in_id": check_in_id,
            "patient_id": request.patient.patient_id,
            "timestamp": datetime.now().isoformat(),
            "symptoms": [s.name for s in symptoms],
            "vitals": vitals_dict,
            "message": request.message,
            "triage_result": triage_result,
        }
        
        # Step 4: Create nurse alert if urgent
        nurse_alert = None
        if triage_result["nurse_alert"]:
            alert_id = str(uuid.uuid4())
            nurse_alert = {
                "alert_id": alert_id,
                "check_in_id": check_in_id,
                "patient_id": request.patient.patient_id,
                "timestamp": datetime.now().isoformat(),
                "urgency": triage_result["urgency_level"].name,
                "symptoms": [s.name for s in symptoms],
                "reasoning": triage_result["reasoning"],
                "recommended_action": triage_result["recommended_action"],
                "status": "pending",
                "reviewed_at": None,
                "reviewed_by": None,
            }

        # Saved BEFORE the (slow) LLM call so an urgent alert is never lost if the LLM hangs or fails.
        store.save_check_in(check_in_record, patient_history, nurse_alert)
        
        # Step 5: Generate LLM response
        symptoms_text = ", ".join([s.name for s in symptoms])
        llm_response = await generate_llm_response(
            symptoms_text,
            triage_result["urgency_level"].name,
            patient_history,
        )
        
        store.set_llm_response(check_in_id, llm_response)
        
        # Step 6: Return response
        return CheckInResponse(
            check_in_id=check_in_id,
            timestamp=check_in_record["timestamp"],
            urgency_level=triage_result["urgency_level"].name,
            urgency_score=triage_result["score"],
            confidence=triage_result["confidence"],
            reasoning=triage_result["reasoning"],
            red_flags=triage_result["red_flags_detected"],
            nurse_alert=triage_result["nurse_alert"],
            recommended_action=triage_result["recommended_action"],
            llm_response=llm_response,
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/check-in/{check_in_id}", response_model=Dict)
async def get_check_in(check_in_id: str):
    """Retrieve check-in details"""
    record = store.get_check_in(check_in_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Check-in not found")
    return record

@app.get("/alerts/pending")
async def get_pending_alerts():
    """Get all pending nurse alerts"""
    pending = store.pending_alerts()  # newest first
    return {
        "count": len(pending),
        "alerts": pending,
    }

@app.get("/alerts/{alert_id}", response_model=NurseAlertResponse)
async def get_alert(alert_id: str):
    """Get alert details"""
    alert = store.get_alert(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return NurseAlertResponse(
        alert_id=alert["alert_id"],
        check_in_id=alert["check_in_id"],
        patient_id=alert["patient_id"],
        urgency_level=alert["urgency"],
        symptoms_summary=", ".join(alert["symptoms"]),
        reasoning=alert["reasoning"],
        timestamp=alert["timestamp"],
        status=alert["status"],
    )

@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, reviewed_by: str):
    """Nurse acknowledges alert"""
    if not store.acknowledge_alert(alert_id, reviewed_by):
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "acknowledged", "alert_id": alert_id}

@app.get("/patient/{patient_id}/history")
async def get_patient_history(patient_id: str):
    """Get patient's check-in history"""
    total, last_ten = store.patient_history(patient_id, limit=10)
    return {
        "patient_id": patient_id,
        "total_check_ins": total,
        "check_ins": last_ten,  # newest first
    }

@app.get("/stats")
async def get_system_stats():
    """System statistics"""
    return {**store.stats(), "timestamp": datetime.now().isoformat()}

@app.post("/conversation/start")
async def start_conversation(request: LiveConversationRequest):
    """
    Start a new live conversation session.
    Returns session_id for multi-turn chat.
    """
    import uuid
    session_id = request.session_id or str(uuid.uuid4())
    
    store.create_session(session_id, request.patient_id, request.context or {}, datetime.now().isoformat())
    
    # Add patient's initial message
    store.add_message(session_id, "patient", request.message, datetime.now().isoformat())
    
    # Generate LLM response
    context = request.context or {}
    patient_history = {
        "patient_id": request.patient_id,
        "age": context.get("age"),
        "comorbidities": context.get("comorbidities", []),
        "current_meds": context.get("current_meds", []),
        "allergies": context.get("allergies", []),
    }
    
    llm_response = await generate_llm_response(
        request.message,
        context.get("urgency_level", "ROUTINE"),
        patient_history,
    )
    
    store.add_message(session_id, "assistant", llm_response, datetime.now().isoformat())
    
    return LiveConversationResponse(
        session_id=session_id,
        message=llm_response,
        role="assistant",
        timestamp=datetime.now().isoformat(),
        conversation_history=store.get_session(session_id)["history"],
    )

@app.post("/conversation/import")
async def import_conversation(request: ConversationImportRequest):
    """
    Load a finished doctor-patient conversation as a session, so a draft note can be generated from it.
    Clinician turns (doctor/nurse/assistant) are stored as role "assistant", which the note model reads as the
    clinician. No LLM reply is generated.
    """
    turns = [t for t in request.turns if t.content.strip()]
    if not turns:
        raise HTTPException(status_code=422, detail="No non-empty turns to import")
    if len(turns) > 300:
        raise HTTPException(status_code=422, detail="Too many turns (limit 300)")
    session_id = str(uuid.uuid4())
    store.create_session(session_id, request.patient_id, {}, datetime.now().isoformat())
    for turn in turns:
        role = "patient" if turn.role.strip().lower() == "patient" else "assistant"
        store.add_message(session_id, role, turn.content.strip(), datetime.now().isoformat())
    return {
        "session_id": session_id,
        "patient_id": request.patient_id,
        "message_count": len(turns),
        "conversation_history": store.get_session(session_id)["history"],
    }

@app.post("/conversation/{session_id}/message")
async def send_conversation_message(session_id: str, request: LiveConversationRequest):
    """
    Send a message in ongoing conversation.
    Returns LLM response with full conversation history.
    """
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Add patient message
    store.add_message(session_id, "patient", request.message, datetime.now().isoformat())
    
    # Build context for LLM
    context = session.get("context", {})
    patient_history = {
        "patient_id": session["patient_id"],
        "age": context.get("age"),
        "comorbidities": context.get("comorbidities", []),
        "current_meds": context.get("current_meds", []),
        "allergies": context.get("allergies", []),
    }
    
    # Generate response
    llm_response = await generate_llm_response(
        request.message,
        context.get("urgency_level", "ROUTINE"),
        patient_history,
    )
    
    store.add_message(session_id, "assistant", llm_response, datetime.now().isoformat())
    
    return LiveConversationResponse(
        session_id=session_id,
        message=llm_response,
        role="assistant",
        timestamp=datetime.now().isoformat(),
        conversation_history=store.get_session(session_id)["history"],
    )

@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """
    Retrieve full conversation history.
    """
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "patient_id": session["patient_id"],
        "created_at": session["created_at"],
        "message_count": len(session["history"]),
        "history": session["history"],
    }

# ============================================================================
# Draft clinical notes (fine-tuned model) + nurse review
# ============================================================================

@app.post("/conversation/{session_id}/note")
async def generate_draft_note(session_id: str):
    """
    Turn a finished conversation into a DRAFT clinical note for nurse review.
    The draft is never final: a nurse must approve it via /notes/{note_id}/review.
    """
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    reason = note_generator.disabled_reason()
    if reason:
        raise HTTPException(status_code=503, detail=f"Note generation unavailable: {reason}")

    try:
        # model call blocks (GPU), so run it off the event loop
        text, quality_flags = await asyncio.to_thread(note_generator.generate_with_flags, session["history"])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Note generation failed: {e}")

    note_id = str(uuid.uuid4())
    store.save_draft({
        "note_id": note_id,
        "session_id": session_id,
        "patient_id": session["patient_id"],
        "created_at": datetime.now().isoformat(),
        "draft_text": text,
        # output-guard flags: repetition / truncated / continued_dialogue / very_short.
        # Any flag means the draft is low confidence and should be checked closely.
        "quality_flags": quality_flags,
        "low_confidence": bool(quality_flags),
        "model_version": note_generator.model_version,  # e.g. llama-2-aftercare-v4
        "disclaimer": "AI-generated draft. Not clinically validated. Requires nurse review.",
    })
    return store.get_draft(note_id)

@app.get("/notes/pending")
async def get_pending_notes():
    """Get all pending notes awaiting nurse review"""
    pending = store.pending_drafts()  # newest first
    return {
        "count": len(pending),
        "notes": pending,
    }

@app.get("/notes/{note_id}")
async def get_note(note_id: str):
    """Retrieve a single note"""
    note = store.get_draft(note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return note

@app.post("/notes/{note_id}/review")
async def review_note(note_id: str, request: NoteReviewRequest):
    """Nurse approves (optionally edited) or rejects a draft. Edits are logged as feedback data."""
    outcome, detail = store.review_draft(note_id, request.reviewed_by, request.final_text, request.approved)
    if outcome == "missing":
        raise HTTPException(status_code=404, detail="Note not found")
    if outcome == "conflict":
        raise HTTPException(status_code=409, detail=f"Note already {detail}")
    # review_draft also wrote the (draft, nurse final) pair to note_feedback_log
    return {"status": detail["status"], "note_id": note_id}


@app.get("/feedback/notes")
async def get_note_feedback(limit: int = 100):
    """
    (model draft, nurse final) pairs logged by /notes/{id}/review, for the next fine-tuning round.
    Contains the conversation text. Prototype only: protect or remove before real patient data.
    """
    records = store.feedback_records(limit=limit)
    return {"count": len(records), "records": records}

@app.websocket("/ws/conversation/{session_id}")
async def websocket_conversation(websocket: WebSocket, session_id: str):
    """
    WebSocket for real-time streaming conversation.
    Streams LLM response token-by-token.
    """
    if store.get_session(session_id) is None:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    await websocket.accept()
    try:
        while True:
            # Receive patient message
            data = await websocket.receive_json()
            patient_message = data.get("message", "")
            
            if not patient_message:
                continue
            
            session = store.get_session(session_id)
            
            # Add to history
            store.add_message(session_id, "patient", patient_message, datetime.now().isoformat())
            
            # Build context
            context = session.get("context", {})
            patient_history = {
                "patient_id": session["patient_id"],
                "age": context.get("age"),
                "comorbidities": context.get("comorbidities", []),
                "current_meds": context.get("current_meds", []),
                "allergies": context.get("allergies", []),
            }
            
            # Generate response
            llm_response = await generate_llm_response(
                patient_message,
                context.get("urgency_level", "ROUTINE"),
                patient_history,
            )
            
            store.add_message(session_id, "assistant", llm_response, datetime.now().isoformat())
            
            # Send response
            await websocket.send_json({
                "type": "message",
                "role": "assistant",
                "content": llm_response,
                "timestamp": datetime.now().isoformat(),
                "message_count": len(store.get_session(session_id)["history"]),
            })
    except Exception as e:
        print(f"WebSocket conversation error: {e}")

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket for real-time nurse alerts"""
    await websocket.accept()
    try:
        while True:
            # Send pending alerts every 5 seconds
            pending = store.pending_alerts()
            await websocket.send_json({
                "type": "alerts_update",
                "pending_count": len(pending),
                "alerts": pending,
                "timestamp": datetime.now().isoformat(),
            })
            await asyncio.sleep(5)
    except Exception as e:
        print(f"WebSocket error: {e}")

# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    store.init_schema()  # creates the tables if missing; fails fast if the database is unreachable
    print("🏥 AI Aftercare API started")
    print(f"🗄️  Database: {store.describe()}")
    print("📚 Docs: http://localhost:8000/docs")
    print(f"🧠 LLM (responses): {OLLAMA_URL}/api/generate (model: {LLM_MODEL})")
    print(f"📝 Note adapter: {note_generator.adapter_path or 'NOT SET'} -> "
          f"{'enabled' if note_generator.enabled else 'disabled (' + str(note_generator.disabled_reason()) + ')'}")
    
    # Test Ollama connection
    try:
        import requests
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama connected successfully")
        else:
            print("⚠️ Ollama connection failed - check if Ollama is running")
    except Exception as e:
        print(f"⚠️ Could not connect to Ollama: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down AI Aftercare API")

# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print(f"Starting with Ollama at {OLLAMA_URL}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,  # Set True for development
    )
