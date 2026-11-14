"""
AI Aftercare: persistence layer.

Replaces the in-memory dictionaries that api.py used (check-ins, alerts, conversations,
draft notes, feedback log) with database tables, so data survives an API restart.

Backends
  * PostgreSQL  when POSTGRES_URL (or DATABASE_URL) starts with postgresql://   (docker compose)
  * SQLite file when the URL is sqlite:///path/to/file.db
  * SQLite in memory when no URL is set (local development / tests; NOT persistent)

One schema works on both backends: ids and timestamps are TEXT (ISO-8601 strings, which sort
correctly), lists/dicts are stored as JSON text, booleans as 0/1.
This is a prototype layer: one connection guarded by a lock, calls are synchronous.
"""
import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

LEGACY_SCHEMA_MESSAGE = (
    "The database still has the old init_db.sql tables (patients has an 'id' column). "
    "They were never written to by the API, so delete only the Postgres volume once: "
    "docker compose down, then docker volume rm <project>_postgres_data (find the exact name with "
    "docker volume ls), then start the stack again. The API now creates its own tables at startup."
)

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        age INTEGER,
        comorbidities TEXT NOT NULL DEFAULT '[]',
        current_meds TEXT NOT NULL DEFAULT '[]',
        allergies TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS check_ins (
        check_in_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(patient_id),
        timestamp TEXT NOT NULL,
        symptoms TEXT NOT NULL DEFAULT '[]',
        vitals TEXT NOT NULL DEFAULT '{}',
        message TEXT,
        urgency_level TEXT NOT NULL,
        urgency_score INTEGER NOT NULL,
        confidence REAL NOT NULL,
        reasoning TEXT,
        red_flags TEXT NOT NULL DEFAULT '[]',
        nurse_alert INTEGER NOT NULL DEFAULT 0,
        recommended_action TEXT,
        llm_response TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS nurse_alerts (
        alert_id TEXT PRIMARY KEY,
        check_in_id TEXT NOT NULL REFERENCES check_ins(check_in_id),
        patient_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        urgency TEXT NOT NULL,
        symptoms TEXT NOT NULL DEFAULT '[]',
        reasoning TEXT,
        recommended_action TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        reviewed_by TEXT,
        reviewed_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS conversation_sessions (
        session_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL,
        context TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS conversation_messages (
        session_id TEXT NOT NULL REFERENCES conversation_sessions(session_id),
        seq INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TEXT,
        PRIMARY KEY (session_id, seq)
    )""",
    """CREATE TABLE IF NOT EXISTS draft_notes (
        note_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES conversation_sessions(session_id),
        patient_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        draft_text TEXT NOT NULL,
        quality_flags TEXT NOT NULL DEFAULT '[]',
        low_confidence INTEGER NOT NULL DEFAULT 0,
        model_version TEXT,
        final_text TEXT,
        status TEXT NOT NULL DEFAULT 'draft',
        reviewed_by TEXT,
        reviewed_at TEXT,
        disclaimer TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS note_feedback_log (
        note_id TEXT PRIMARY KEY REFERENCES draft_notes(note_id),
        session_id TEXT NOT NULL,
        reviewed_at TEXT NOT NULL,
        reviewed_by TEXT,
        approved INTEGER NOT NULL,
        edited INTEGER NOT NULL,
        model_version TEXT,
        quality_flags TEXT NOT NULL DEFAULT '[]',
        dialogue_history TEXT NOT NULL DEFAULT '[]',
        model_draft TEXT NOT NULL,
        nurse_final TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS audit_log (
        audit_id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        actor TEXT NOT NULL,
        action TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        detail TEXT NOT NULL DEFAULT '{}'
    )""",
    "CREATE INDEX IF NOT EXISTS idx_check_ins_patient ON check_ins(patient_id, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_status ON nurse_alerts(status, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_drafts_status ON draft_notes(status, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id)",
]

DEMO_PATIENTS = [
    ("DEMO001", 62, ["COPD", "HTN"], ["Albuterol", "Lisinopril"]),
    ("DEMO002", 45, ["Diabetes"], ["Metformin"]),
]


def _now() -> str:
    return datetime.now().isoformat()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _loads(text: Optional[str], default: Any) -> Any:
    if text is None or text == "":
        return default
    return json.loads(text)


class Store:
    def __init__(self, url: Optional[str] = None):
        if url is None:
            url = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL") or ""
        self._lock = threading.RLock()
        self._conn = None
        if url.startswith(("postgres://", "postgresql://")):
            self.backend = "postgres"
            self._url = url
        else:
            self.backend = "sqlite"
            self._path = url[len("sqlite:///"):] if url.startswith("sqlite:///") else ":memory:"
            if not self._path:
                self._path = ":memory:"

    # ------------------------------------------------------------------ plumbing
    @property
    def persistent(self) -> bool:
        return self.backend == "postgres" or self._path != ":memory:"

    def describe(self) -> str:
        if self.backend == "postgres":
            return "postgres"
        return "sqlite (memory, NOT persistent)" if self._path == ":memory:" else f"sqlite file {self._path}"

    def _connect(self):
        if self.backend == "postgres":
            try:
                import psycopg2
                import psycopg2.extras
            except ImportError as exc:
                raise RuntimeError("psycopg2 is not installed: add psycopg2-binary to requirements-prod.txt") from exc
            conn = psycopg2.connect(self._url, cursor_factory=psycopg2.extras.RealDictCursor)
            conn.autocommit = False
            return conn
        import sqlite3
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _q(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.backend == "postgres" else sql

    @contextmanager
    def _tx(self):
        """One transaction: commit on success, roll back on error."""
        with self._lock:
            if self._conn is None or (self.backend == "postgres" and self._conn.closed):
                self._conn = self._connect()
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception as exc:
                try:
                    self._conn.rollback()
                except Exception:
                    self._conn = None
                if self.backend == "postgres" and type(exc).__module__.startswith("psycopg2") \
                        and type(exc).__name__ in ("OperationalError", "InterfaceError"):
                    self._conn = None  # reconnect on the next call
                raise
            finally:
                try:
                    cur.close()
                except Exception:
                    pass

    def _run(self, cur, sql: str, params: tuple = ()):
        cur.execute(self._q(sql), params)
        return cur

    def _rows(self, cur, sql: str, params: tuple = ()) -> List[dict]:
        self._run(cur, sql, params)
        return [dict(r) for r in cur.fetchall()]

    def _row(self, cur, sql: str, params: tuple = ()) -> Optional[dict]:
        rows = self._rows(cur, sql, params)
        return rows[0] if rows else None

    # ------------------------------------------------------------------ schema
    def init_schema(self) -> None:
        with self._tx() as cur:
            if self.backend == "postgres":
                cols = self._rows(
                    cur,
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'patients' AND table_schema = current_schema()",
                )
                if "id" in {c["column_name"] for c in cols}:
                    raise RuntimeError(LEGACY_SCHEMA_MESSAGE)
            for statement in SCHEMA:
                cur.execute(statement)
            now = _now()
            for pid, age, comorbidities, meds in DEMO_PATIENTS:
                self._run(
                    cur,
                    "INSERT INTO patients (patient_id, age, comorbidities, current_meds, allergies, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, '[]', ?, ?) ON CONFLICT (patient_id) DO NOTHING",
                    (pid, age, _dumps(comorbidities), _dumps(meds), now, now),
                )

    def ping(self) -> bool:
        try:
            with self._tx() as cur:
                self._run(cur, "SELECT 1")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ audit
    def _audit(self, cur, actor: str, action: str, entity_type: str, entity_id: str, detail: Optional[dict] = None):
        import uuid
        self._run(
            cur,
            "INSERT INTO audit_log (audit_id, timestamp, actor, action, entity_type, entity_id, detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), _now(), actor or "system", action, entity_type, entity_id, _dumps(detail or {})),
        )

    def audit_events(self, limit: int = 100) -> List[dict]:
        with self._tx() as cur:
            rows = self._rows(cur, "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,))
        for r in rows:
            r["detail"] = _loads(r["detail"], {})
        return rows

    # ------------------------------------------------------------------ patients
    def _upsert_patient(self, cur, patient: dict):
        now = _now()
        self._run(
            cur,
            "INSERT INTO patients (patient_id, age, comorbidities, current_meds, allergies, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (patient_id) DO UPDATE SET "
            "age = COALESCE(excluded.age, patients.age), comorbidities = excluded.comorbidities, "
            "current_meds = excluded.current_meds, allergies = excluded.allergies, updated_at = excluded.updated_at",
            (patient["patient_id"], patient.get("age"), _dumps(patient.get("comorbidities") or []),
             _dumps(patient.get("current_meds") or []), _dumps(patient.get("allergies") or []), now, now),
        )

    def _ensure_patient(self, cur, patient_id: str):
        now = _now()
        self._run(
            cur,
            "INSERT INTO patients (patient_id, created_at, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT (patient_id) DO NOTHING",
            (patient_id, now, now),
        )

    # ------------------------------------------------------------------ check-ins and alerts
    def save_check_in(self, record: dict, patient: dict, alert: Optional[dict] = None) -> None:
        """Save a check-in (and its nurse alert, if any) in one transaction."""
        triage = record["triage_result"]
        level = triage["urgency_level"]
        level = getattr(level, "name", level)
        with self._tx() as cur:
            self._upsert_patient(cur, patient)
            self._run(
                cur,
                "INSERT INTO check_ins (check_in_id, patient_id, timestamp, symptoms, vitals, message, urgency_level, "
                "urgency_score, confidence, reasoning, red_flags, nurse_alert, recommended_action) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record["check_in_id"], record["patient_id"], record["timestamp"], _dumps(record.get("symptoms") or []),
                 _dumps(record.get("vitals") or {}), record.get("message"), level, int(triage["score"]),
                 float(triage["confidence"]), triage.get("reasoning"), _dumps(triage.get("red_flags_detected") or []),
                 1 if triage.get("nurse_alert") else 0, triage.get("recommended_action")),
            )
            self._audit(cur, "patient:" + record["patient_id"], "check_in_created", "check_in",
                        record["check_in_id"], {"urgency": level})
            if alert:
                self._run(
                    cur,
                    "INSERT INTO nurse_alerts (alert_id, check_in_id, patient_id, timestamp, urgency, symptoms, reasoning, "
                    "recommended_action, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
                    (alert["alert_id"], alert["check_in_id"], alert["patient_id"], alert["timestamp"], alert["urgency"],
                     _dumps(alert.get("symptoms") or []), alert.get("reasoning"), alert.get("recommended_action")),
                )
                self._audit(cur, "system", "alert_created", "alert", alert["alert_id"], {"urgency": alert["urgency"]})

    def set_llm_response(self, check_in_id: str, text: str) -> None:
        with self._tx() as cur:
            self._run(cur, "UPDATE check_ins SET llm_response = ? WHERE check_in_id = ?", (text, check_in_id))

    @staticmethod
    def _check_in_from_row(r: dict) -> dict:
        return {
            "check_in_id": r["check_in_id"],
            "patient_id": r["patient_id"],
            "timestamp": r["timestamp"],
            "symptoms": _loads(r["symptoms"], []),
            "vitals": _loads(r["vitals"], {}),
            "message": r["message"],
            "triage_result": {
                "urgency_level": r["urgency_level"],
                "score": r["urgency_score"],
                "confidence": r["confidence"],
                "reasoning": r["reasoning"],
                "red_flags_detected": _loads(r["red_flags"], []),
                "nurse_alert": bool(r["nurse_alert"]),
                "recommended_action": r["recommended_action"],
            },
            "llm_response": r["llm_response"],
        }

    def get_check_in(self, check_in_id: str) -> Optional[dict]:
        with self._tx() as cur:
            r = self._row(cur, "SELECT * FROM check_ins WHERE check_in_id = ?", (check_in_id,))
        return self._check_in_from_row(r) if r else None

    def patient_history(self, patient_id: str, limit: int = 10) -> Tuple[int, List[dict]]:
        with self._tx() as cur:
            total = self._row(cur, "SELECT COUNT(*) AS n FROM check_ins WHERE patient_id = ?", (patient_id,))["n"]
            rows = self._rows(
                cur, "SELECT * FROM check_ins WHERE patient_id = ? ORDER BY timestamp DESC LIMIT ?", (patient_id, limit))
        return int(total), [self._check_in_from_row(r) for r in rows]

    @staticmethod
    def _alert_from_row(r: dict) -> dict:
        return {
            "alert_id": r["alert_id"], "check_in_id": r["check_in_id"], "patient_id": r["patient_id"],
            "timestamp": r["timestamp"], "urgency": r["urgency"], "symptoms": _loads(r["symptoms"], []),
            "reasoning": r["reasoning"], "recommended_action": r["recommended_action"], "status": r["status"],
            "reviewed_at": r["reviewed_at"], "reviewed_by": r["reviewed_by"],
        }

    def get_alert(self, alert_id: str) -> Optional[dict]:
        with self._tx() as cur:
            r = self._row(cur, "SELECT * FROM nurse_alerts WHERE alert_id = ?", (alert_id,))
        return self._alert_from_row(r) if r else None

    def pending_alerts(self) -> List[dict]:
        with self._tx() as cur:
            rows = self._rows(cur, "SELECT * FROM nurse_alerts WHERE status = 'pending' ORDER BY timestamp DESC")
        return [self._alert_from_row(r) for r in rows]

    def acknowledge_alert(self, alert_id: str, reviewed_by: str) -> bool:
        with self._tx() as cur:
            self._run(
                cur,
                "UPDATE nurse_alerts SET status = 'reviewed', reviewed_by = ?, reviewed_at = ? WHERE alert_id = ?",
                (reviewed_by, _now(), alert_id),
            )
            found = cur.rowcount > 0
            if found:
                self._audit(cur, reviewed_by, "alert_acknowledged", "alert", alert_id)
        return found

    def stats(self) -> dict:
        with self._tx() as cur:
            def count(sql, params=()):
                return int(self._row(cur, sql, params)["n"])
            return {
                "total_check_ins": count("SELECT COUNT(*) AS n FROM check_ins"),
                "total_alerts": count("SELECT COUNT(*) AS n FROM nurse_alerts"),
                "urgent_cases": count("SELECT COUNT(*) AS n FROM check_ins WHERE urgency_level IN ('URGENT', 'EMERGENCY')"),
                "pending_alerts": count("SELECT COUNT(*) AS n FROM nurse_alerts WHERE status = 'pending'"),
                "draft_notes_pending": count("SELECT COUNT(*) AS n FROM draft_notes WHERE status = 'draft'"),
                "feedback_records": count("SELECT COUNT(*) AS n FROM note_feedback_log"),
            }

    # ------------------------------------------------------------------ conversations
    def create_session(self, session_id: str, patient_id: str, context: Optional[dict], created_at: str) -> None:
        """Start (or restart) a session; restarting an existing id clears its messages, as before."""
        with self._tx() as cur:
            self._ensure_patient(cur, patient_id)
            self._run(cur, "DELETE FROM conversation_messages WHERE session_id = ?", (session_id,))
            self._run(
                cur,
                "INSERT INTO conversation_sessions (session_id, patient_id, context, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (session_id) DO UPDATE SET patient_id = excluded.patient_id, "
                "context = excluded.context, created_at = excluded.created_at",
                (session_id, patient_id, _dumps(context or {}), created_at),
            )
            self._audit(cur, "patient:" + patient_id, "session_started", "session", session_id)

    def add_message(self, session_id: str, role: str, content: str, timestamp: Optional[str] = None) -> None:
        with self._tx() as cur:
            self._run(
                cur,
                "INSERT INTO conversation_messages (session_id, seq, role, content, timestamp) "
                "SELECT ?, COALESCE(MAX(seq), 0) + 1, ?, ?, ? FROM conversation_messages WHERE session_id = ?",
                (session_id, role, content, timestamp or _now(), session_id),
            )

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._tx() as cur:
            s = self._row(cur, "SELECT * FROM conversation_sessions WHERE session_id = ?", (session_id,))
            if not s:
                return None
            msgs = self._rows(
                cur, "SELECT role, content, timestamp FROM conversation_messages WHERE session_id = ? ORDER BY seq",
                (session_id,))
        return {
            "session_id": s["session_id"], "patient_id": s["patient_id"], "created_at": s["created_at"],
            "context": _loads(s["context"], {}), "history": msgs,
        }

    # ------------------------------------------------------------------ draft notes and feedback
    @staticmethod
    def _draft_from_row(r: dict) -> dict:
        return {
            "note_id": r["note_id"], "session_id": r["session_id"], "patient_id": r["patient_id"],
            "created_at": r["created_at"], "draft_text": r["draft_text"],
            "quality_flags": _loads(r["quality_flags"], []), "low_confidence": bool(r["low_confidence"]),
            "model_version": r["model_version"], "final_text": r["final_text"], "status": r["status"],
            "reviewed_by": r["reviewed_by"], "reviewed_at": r["reviewed_at"], "disclaimer": r["disclaimer"],
        }

    def save_draft(self, note: dict) -> None:
        with self._tx() as cur:
            self._run(
                cur,
                "INSERT INTO draft_notes (note_id, session_id, patient_id, created_at, draft_text, quality_flags, "
                "low_confidence, model_version, final_text, status, reviewed_by, reviewed_at, disclaimer) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 'draft', NULL, NULL, ?)",
                (note["note_id"], note["session_id"], note["patient_id"], note["created_at"], note["draft_text"],
                 _dumps(note.get("quality_flags") or []), 1 if note.get("low_confidence") else 0,
                 note.get("model_version"), note.get("disclaimer")),
            )
            self._audit(cur, "system", "note_drafted", "note", note["note_id"],
                        {"model_version": note.get("model_version"), "flags": note.get("quality_flags") or []})

    def get_draft(self, note_id: str) -> Optional[dict]:
        with self._tx() as cur:
            r = self._row(cur, "SELECT * FROM draft_notes WHERE note_id = ?", (note_id,))
        return self._draft_from_row(r) if r else None

    def pending_drafts(self) -> List[dict]:
        with self._tx() as cur:
            rows = self._rows(cur, "SELECT * FROM draft_notes WHERE status = 'draft' ORDER BY created_at DESC")
        return [self._draft_from_row(r) for r in rows]

    def review_draft(self, note_id: str, reviewed_by: str, final_text: str, approved: bool):
        """Approve (optionally edited) or reject a draft and log the pair as feedback.

        Returns ("ok", note) | ("missing", None) | ("conflict", current_status).
        """
        with self._tx() as cur:
            r = self._row(cur, "SELECT * FROM draft_notes WHERE note_id = ?", (note_id,))
            if not r:
                return "missing", None
            if r["status"] != "draft":
                return "conflict", r["status"]
            status = "approved" if approved else "rejected"
            when = _now()
            self._run(
                cur,
                "UPDATE draft_notes SET status = ?, final_text = ?, reviewed_by = ?, reviewed_at = ? WHERE note_id = ?",
                (status, final_text, reviewed_by, when, note_id),
            )
            history = self._rows(
                cur, "SELECT role, content, timestamp FROM conversation_messages WHERE session_id = ? ORDER BY seq",
                (r["session_id"],))
            edited = final_text.strip() != r["draft_text"].strip()
            self._run(
                cur,
                "INSERT INTO note_feedback_log (note_id, session_id, reviewed_at, reviewed_by, approved, edited, "
                "model_version, quality_flags, dialogue_history, model_draft, nurse_final) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (note_id, r["session_id"], when, reviewed_by, 1 if approved else 0, 1 if edited else 0,
                 r["model_version"], r["quality_flags"], _dumps(history), r["draft_text"], final_text),
            )
            self._audit(cur, reviewed_by, "note_" + status, "note", note_id, {"edited": edited})
            note = self._draft_from_row(self._row(cur, "SELECT * FROM draft_notes WHERE note_id = ?", (note_id,)))
        return "ok", note

    def feedback_records(self, limit: int = 100) -> List[dict]:
        with self._tx() as cur:
            rows = self._rows(cur, "SELECT * FROM note_feedback_log ORDER BY reviewed_at DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            out.append({
                "note_id": r["note_id"], "session_id": r["session_id"], "reviewed_at": r["reviewed_at"],
                "reviewed_by": r["reviewed_by"], "approved": bool(r["approved"]), "edited": bool(r["edited"]),
                "model_version": r["model_version"], "quality_flags": _loads(r["quality_flags"], []),
                "dialogue_history": _loads(r["dialogue_history"], []),
                "model_draft": r["model_draft"], "nurse_final": r["nurse_final"],
            })
        return out


# Shared instance used by api.py (reads POSTGRES_URL / DATABASE_URL once, at import time).
store = Store()
