# AI Aftercare app, step 1: data now persists (PostgreSQL) + v4 adapter

Built from the files you attached on 2026-11-11. The model is **v4** (v5 did not beat it).

## What changed
| File | Change |
|---|---|
| `db.py` (new) | Database layer. Replaces the 4 in-memory dicts in `api.py`: check-ins, nurse alerts, conversations and messages, draft notes, plus `note_feedback_log` and an `audit_log`. Creates its own tables at startup. PostgreSQL when `POSTGRES_URL` is set; in-memory SQLite otherwise (not persistent). |
| `api.py` | Same endpoints, now backed by `db.py`. Check-ins and alerts are saved **before** the slow LLM call. Draft notes keep the output-guard flags (`quality_flags`, `low_confidence`) and the `model_version`. New `GET /feedback/notes`. `/health` also reports the database and note-model status. |
| `note_generator_aftercare.py` | Only additions: refuses to run on a CPU-only container (see "Important" below), `disabled_reason()`, `model_version`. `clean_note` and prompts are untouched. |
| `Dockerfile.api` | Ships the **v4** adapter (it was v3, the one with the stop-token bug). `--build-arg ADAPTER_DIR=...` selects another. Copies `db.py`. |
| `docker-compose.yml`, `docker-compose.prod.yml` | `POSTGRES_URL` password from `${POSTGRES_PASSWORD:-aftercare}`; mounts `db.py` and `note_generator_aftercare.py`; prod adapter path v1 -> v4. |
| `init_db.sql` | Now only a comment. The old tables were never written to and would clash with the new ones. |
| `requirements-additions.txt` | Add `psycopg2-binary` (and `httpx`, `requests` if missing) to `requirements-prod.txt`. |
| `demo.py` (new) | Streamlit nurse dashboard written from scratch (your original `demo.py` was not available). Tabs: check-in and triage, nurse alerts, conversation and draft note, **draft note review** (conversation next to an editable draft, quality-flag warnings, approve/reject, writes the feedback log), system status. Talks only to the API. |
| `Dockerfile.frontend`, `requirements-frontend.txt` | Lean frontend image (`streamlit`, `requests`; no torch). Fixes the old build, whose requirements list had no `streamlit`. |
| `tests/` | `test_store.py`, `test_note_generator.py`, `test_api.py`, `test_demo.py`. |

## Install
1. Copy these files over yours. Make sure `models/llama-2-aftercare-v4/` (with `adapter_config.json` and `adapter_model.safetensors`) is in the build folder.
2. Add the lines in `requirements-additions.txt` to `requirements-prod.txt`.
3. Reset the old, empty Postgres volume **once** (the API tells you if you forget):
   `docker compose down`, then `docker volume rm <project>_postgres_data` (see `docker volume ls`). Do not use `down -v`: it also deletes your Ollama models.
4. `docker compose up -d --build api`
5. Open http://localhost:8000/health. `"database"` should say `postgres`.
6. Prove persistence: post a check-in, run `docker compose restart api`, then `GET /alerts/pending` still lists it.

## Important: how to demo it (corrections to what I told you earlier)
* **Symptom names must match the triage rules exactly, with underscores:** `chest_pain`, `difficulty_breathing`, `dyspnea`, `fever`, `cough`, `mild_headache`... My earlier example used `"chest pain"` and `"shortness of breath"`, which the scorer does not recognise, so it scores **ROUTINE**. Use this instead (scores EMERGENCY):
```json
{"patient": {"patient_id": "P001", "age": 68, "comorbidities": ["HTN"], "current_meds": ["furosemide"], "allergies": []},
 "symptoms": [{"name": "chest_pain", "severity": 9, "duration_hours": 2}, {"name": "dyspnea", "severity": 8, "duration_hours": 2}],
 "vitals": {"heart_rate": 118, "oxygen_sat": 89, "systolic": 90, "diastolic": 60},
 "message": "My chest hurts and I can't catch my breath."}
```
  For URGENT use `dyspnea` severity 7. For a mild case use `mild_headache` severity 2 with normal vitals.
* **Note generation on your PC:** your old `Dockerfile.api` baked the adapter path into the image, so on a machine without an NVIDIA GPU the first `/note` call would try to load the 7B model on CPU (needs about 28 GB RAM and can freeze the machine). It now answers **503** with the reason. The earlier "503 on your PC" description is only true with this version.

## Trying the note + review flow on a PC without a GPU: demo mode
Real note generation needs an NVIDIA GPU. To try the whole review flow on your PC, start the API in **demo mode**
(PowerShell, in the project folder):
```
$env:NOTE_DEMO_MODE = "1"
docker compose -p ai-aftercare up -d api
```
Then in the dashboard: Conversation tab -> start a chat -> **Generate draft note** -> Draft note review tab -> edit -> Approve.
`/health` shows "DEMO mode".
* Demo mode loads **no model**. It returns one of three notes the real v4 model wrote earlier on a GPU (knee replacement, fever and cough, heart-failure check-in), picked by keywords in the patient's messages (default: knee).
* The text is **not generated from your conversation**. Every draft carries `model_version = "DEMO: recorded v4 output, NOT generated from this conversation"` and the dashboard shows a blue DEMO banner on it. Do not present it as live generation.
* Turn it off with `$env:NOTE_DEMO_MODE = ""` and `docker compose -p ai-aftercare up -d api`.

## Real note generation on RunPod
See `RUNPOD_API.md` (API and dashboard on a GPU pod, load a conversation from `docs/sample_conversations.txt`). New endpoint:
`POST /conversation/import` takes `{"patient_id", "turns":[{"role":"doctor|patient","content":"..."}]}` and returns a
`session_id` for `/conversation/{id}/note`; the dashboard has a paste box for it.

## Behaviour changes to know about
* `GET /check-in/{id}`: `triage_result.urgency_level` is now the name (`"EMERGENCY"`), not the numeric enum value.
* Rejected notes are also written to `note_feedback_log` (with `approved=false`); before, only approvals were.
* `/stats` has two more keys (`draft_notes_pending`, `feedback_records`).

## What was tested, and where
| Check | Result |
|---|---|
| Store: 9 tests on SQLite | 9 passed |
| Same store tests on a real PostgreSQL 16 server | 9 passed, plus restart persistence and legacy-schema detection |
| Note generator guards, GPU/adapter gating: 10 tests | 10 passed |
| Every `api.py` handler called end to end against `db.py` (25 checks) | passed |
| `tests/test_api.py` (FastAPI TestClient) | **not run**: FastAPI could not be installed where I worked. Run `pytest tests/test_api.py` yourself. |
| Real `psycopg2` driver | **not run**: I used a stand-in that sends the same SQL to the real PostgreSQL server. |
| Docker build and `docker compose up` | **not run** (no Docker here). The compose files parse as valid YAML. |

## Not done yet
* **Frontend not run in a browser.** `demo.py` is tested with a fake Streamlit (every tab's code path and the API calls it makes), not with real Streamlit. Rebuild and look at it: `docker compose up -d --build frontend`, then http://localhost:8501.
* **Login / access control.** CORS allows all origins and `GET /feedback/notes` returns conversation text with no authentication. Prototype only: do not put real patient data in it.
* **Triage input check.** Unrecognised symptom names silently score ROUTINE. `tests/test_api.py` has an expected-failure test that documents this.
* **Redis** is still unused; **Prometheus** has no `/metrics` endpoint and the postgres/redis exporters are not in the compose file, so those targets show down.
* `llm_inference.py` is not used by the API (it calls Ollama directly). `DEPLOYMENT_GUIDE.md` you attached is the old v1 text; use `docs/DEPLOYMENT_GUIDE.md` from earlier and add the points above.
* Database calls are synchronous (fine for a demo); timestamps are stored as ISO text.
