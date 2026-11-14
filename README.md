# Agentic Aftercare

A prototype nurse-assistant for post-visit check-ins. A patient checks in, rule-based **triage** scores the urgency
and raises a nurse alert when needed, and a fine-tuned **Llama-2-7B-chat (LoRA)** model turns a finished
doctor-patient conversation into a **draft clinical note** that a nurse reviews, edits and approves.

> **Prototype for a course project. Not a medical device. Not clinically validated. Use synthetic data only.**
> The note model can omit things. A nurse must check every draft.

## What is in here

| Part | What it does |
|---|---|
| `triage_scorer.py` | Deterministic, explainable urgency rules (symptoms and vitals).|
| `api.py` + `db.py` | FastAPI service. Check-ins, alerts, conversations, draft notes, nurse review, feedback log and audit log in PostgreSQL (SQLite in memory when no `POSTGRES_URL` is set, for tests). |
| `note_generator_aftercare.py` | Loads the LoRA adapter and writes a draft note. Output guards flag loops, cut-off text, runaway dialogue and very short notes (`quality_flags`). |
| `demo.py` | Streamlit nurse dashboard: check-in, alerts, conversation, draft note review, system status. |
| `training/train_v4.py` | The LoRA fine-tuning script (data is not included). |
| `docs/` | Experiment log with honest results, app notes, RunPod GPU guide, sample conversations. |

```
Streamlit dashboard (8501) --> FastAPI (8000) --> PostgreSQL   (records, drafts, feedback, audit)
                                  |-> Ollama                    (patient chat replies)
                                  |-> Llama-2-7B + LoRA v4      (draft notes, needs an NVIDIA GPU)
```

## Quick start (no GPU needed)

```bash
cp .env.example .env            # set the two passwords BEFORE the first start (a later change needs a Postgres volume reset)
docker compose up -d --build    # api, frontend, postgres, redis, ollama, prometheus, grafana
```
* Dashboard: http://localhost:8501   API docs: http://localhost:8000/docs   Health: http://localhost:8000/health
* Chat replies need a model in Ollama: `docker exec ai-aftercare-ollama ollama pull llama2`.
* `Dockerfile.api` copies the adapter, which is **not** in git: see [`models/README.md`](models/README.md) first.
* Without an NVIDIA GPU, `POST /conversation/{id}/note` answers **503** with the reason. To try the whole review flow
  anyway, set `NOTE_DEMO_MODE=1` in `.env`: the API then returns **recorded** v4 notes, clearly labelled `DEMO`
  (not generated from your conversation).
* Symptom names must match the triage rules (`chest_pain`, `dyspnea`, ...); the dashboard offers them in a list.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests/test_store.py tests/test_note_generator.py
python tests/test_demo.py
python -m pytest -q tests/test_api.py      # written but not yet run end to end; see CI note
```
GitHub Actions runs these on every push (`.github/workflows/tests.yml`). The API tests are allowed to fail until
they have passed once in CI.

## The model

* Base: `NousResearch/Llama-2-7b-chat-hf` (a mirror of Meta's Llama 2; the Llama 2 Community License applies).
  Adapter: LoRA r=16, alpha=32, trained on MTS-Dialog and ACI-Bench (not redistributed here; check each licence).
* On ACI-Bench (3 test sets of 18-19 notes, one run). It wins on section coverage (0.25-0.26 against 0.04 on two sets) and writes
  more unsupported numbers than the base model. Fine-tuning taught structure and style.
* A bug in v3 (the stop token was masked during training) made notes run on; v4 fixed it
  (15 of 16 test notes with stop failures down to 1 of 16).
* The adapter in this project is a retrain of v4 with the same settings.


## Known limits

* Prototype security: no login on the API, CORS open to all origins, demo database passwords. Do not expose it to the
  internet as it is, and never load real patient data. Real patient data would bring HIPAA obligations
  (agreements with every host, access control, audit review) that this project does not meet.
* Notes can be very short and repetitive.
* No clinician validation. Small evaluation sets.

