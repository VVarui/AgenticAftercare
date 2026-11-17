# Agentic Aftercare

A prototype nurse-assistant for post-visit check-ins. A patient checks in, rule-based **triage** scores the urgency
and raises a **nurse alert** when needed, and a fine-tuned **Llama-2-7B-chat (LoRA)** model turns a finished
doctor-patient conversation into a **draft clinical note** that a nurse reviews, edits and approves.

> **Prototype for a course project. Not a medical device. Not clinically validated. Use synthetic data only.**

## What is in here

| Part | What it does |
|---|---|
| `triage_scorer.py` | Deterministic, explainable urgency rules (symptoms and vitals). Not an LLM. |
| `api.py` + `db.py` | FastAPI service. Check-ins, alerts, conversations, draft notes, nurse review, feedback log and audit log in PostgreSQL (SQLite in memory when no `POSTGRES_URL` is set, for tests). |
| `note_generator_aftercare.py` | Loads the LoRA adapter and writes a draft note. Output guards flag loops, cut-off text, runaway dialogue and very short notes (`quality_flags`). |
| `demo.py` | Streamlit nurse dashboard: check-in, alerts, conversation, **draft note review**, system status. |
| `patient.py` | Separate Streamlit **patient page**: plain-language check-in, one safety message, follow-up chat. Shows no triage scores or rule details. |
| `training/train_v4.py` | The LoRA fine-tuning script (data is not included). |
| `docs/` | **[User guide](docs/USER_GUIDE.md)** (nurse walkthrough and admin reference), experiment log with honest results, app notes, RunPod and AWS guides, sample conversations. |

```
Patient page (8502) ----------+
                              +--> FastAPI (8000) --> PostgreSQL   (records, drafts, feedback, audit)
Nurse dashboard (8501) -------+         |-> Ollama                  (patient chat replies)
                                        |-> Llama-2-7B + LoRA v4   (draft notes, needs an NVIDIA GPU)
```

## Quick start (no GPU needed)

```bash
cp .env.example .env            # set the two passwords BEFORE the first start (a later change needs a Postgres volume reset)
docker compose up -d --build    # api, frontend, postgres, redis, ollama, prometheus, grafana
```
* Nurse dashboard: http://localhost:8501   **Patient page: http://localhost:8502**   API docs: http://localhost:8000/docs   Health: http://localhost:8000/health
* Chat replies need a model in Ollama: `docker exec ai-aftercare-ollama ollama pull llama2`.
* `Dockerfile.api` copies the adapter, which is **not** in git: see [`models/README.md`](models/README.md) first.
* Without an NVIDIA GPU, `POST /conversation/{id}/note` answers **503** with the reason. To try the whole review flow
  anyway, set `NOTE_DEMO_MODE=1` in `.env`: the API then returns **recorded** v4 notes, clearly labelled `DEMO`
  (not generated from your conversation).
* The triage rules know a fixed list of symptom names (`chest_pain`, `dyspnea`, ...); the dashboard offers them in a list. Spaces and capitals are tolerated; an unknown name is reported back and the check-in is held at MONITOR instead of scoring as stable.

Real note generation on a GPU: [`docs/RUNPOD_API.md`](docs/RUNPOD_API.md).

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests/test_store.py tests/test_note_generator.py tests/test_triage.py tests/test_patient.py
python tests/test_demo.py
python -m pytest -q tests/test_api.py
```
GitHub Actions runs these on every push (`.github/workflows/tests.yml`): unit tests, API tests and a check that the
compose files are valid. None of them need a GPU, so model quality is not tested by CI.

## The model

* Base: `NousResearch/Llama-2-7b-chat-hf` (a mirror of Meta's Llama 2; the Llama 2 Community License applies).
  Adapter: LoRA r=16, alpha=32, trained on MTS-Dialog and ACI-Bench (not redistributed here; check each licence).
* On ACI-Bench (3 test sets of 18-19 notes, one run) v4 did **not** beat the untouched base model on ROUGE-1
  (0.42-0.46 against 0.46-0.50). It wins on section coverage (0.25-0.26 against 0.04 on two sets) and writes
  more unsupported numbers than the base model.

Full details, including what was not measured: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Known limits

* Notes can be very short and repetitive.
* No clinician validation. Small evaluation sets.

## Deploying

GPU-pod testing: [`docs/RUNPOD_API.md`](docs/RUNPOD_API.md).

AWS (one GPU EC2 instance, HTTPS plus a password gate, real note generation): [`docs/AWS_DEPLOY.md`](docs/AWS_DEPLOY.md),
using `docker-compose.aws.yml` and `Caddyfile`.
