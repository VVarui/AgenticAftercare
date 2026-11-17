# AI Aftercare: User Guide

For nurses using the dashboard, and for whoever runs the system (section 7).

> **This is a prototype built for a course project. It is not a medical device and has not been validated by clinicians.
> Use synthetic (made-up) patients only. Never enter real patient information.**

## 1. What the system does

| Job | How it works | Who decides |
|---|---|---|
| **Triage** a patient check-in (how urgent?) | Fixed, written rules on symptoms and vital signs. Not an AI model. | The rules suggest; **a nurse acts** |
| **Chat** replies to the patient | A general language model (Ollama) | Not clinical advice |
| **Draft a clinical note** from a doctor-patient conversation | A fine-tuned Llama-2-7B model ("v4") | **A nurse must review and approve every draft** |

Nothing the system produces goes into a patient record on its own. A draft note is only a starting point.

## 2. Getting started
1. Open the dashboard address you were given (locally `http://localhost:8501`; on AWS an `https://` address with a username and password).
2. In the **left sidebar**, type **your name**. Approving, rejecting and acknowledging are disabled until you do, so every action is recorded under a name.
3. Check the sidebar status. **API connected** means it is working. It also shows the database and the note-model status:
   * a model name or "available": real notes can be generated
   * **DEMO**: notes are recorded examples, not generated from your conversation
   * "disabled" or "no GPU": the note button will explain that it is unavailable

**The patient page.** Patients use a separate page (locally `http://localhost:8502`; on AWS the same address with `patient.` in front). It asks in plain language how they feel, shows one clear message, and offers a chat with an automated assistant:

| Patient's check-in scores | What the patient is told |
|---|---|
| EMERGENCY | Call 911 or go to the emergency room now; do not wait for a call back |
| URGENT | A nurse will contact you today; call 911 if you feel worse |
| MONITOR | Your care team will follow up within a day or two |
| ROUTINE | Check-in received; send another if anything changes |

Patients never see scores, red flags or the rules' reasoning; those appear on the nurse dashboard. A patient who picks **"Something else"** is held at MONITOR and the nurse is told to read their message, because the rules cannot score free text. A check-in that scores URGENT or EMERGENCY appears under **Nurse alerts** within seconds, and the patient's chat messages can be read in the conversation list. The patient page has **no sign-in**, so anyone can type any Patient ID: it is a demo of the flow, not a deployable patient app.

The five tabs are along the top of the nurse dashboard: **Check-in, Nurse alerts, Conversation & note, Draft note review, System.**

## 3. Check-in tab: score a patient's report
1. Fill in **Patient ID** (for example `P001`), age, conditions and medications (comma separated).
2. Pick one or more **Symptoms** from the list. A slider appears for each; set severity 1 (mild) to 10 (worst).
3. Set **Duration in hours** (applies to every symptom).
4. Enter vitals if you have them: temperature (F), heart rate, SpO2, systolic blood pressure. **Leave a vital at 0 to leave it out.**
5. Type the patient's message, then click **Submit check-in**.

The result appears as a colored box:

| Level | Color | Meaning | Alert created? |
|---|---|---|---|
| **ROUTINE** | green | Nothing in the rules was triggered. | No |
| **MONITOR** | blue | Watch closely; follow up in 24-48 hours. | No |
| **URGENT** | orange | A nurse should call the patient today. | **Yes** |
| **EMERGENCY** | red | Refer to emergency care immediately. | **Yes** |

Below the box you also see the **recommended action**, the **reasons** (which rule fired) and any **red flags**.

Things to know:
* A symptom the rules do not recognise is not scored. It is listed in the reasoning, and the check-in is held at **MONITOR** so it is not reported as stable. Read those yourself.
* "ROUTINE" means "no rule fired", not "the patient is fine". The rules only look at the symptoms and vitals you enter. For example, a low-but-not-critical oxygen reading (SpO2 88) does not trigger a rule at all. When in doubt, use clinical judgement and escalate.

## 4. Nurse alerts tab
Lists check-ins that scored URGENT or EMERGENCY and have not been handled. Each card shows the patient, time, symptoms, why it was flagged and the recommended action.
* Click **Refresh** to reload.
* After you have dealt with the patient, click **Acknowledge**. The alert disappears and your name is saved with it.

## 5. Conversation & note tab: produce a draft note
There are two ways to get a conversation into the system.

**A. Load a finished conversation (recommended for notes).** Use the second form, headed **Or load a doctor-patient conversation**: enter the Patient ID and paste the transcript, one line per turn, each starting with `Doctor:` or `Patient:`:
```
Doctor: What brings you in today?
Patient: My knee has been swollen since the surgery last week.
Doctor: Any fever or drainage from the incision?
Patient: No fever, a little redness.
```
Click **Load conversation**. A line without a label continues the previous turn.

**B. Chat live.** Use the first form: enter the patient's first message and click **Start conversation**, then type further messages at the bottom. Replies come from the general chat model; the note model works best on a real doctor-patient exchange as in A.

Then click **Generate draft note**. The first note after a restart can take a few minutes while the model loads. The draft appears, any **warnings** appear under it (section 6), and the draft is sent to the review queue. **New conversation** clears the screen to start another.

If you see "Note generation is not available here", the server has no GPU or the model files are missing (section 8). If you see a **DEMO mode** message, the note is a recorded example and will not match your conversation; use it only to practise the review steps.

## 6. Draft note review tab: the step that matters
Every draft waits here until a nurse approves or rejects it. Each draft shows the **conversation on the left** and an **editable draft on the right**.

**Read the draft against the conversation, sentence by sentence.** In testing, this model:
* **invents details** that were never said (numbers, diagnoses, exam findings)
* **assumes the patient's sex** (he/she) when the conversation never states it
* sometimes writes very short notes, repeats itself or stops mid-sentence
* can omit things the nurse would want recorded

Warning boxes explain what the automatic output checks found:

| Warning | What it means | What to do |
|---|---|---|
| The model repeated itself | Repeated sentences were removed | Check that nothing important was lost |
| The draft may be cut off | Ends mid-sentence | Finish or rewrite the end |
| The model kept inventing dialogue | The extra text was removed | Check the rest carefully |
| The draft is very short, so it is probably incomplete | Fewer than 8 words | It missed content; rewrite it from the conversation |
| **CHECK CLOSELY** in the title | At least one warning fired | Treat the whole draft with extra suspicion |

These checks catch formatting failures, **not wrong facts**. A draft with no warning can still contain invented details.

**To finish:**
1. Edit the text on the right until it is accurate and complete. Delete anything the conversation does not support.
2. Click **Approve** (the edited text is the final note) or **Reject** (the draft is not usable).
3. Your name, the original draft and your final text are saved. This feedback record is how the model's quality can be measured later.

## 7. System tab and admin reference
The **System** tab shows the database type, the note-model status, counts (check-ins, pending alerts, drafts waiting, feedback records) and the latest draft-versus-final feedback.

**For whoever runs the system**

| Task | Command (run in the project folder) |
|---|---|
| Patient page | `http://localhost:8502` (set `CLINIC_PHONE` in `.env` to show a phone number in its safety messages) |
| Start locally | `cp .env.example .env`, set passwords, then `docker compose -p ai-aftercare up -d --build` |
| Check health | open `http://localhost:8000/health` (API docs at `/docs`) |
| Stop | `docker compose -p ai-aftercare down` (never add `-v`: it deletes the stored chat models) |
| Real notes on a GPU | see [`RUNPOD_API.md`](RUNPOD_API.md) (test) or [`AWS_DEPLOY.md`](AWS_DEPLOY.md) (server) |
| Demo without a GPU | set `NOTE_DEMO_MODE=1` in `.env` and restart: recorded notes, clearly labelled |
| Back up the database | `docker exec ai-aftercare-postgres pg_dump -U aftercare aftercare > backup.sql` |

Where the data lives: patients, check-ins, alerts, conversations, draft notes, nurse feedback and an audit log are in **PostgreSQL**; the model adapter is a file (not in git). Monitoring numbers (Prometheus/Grafana) do not contain patient information.

## 8. If something goes wrong
| What you see | Likely cause | What to try |
|---|---|---|
| Sidebar shows an error instead of "API connected" | API not running or wrong address | Ask the administrator; `docker compose ps` |
| "Check-in failed" | The API rejected the input | Read the message; check Patient ID and that age and severities are valid |
| Chat replies are an error or empty | The chat model is not downloaded | `docker exec ai-aftercare-ollama ollama pull llama2` |
| "Note generation is not available here" | No NVIDIA GPU, or adapter files missing | `/health` shows the reason; use DEMO mode or a GPU server |
| First note takes minutes | The model is loading | Wait; later notes are faster |
| Approve / Reject / Acknowledge greyed out | No name in the sidebar | Enter your name |
| "Nothing to load" when pasting | Lines do not start with `Doctor:` or `Patient:` | Add the labels |

## 9. Limits you should keep in mind
* **Prototype.** No clinician has validated the triage rules or the note model. The model was trained on public sample dialogues, and in testing it did not beat the unmodified base model on the standard text-overlap score (it did better at writing the expected note sections, and worse on invented numbers).
* **Rules are narrow.** They cover a short list of symptoms and a few vital-sign limits; many real situations are not covered.
* **One shared login** on the AWS setup and no per-nurse accounts; no automatic backups.
* **Not a HIPAA-compliant system.** Real patient data would require a signed agreement with every hosting provider, access controls, encryption and audit review that this project does not have.

## 10. Glossary
* **Triage**: sorting patients by how urgent their situation is.
* **Draft note**: text the AI proposed; not a record until a nurse approves it.
* **Quality flag**: an automatic warning about the *form* of a draft (loops, cut-off, too short).
* **Hallucination**: the model stating something that was not in the conversation.
* **Adapter (LoRA)**: a small add-on file that teaches the base Llama-2 model to write these notes.
* **DEMO mode**: the system returns pre-recorded notes so the review screen can be shown without a GPU.
