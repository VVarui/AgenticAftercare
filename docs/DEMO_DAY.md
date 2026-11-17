# Demo day checklist (20 minutes, live demo on a RunPod GPU)

## The day before
- [ ] Start a pod following `RUNPOD_API.md` (24 GB GPU, ports 8000, 8501, 8502 exposed). Generate one real note to download the base model and warm everything up.
- [ ] Rehearse the demo below twice with the sample conversations. Write down how long the first note takes.
- [ ] Take screenshots of a good run (alert on the dashboard, a note with flags, the System tab). They are your fallback.
- [ ] Decide what you will say about the metrics (see "Say these out loud"): the slides still list ROUTINE/BLEU/section coverage as pending.
- [ ] Put the repo link, the user guide and the experiment log in the final package.

## One hour before
- [ ] Start (or restart) the pod, start the three tmux sessions, open `/health` and check `note_generation` says `enabled`.
- [ ] Generate one note now. After a restart the first note loads the model (minutes), later notes take seconds.
- [ ] Open the patient page and the nurse dashboard in two browser windows, side by side. Type your name in the dashboard sidebar.
- [ ] Set the browser zoom so the back row can read it.

## Suggested running order (20 minutes)
| Min | What | Notes |
|---|---|---|
| 0-2 | The problem: nurses re-type notes and triage by hand; what this prototype is and is not | Say "prototype, synthetic data, not a medical device" once, early |
| 2-5 | Architecture: patient page, nurse dashboard, API, database, models | One diagram (README) |
| 5-8 | The model: Llama-2-7B + LoRA, the v4 stop-token fix, honest results | See "Say these out loud" |
| 8-15 | **Live demo** (below) | |
| 15-17 | Limits and next steps: clinician validation, patient sign-in, HIPAA, hosting (AWS files prepared, not deployed) | |
| 17-20 | Questions | |

## The live demo (about 7 minutes)
1. **Patient page:** Patient ID `P001`, symptom *Chest pain*, severity 9. Send. The patient sees "Call 911 or go to the emergency room now" and no scores.
2. **Nurse dashboard, Nurse alerts:** the emergency appears with the reasons. Click Acknowledge (your name must be in the sidebar). Point out that the triage is fixed rules, not AI.
3. **Patient page again:** *Mild cough*, severity 3, 24 hours. The patient gets "check-in received". No alert is created.
4. **Something else:** choose *Something else*, type "my wound smells bad". The check-in is held at MONITOR and the nurse is told to read it. This shows why free text is never reported as stable.
5. **Conversation and note tab:** paste a sample conversation from `docs/sample_conversations.txt`, Load conversation, **Generate draft note**.
6. **Draft note review:** read the draft next to the conversation. Show a quality flag if one fired. Find one unsupported detail and delete it. Approve.
7. **System tab:** show the feedback record (draft versus final text). This is how model quality would be measured later.

## Say these out loud
- v4 did **not** beat the unmodified base model on the standard text-overlap score (ROUGE-1, about 0.44 for v4). It wrote the expected note sections more often, and it invented more numbers than the base model (the invented-number proxy: v4 0.51-0.68 versus base 0.26-0.39, on 18-19 conversations, one run each).
- The "flags fell from 15 of 16 to 1 of 16" result counts stop failures (runaway text, cut-off notes, repeats), **not** hallucinations.
- A v5 model trained on grounded notes was tested and did worse, so v4 stayed. The retrained v4 adapter was not re-measured.
- The triage rules have not been reviewed by a clinician. Vitals alone (for example oxygen 88) do not trigger a rule yet.

## If something breaks
| Problem | Do this |
|---|---|
| Proxy page will not load | Use the API's `/docs` page (steps at the end of section 4 of `RUNPOD_API.md`) or show your screenshots |
| Note generation fails or is too slow | Switch the API to demo mode (`NOTE_DEMO_MODE=1`, restart): recorded v4 notes, clearly labelled, and say so |
| Pod will not start (no GPUs free) | Try another region or GPU type; keep demo mode and screenshots as the fallback |
| Wi-Fi is bad | Run the stack on your own PC in demo mode (`docker compose -p ai-aftercare up -d`), same screens, recorded notes |

## After the demo
- [ ] Terminate the pod (billing continues until you do).
