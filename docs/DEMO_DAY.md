# Demo day checklist (live demo on a RunPod GPU)

## One hour before
- [ ] Start (or restart) the pod, start the three tmux sessions, open `/health` and check `note_generation` says `enabled`.


## The live demo (about 7 minutes)
1. **Patient page:** Patient ID `P001`, symptom *Chest pain*, severity 9. Send. The patient sees "Call 911 or go to the emergency room now" and no scores.
2. **Nurse dashboard, Nurse alerts:** the emergency appears with the reasons. Click Acknowledge (your name must be in the sidebar). Point out that the triage is fixed rules, not AI.
3. **Patient page again:** *Mild cough*, severity 3, 24 hours. The patient gets "check-in received". No alert is created.
4. **Something else:** choose *Something else*, type "my wound smells bad". The check-in is held at MONITOR and the nurse is told to read it. This shows why free text is never reported as stable.
5. **Conversation and note tab:** paste a sample conversation from `docs/sample_conversations.txt`, Load conversation, **Generate draft note**.
6. **Draft note review:** read the draft next to the conversation. Show a quality flag if one fired. Find one unsupported detail and delete it. Approve.
7. **System tab:** show the feedback record (draft versus final text). This is how model quality would be measured later.


