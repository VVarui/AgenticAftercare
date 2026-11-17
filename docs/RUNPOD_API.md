# Live note generation on a RunPod GPU (API + nurse dashboard + patient page on the pod)

## 1. Pod
* Template: a RunPod PyTorch template. GPU: 24 GB or more (the 7B model needs about 14 GB in bf16).
* **Expose HTTP ports 8000, 8501 and 8502** when you deploy (Edit template / Edit pod -> "Expose HTTP Ports"). Your URLs are
  `https://<POD_ID>-8000.proxy.runpod.net` (API), `https://<POD_ID>-8501.proxy.runpod.net` (nurse dashboard) and
  `https://<POD_ID>-8502.proxy.runpod.net` (patient page).
  If you forget, add them and let the pod restart; files on the pod volume stay.
* Zip your whole project folder on your PC (the one that holds `api.py`, `patient.py` and `models\llama-2-aftercare-v4`; name the zip
  `ai-aftercare.zip`), upload it to `/workspace` through the pod's Jupyter file browser, then open a terminal.

## 2. Install and start the API (in tmux)
```bash
cd /workspace && unzip -qo ai-aftercare.zip && cd ai-aftercare   # if the zip has an extra folder level, cd into it
ls models/llama-2-aftercare-v4                                # adapter_config.json + adapter_model.safetensors
export HF_HOME=/root/hf
pip install "transformers==4.44.2" "peft==0.12.0" "accelerate==0.33.0" sentencepiece protobuf "numpy<2" \
    fastapi "uvicorn[standard]" pydantic requests httpx psutil streamlit
tmux new -s api
export NOTE_ADAPTER_PATH=/workspace/ai-aftercare/models/llama-2-aftercare-v4
export NOTE_BASE_MODEL=NousResearch/Llama-2-7b-chat-hf
unset POSTGRES_URL                                            # in-memory database
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```
Open `https://<POD_ID>-8000.proxy.runpod.net/health`. `note_generation` must say `enabled`.
If it says `disabled: ...`, the reason is in the text (adapter path wrong, or no GPU).
(Detach tmux with Ctrl+B then D.)

## 3. Start the dashboard (second tmux)
```bash
tmux new -s ui
cd /workspace/ai-aftercare
API_URL=http://localhost:8000 streamlit run demo.py --server.port 8501 --server.address 0.0.0.0 \
    --server.headless true --server.enableCORS false --server.enableXsrfProtection false
```
Open `https://<POD_ID>-8501.proxy.runpod.net`.

Then the patient page, in a third tmux session (`tmux new -s patient`):
```bash
cd /workspace/ai-aftercare
API_URL=http://localhost:8000 streamlit run patient.py --server.port 8502 --server.address 0.0.0.0 \
    --server.headless true --server.enableCORS false --server.enableXsrfProtection false
```
Open `https://<POD_ID>-8502.proxy.runpod.net`. Both pages talk to the same API, so a check-in sent from the patient page shows up
under **Nurse alerts** on the dashboard. Everything is in memory: restarting the API empties it.

## 4. Generate a real note
1. Conversation tab -> "Load a doctor-patient conversation" -> paste one block from `docs/sample_conversations.txt`
   (without the `===` line) -> Load conversation.
2. **Generate draft note.** The first call downloads the 13 GB base model and loads it: allow 5-10 minutes.
   Later notes take a few seconds. Do the first one well before any demo.
3. Draft note review tab: read the draft next to the conversation, edit, Approve. The System tab shows the feedback record.

**If the dashboard page will not load through the proxy** (blank page or endless "connecting"), use the API directly:
`https://<POD_ID>-8000.proxy.runpod.net/docs` -> `POST /conversation/import` with
`{"patient_id":"P1","turns":[{"role":"doctor","content":"How is your knee?"},{"role":"patient","content":"Better, pain is a 4 out of 10."}]}`
-> copy `session_id` -> `POST /conversation/{session_id}/note` -> `GET /notes/pending`.

## 5. What to expect (be honest about it)
* The note is the real v4 model on your conversation. It may be very short, repeat itself, or invent a detail or the
  patient's sex. Quality flags appear on the draft and the review screen. That is the point of the nurse review.
* Rehearse with the sample conversations before showing anyone: the recorded v4 notes for them are in the notebook.

## 6. Cleanup
Download anything you want to keep (nothing is saved), then **terminate the pod**. Billing continues until you do.
