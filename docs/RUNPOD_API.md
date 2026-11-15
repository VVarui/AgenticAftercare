# Live note generation on a RunPod GPU (API + dashboard on the pod)

Your PC has no NVIDIA GPU, so the fine-tuned v4 model runs on a pod. The pod runs the API and the dashboard only:
no Docker, no Postgres (the API falls back to an in-memory database), no Ollama (chat replies are the canned
fallback, so **load a full conversation instead of chatting**: the dashboard's "Load a doctor-patient conversation").
Everything on the pod is lost when it is stopped. That is fine for a test.

## 1. Pod
* Template: a RunPod PyTorch template. GPU: 24 GB or more (the 7B model needs about 14 GB in bf16).
* **Expose HTTP ports 8000 and 8501** when you deploy (Edit template / Edit pod -> "Expose HTTP Ports"). Your URLs are
  `https://<POD_ID>-8000.proxy.runpod.net` (API) and `https://<POD_ID>-8501.proxy.runpod.net` (dashboard).
  If you forget, add them and let the pod restart; files on the pod volume stay.
* Zip your whole `app_v1` folder on your PC (it already holds `models\llama-2-aftercare-v4`, `triage_scorer.py` and
  everything else), upload the zip to `/workspace` through the pod's Jupyter file browser, then open a terminal.

## 2. Install and start the API (in tmux)
```bash
cd /workspace && unzip -qo app_v1.zip && cd app_v1          # if the zip has an extra folder level, cd into it
ls models/llama-2-aftercare-v4                                # adapter_config.json + adapter_model.safetensors
export HF_HOME=/root/hf
pip install "transformers==4.44.2" "peft==0.12.0" "accelerate==0.33.0" sentencepiece protobuf "numpy<2" \
    fastapi "uvicorn[standard]" pydantic requests httpx psutil streamlit
tmux new -s api
export NOTE_ADAPTER_PATH=/workspace/app_v1/models/llama-2-aftercare-v4
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
cd /workspace/app_v1
API_URL=http://localhost:8000 streamlit run demo.py --server.port 8501 --server.address 0.0.0.0 \
    --server.headless true --server.enableCORS false --server.enableXsrfProtection false
```
Open `https://<POD_ID>-8501.proxy.runpod.net`.

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
