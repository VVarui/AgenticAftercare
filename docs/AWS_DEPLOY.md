# Deploying to AWS: one GPU server (option A)

One EC2 GPU machine runs everything with `docker-compose.aws.yml`: Streamlit dashboard, API, PostgreSQL, Redis,
Ollama and the Llama-2-7B + v4 adapter that writes the draft notes. Only HTTPS (port 443) is open to the internet,
behind a username and password.

> **Status: written but not yet tested on a real AWS account.** Console menu names, AMI names and prices change.
> Where a step says "check", look at the current AWS page rather than trusting this file.
> **Synthetic data only.** A password gate is not HIPAA compliance. Real patient data would need a signed AWS BAA,
> encryption, access control and audit review that this project does not have.

## What it costs (check current prices before you start)
| Item | Rough size | Note |
|---|---|---|
| g5.xlarge (1x A10G 24 GB) | about $1.00 per hour in us-east-1 | **only while the instance is running** |
| Disk, 150 GB gp3 | about $12 per month | charged while the instance is stopped too |
| Public IPv4 address | about $3.60 per month | charged even when the instance is stopped |

A run of 3 hours of testing plus a presentation is a few dollars. **Leaving it running for a month is about $750.**
Stop the instance when you are not using it, and set a billing alarm in step 1.

## 1. Account setup (do this first: the GPU quota can take hours or days)
1. Sign in to the AWS console, pick one region (for example `us-east-1`) and **stay in it** for everything.
2. **Billing alarm:** Billing and Cost Management, then Budgets, then create a monthly cost budget of, say, $25 with an email alert at 80%.
3. **GPU quota:** Service Quotas, then Amazon EC2, then search for **"Running On-Demand G and VT instances"** and request **4** vCPUs
   (a g5.xlarge uses 4). New accounts often start at 0, so launching fails until this is approved. Request it today.

## 2. Launch the server
EC2, then **Launch instance**:
* **Name:** `aftercare`
* **AMI:** search the AMI catalog for **"Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)"** (Amazon-owned).
  It already has the NVIDIA driver, Docker and the NVIDIA container toolkit. Names change; any current "Deep Learning Base ... Nvidia Driver" Ubuntu AMI works.
* **Instance type:** `g5.xlarge`
* **Key pair:** create new, type RSA, `.pem`. Save the file somewhere safe (for example `C:\Users\Warui\.ssh\aftercare.pem`). You cannot download it again.
* **Network, security group:** create new with exactly three inbound rules:
  * SSH, port 22, source **My IP** (not 0.0.0.0/0)
  * HTTP, port 80, source Anywhere (needed once to get the HTTPS certificate)
  * HTTPS, port 443, source Anywhere
* **Storage:** 150 GiB gp3 (the image, the 13 GB base model and Docker layers need room).

Then **Elastic IPs**, then **Allocate Elastic IP address**, then **Associate** it to the instance. This keeps the address the same after you stop and start.
Write the address down. Make a dashed copy for the free domain below, for example `3.91.12.45` becomes `3-91-12-45.sslip.io`.

## 3. Connect (from PowerShell on your PC)
```powershell
icacls "$env:USERPROFILE\.ssh\aftercare.pem" /inheritance:r /grant:r "$($env:USERNAME):R"
ssh -i "$env:USERPROFILE\.ssh\aftercare.pem" ubuntu@3.91.12.45
```
Everything from here down runs **on the server** unless it says "on your PC". If your IP changes (new network), update the SSH rule's source in the security group.

## 4. Check the GPU and Docker
```bash
nvidia-smi                                       # should list an NVIDIA A10G
docker --version && docker compose version
docker run --rm --gpus all ubuntu nvidia-smi     # a container can see the GPU
```
If the last command fails, the AMI lacks the container toolkit: pick the Deep Learning Base AMI above, or install `nvidia-container-toolkit`.

## 5. Get the code (private repo, read-only deploy key)
```bash
ssh-keygen -t ed25519 -f ~/.ssh/deploy_key -N ""
cat ~/.ssh/deploy_key.pub
```
On github.com: repo, Settings, **Deploy keys**, Add deploy key, paste the line, leave "Allow write access" **unchecked**.
```bash
GIT_SSH_COMMAND="ssh -i ~/.ssh/deploy_key -o IdentitiesOnly=yes" git clone git@github.com:VVarui/AgenticAftercare.git
cd AgenticAftercare
```

## 6. Copy the adapter (it is not in git)
On the server: `mkdir -p ~/AgenticAftercare/models/llama-2-aftercare-v4`

On your PC (PowerShell, in the folder that holds your adapter; change the paths and IP):
```powershell
scp -i "$env:USERPROFILE\.ssh\aftercare.pem" models\llama-2-aftercare-v4\adapter_config.json models\llama-2-aftercare-v4\adapter_model.safetensors ubuntu@3.91.12.45:~/AgenticAftercare/models/llama-2-aftercare-v4/
```
Back on the server, `ls -l models/llama-2-aftercare-v4` should show `adapter_model.safetensors` of **159,967,880 bytes**.
(Later you can keep the adapter in a private S3 bucket instead and `aws s3 cp` it down.)

## 7. Settings (`.env`)
```bash
cp .env.example .env
docker run --rm caddy caddy hash-password --plaintext 'ChooseALongPassword'   # prints a hash starting with $2a$
nano .env
```
Set these lines:
```
POSTGRES_PASSWORD=<long random>
GRAFANA_PASSWORD=<long random>
NOTE_DEMO_MODE=                         # leave EMPTY: this server makes real notes
SITE_ADDRESS=3-91-12-45.sslip.io        # your Elastic IP with dashes, or your own domain pointing at it
BASIC_AUTH_USER=nurse
BASIC_AUTH_HASH='$2a$14$...the hash...'   # keep the single quotes
```
`sslip.io` is a free service that turns `3-91-12-45.sslip.io` into the IP address, so Caddy can get a real HTTPS certificate without you buying a domain.
Keep `.env` private: it is in `.gitignore`.

## 8. Start it
```bash
docker compose -f docker-compose.aws.yml up -d --build      # first build takes roughly 10-20 minutes
docker compose -f docker-compose.aws.yml ps                  # wait until api, ollama, postgres, redis show "healthy"
docker exec ai-aftercare-ollama ollama pull llama2           # chat replies (a few GB, one time)
```
Download the base model now, so the first note does not stall for minutes:
```bash
docker exec ai-aftercare-api python -c "from huggingface_hub import snapshot_download as d; d('NousResearch/Llama-2-7b-chat-hf', allow_patterns=['*.json','*.safetensors','tokenizer.model'])"
```

## 9. Check it works
1. Browser: `https://3-91-12-45.sslip.io` (your address). Log in with `nurse` and your password. The first visit can take a minute while the certificate is issued.
2. On the server: `docker exec ai-aftercare-api curl -s localhost:8000/health`. `note_generation` should say the model is available, not `DEMO` and not a GPU error.
3. In the dashboard: create a patient, run a check-in, import a conversation (use `docs/sample_conversations.txt`), generate a draft note. The first note loads the model onto the GPU (about a minute); later ones are faster.
4. `nvidia-smi` on the server during a note shows memory use; the two models together should fit in the 24 GB.

## 10. Daily use and cost control
* **Stop when finished** (console: instance, Instance state, **Stop**). Containers restart by themselves when you start it again, and the data in the Docker volumes stays on the disk.
* Stop the instance **before** you close the laptop: a running GPU instance costs about $1 an hour whether you use it or not.
* Open Grafana (optional): `docker compose -f docker-compose.aws.yml --profile monitoring up -d`, then on your PC `ssh -i ...aftercare.pem -L 3000:localhost:3000 ubuntu@3.91.12.45` and visit http://localhost:3000.
* Update after pushing code: `cd ~/AgenticAftercare && GIT_SSH_COMMAND="ssh -i ~/.ssh/deploy_key -o IdentitiesOnly=yes" git pull && docker compose -f docker-compose.aws.yml up -d --build`.
* Back up the database before tearing down: `docker exec ai-aftercare-postgres pg_dump -U aftercare aftercare > backup.sql`.

## 11. Tear down when the course is over
Terminate the instance, **release the Elastic IP**, delete the volume if it is left over, and delete the key pair and security group. Then check Billing the next day.

## Troubleshooting
| Symptom | Likely cause and fix |
|---|---|
| `VcpuLimitExceeded` at launch | The GPU quota in step 1 is not approved yet. |
| Browser: connection refused / certificate error | Ports 80 and 443 not open in the security group, or `SITE_ADDRESS` does not match the Elastic IP. `docker logs ai-aftercare-caddy` shows the reason. |
| Login prompt rejects the right password | `BASIC_AUTH_HASH` lost its single quotes or was truncated. Recreate it and run `docker compose -f docker-compose.aws.yml up -d`. |
| `docker compose` says a variable is missing | A required line in `.env` is empty (the message names it). |
| api not healthy, `could not select device driver "nvidia"` | The container toolkit is missing (step 4 last command). |
| Note endpoint answers 503 | `/health` shows why: no GPU visible, or the adapter files are missing from `models/llama-2-aftercare-v4`. |
| CUDA out of memory | Another process is using the GPU. `nvidia-smi`; stop Ollama's model with `docker exec ai-aftercare-ollama ollama stop llama2`. |
| Database error about a missing table or column | Postgres kept an old volume: that is the legacy-schema case. Back it up if needed, then `docker volume rm ai-aftercare_postgres_data` (the project name is the folder name, so check `docker volume ls`). |
| Disk full | `docker system prune -f`, or grow the volume in the console. |

## Known limits of this setup
* One server, one customer, no autoscaling, no high availability. Each customer would get their own copy.
* The password gate is shared by all users and there is no per-nurse login. The API has no authentication of its own; it is only safe because it is not exposed to the internet.
* No backups run automatically. No HIPAA controls (BAA, encryption review, audit review).
