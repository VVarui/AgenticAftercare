# AI Aftercare: Model Experiment Log

Fine-tuning Llama-2-7B-chat (LoRA) to turn a patient conversation into a **draft** clinical note
for nurse review. This file records every training run, what was measured, what went wrong, and what is still open.

Items marked **TBD** have not been measured yet.

---

## 1. Runs at a glance

| | v1 | v3 | v4 |
|---|---|---|---|
| Script | `train.py` (first version) | `train_v3/train.py` | `train_v4/train.py` |
| Training data | MTS-Dialog only (see leakage below) | MTS-Dialog (1,201 original + 4,802 augmented English) + ACI-Bench train (67 notes x 8) = ~6,541 rows | MTS-Dialog (1,201 + 4,802) + ACI-Bench train (67 notes x 3) = **6,204 rows** |
| Validation | leaked (see below) | ACI valid (20) + MTS validation (100) | same: **120 rows** |
| Test data | leaked | ACI test sets 1-3 + MTS test (kept out of training) | same |
| Epochs / steps | 3 | 3 planned, 1,224 steps | **2, 774 steps** |
| Learning rate | 2e-4 (cosine, 100 warmup steps) | 2e-4 | 2e-4 |
| Effective batch | 16 | 16 (4 x 4) | 16 (4 x 4) |
| LoRA | r=16, alpha=32, dropout 0.05, q/k/v/o/gate/up/down | same | same |
| Max sequence length | -- | 4,096 (run flag) | 4,096 (run flag) |
| Padding token | EOS (bug) | EOS (**bug**) | **`<unk>` (fixed)** |
| Eval / save every | -- | 500 steps | **200 steps** |
| Early stopping | -- | patience 3 | patience 3 |
| Best checkpoint | -- | step 500 (eval loss 1.386) | **TBD** (step 200: 1.1997; step 400: 1.2839; step 600: TBD) |
| Final train loss | -- | -- | 0.949 (average over run) |
| Training time | -- | -- | 1 h 03 min |
| Hardware | RunPod, A100-class GPU | RunPod, A100 80 GB | RunPod GPU pod |
| Status | **Do not cite** (inflated scores) | Superseded | **Current** |

Common settings: base model `NousResearch/Llama-2-7b-chat-hf`, bf16, gradient checkpointing,
trl 0.9.6, transformers 4.44.2, peft 0.12.0.

---

## 2. What went wrong and how it was found

### v1: data leakage
The augmented MTS-Dialog files (near-duplicates of the original cases) were read as ordinary data, and the official
test sets were used in training. Scores looked very good and meant nothing. **Fix:** v3 loader separates train /
augmented / validation / test by file name.

### v3: the model never learned to stop
Symptoms on 8 test conversations: notes that ran to the token limit, repeated the same sentence ~10 times,
or continued into an invented "Doctor: / Patient:" dialogue.
**Cause:** the script set `pad_token = eos_token`. The SFT data collator replaces every pad-token label with the ignore
value, so the real end-of-note token was also ignored and never learned.
**Fix (v4):** padding uses `<unk>`; the script now also checks before training that the text ends with the EOS token
(observed in the run: last token ids `[800, 29889, 2]`, EOS id 2, pad id 0).

### v3: overfitting and invented content
Eval loss was best at step 500 (1.386) and rose to 1.80 by step 1,000. ACI-Bench has only 67 training notes and was
repeated 8 times; full-note outputs filled in exams, medication lists and diagnoses that were never discussed.
**Fix (v4):** ACI repeated 3 times, 2 epochs, evaluation every 200 steps.

### v3 also: prompt format mismatch (found late)
MTS-Dialog uses `Doctor: ... / Patient: ...` lines; ACI-Bench uses `[doctor] ... / [patient] ...`. The first
section-style tests used the bracket format. `note_generator.py` now formats each style the way it was trained.

---

## 3. Hallucination test (8 short check-in conversations)

`integration_v2/test_hallucination.py` runs 8 conversations (fever/cough, knee surgery recovery, diabetes follow-up,
heart failure check-in, child earache, headache, refill phone call, and a control where the doctor really examines the
patient). It flags: exam sections or findings with no exam in the conversation, diagnoses, numbers, symptoms and
medications that were never mentioned, and an assumed gender. The generator also reports output-guard flags
(`repetition`, `truncated`, `continued_dialogue`, `very_short`).

| Model / prompt | Notes with >=1 flag | Total flags (8 notes) | Notes cut off or looping |
|---|---|---|---|
| v3, full-note prompt ("plain"), no guards | 7 / 8 | 43 | most |
| v3, full-note prompt, with guards | 7 / 8 | 32 | most (guards trim them) |
| v3, "grounded" prompt (use only what was said) | 7 / 8 | 27 | most |
| v3, short-summary ("section") prompt | 8 / 8 | 19 | 8 / 8 flagged repetition, 5 truncated |
| **v4, "plain"** | 5 / 8 | **9** | 1 / 8 |
| **v4, "section"** | 3 / 8 | **5** | 0 / 8 |

Notes flagged by the output guards (full-note + short-summary prompts, 16 notes per model): **v3 15 / 16, v4 1 / 16**.
These are the stop-failure flags (repetition, runaway dialogue, truncated). The `very_short` guard, added later, flags 6 of the 16 v4 notes (re-checked on the saved notes).
(v3 counts come from applying the guard rules to the saved v3 outputs.)

Reading (a manual, rough assessment of the 16 v4 notes, not a metric):
- ~6 were usable drafts (e.g. knee-surgery and heart-failure notes, which are faithful).
- ~4 contained invented details: a dose ("metformin 500 mg"), a side ("left ear"), an age and sex ("65-year-old female",
  "4-year-old girl"), or a wrong fact (child "not drinking much" when the mother said she was drinking).
- ~6 were too short to be useful (e.g. "Lisinopril.", "Sore throat.").

**Limits of this test:** the checks are keyword rules and miss subtle errors (e.g. "left ear" and the drinking error were
not flagged). Only 8 conversations, written by the developers. A clean score does not mean a note is correct.
A clinician review of a sample is still required.

---

## 4. Still to do (no results yet)

| Item | Why | Status |
|---|---|---|
| ACI-Bench test evaluation of v4 (`eval_aci.py`) | ROUGE / BLEU / section coverage for the report | **TBD** |
| Base-model baseline (`eval_aci.py --no-adapter`) | Shows whether fine-tuning beats the untouched model | **TBD** |
| Hyperparameter experiment (e.g. LR 1e-4, LoRA rank 8, dropout 0.1) | Lowers overfitting; covers tuning | **TBD** |
| Compare the two saved v4 checkpoints on the hallucination test | Early checkpoints may write shorter notes | **TBD** |
| Clinician review of generated notes (`eval_review.csv` rubric) | Only valid check of clinical correctness | **TBD** |
| Fairness / demographic check (same conversation, swapped names/ages/sex) | The model invents ages and sexes | not planned |

## 5. Known limitations (state these in any report or demo)
- Output is a **draft for nurse review**, never a final record.
- Training data is English, mostly US, role-played or curated (MTS-Dialog, ACI-Bench). It is not real check-in data and is not representative of all patient groups.
- The model sometimes invents ages, sexes, doses, sides of the body, and facts, and sometimes writes notes that are too short.
- The ACI-Bench full-note style has only 67 real training examples; the short-summary style is better supported.
- Fairness across patient groups has not been evaluated.


---

## Update, November 2026: ACI-Bench evaluation, v5, and the adapter in this repo

**ACI-Bench evaluation, v4 vs the untouched base model** (3 test sets, 18-19 notes each, one run, `eval_aci.py` and `make_report.py`):

| | ROUGE-1 | ROUGE-2 | ROUGE-L | Section coverage | Invented-number proxy |
|---|---|---|---|---|---|
| v4, test1/2/3 | 0.42 / 0.45 / 0.46 | 0.19 / 0.20 / 0.21 | 0.25 / 0.24 / 0.25 | 0.26 / 0.25 / 0.07 | 0.51 / 0.58 / 0.68 |
| base model | 0.46 / 0.50 / 0.50 | 0.18 / 0.20 / 0.20 | 0.24 / 0.25 / 0.26 | 0.04 / 0.04 / 0.06 | 0.39 / 0.26 / 0.38 |

v4 did **not** beat the base model on ROUGE-1, tied on ROUGE-2 and ROUGE-L, and clearly won only on section coverage.
It also wrote more unsupported numbers than the base model. Fine-tuning taught structure and style, not facts.
Eval sets are small and BLEU was not run on this data. No clinician has reviewed any output.

**v5 (tried).** Changes: training notes with unsupported sentences removed (`ground.py`), 1 epoch, lr 1e-4,
LoRA r=8, ACI x2. Result on the same evaluation: ROUGE-1 0.27 / 0.33 / 0.32, section coverage 0.04 / 0.03 / 0.00,
hallucination-test flags 7 (short prompt) and 12 (full note) against v4's 5 and 9, and stop failures
(repetition / truncation) in 5 of 16 notes against 1 of 16 for v4. It missed every criterion set in advance except
"not too short". Likely cause (a hypothesis; several settings changed at once): far less ACI exposure
(about 134 note passes against about 402), so the model did not learn the note structure.
**v4 stays the model.**

**The adapter in this repo is a retrain of v4.** The original v4 adapter files could not be recovered, so v4 was retrained with the same script and settings (`training/train_v4.py`), so the numbers above come from
the original run and have **not** been re-measured for the retrained adapter. Training is not bit-for-bit repeatable.
