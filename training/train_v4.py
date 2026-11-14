#!/usr/bin/env python3
"""
AI Aftercare v3: LoRA fine-tune of Llama-2-7B-chat for dialogue -> clinical note.

Data:
  - MTS-Dialog  (data/raw/mts_dialog)  -> short note SECTIONS      (style "section")
  - ACI-Bench   (data/raw/aci_bench/challenge_data/train.csv, valid.csv)
                                       -> FULL visit notes          (style "note")
  The ACI-Bench TEST files are never used here; eval_aci.py uses them for evaluation.
  The prompt tells the model which style to write, so the two do not blur together.

Usage:
  python train.py --dry-run     # check data loading (no GPU needed)
  python train.py               # real run, resumes from last checkpoint automatically
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer

SEED = 42

CONFIG = {
    "model_name": "NousResearch/Llama-2-7b-chat-hf",
    "max_seq_length": 3072,            # ACI-Bench dialogue + note is ~1800 tokens on average
    "output_dir": "./models/llama-2-aftercare-v4",
    "num_epochs": 2,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,  # effective batch = 16
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "save_steps": 200,
    "eval_steps": 200,
    "logging_steps": 50,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    "max_eval_samples": 500,
}

DIALOGUE_KEYS = ("dialogue", "conversation", "conversations", "transcript")
SUMMARY_KEYS = ("summary", "note", "section_text")

INSTRUCTIONS = {
    "section": "Summarize the following patient conversation as a short clinical note section.",
    "note": "Write a complete clinical visit note for the following doctor-patient conversation.",
}


def build_prompt(dialogue: str, style: str) -> str:
    # Keep identical in eval_aci.py. No literal <s>: the tokenizer adds BOS.
    return f"[INST] {INSTRUCTIONS[style]}\n\nDIALOGUE:\n{dialogue}\n\nNOTE: [/INST] "


csv.field_size_limit(10**8)


# ============================================================================
# Data loading
# ============================================================================

def _to_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for turn in value:
            if isinstance(turn, str):
                parts.append(turn)
            elif isinstance(turn, dict):
                parts.append(turn.get("text") or turn.get("value") or "")
        return "\n".join(p for p in parts if p)
    return ""


def _pair_from_record(rec: dict):
    if not isinstance(rec, dict):
        return None
    d = next((rec[k] for k in DIALOGUE_KEYS if rec.get(k)), None)
    s = next((rec[k] for k in SUMMARY_KEYS if rec.get(k)), None)
    if d is None or s is None:
        return None
    d, s = _to_text(d).strip(), _to_text(s).strip()
    return {"dialogue": d, "summary": s} if d and s else None


def _iter_records(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            yield from csv.DictReader(f)
    elif suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            yield data


def load_mts(data_dir: Path, augmented: str = "en") -> dict:
    """Official MTS-Dialog files, sorted into buckets by file name.

    The Augmented-* files are paraphrases/translations of the TRAINING set only, so they
    may go into train but never into validation or test. augmented: none | en | all
    ("en" skips the French/Spanish set 3).
    """
    folder = data_dir / "mts_dialog"
    out = {"train": [], "aug": [], "val": [], "test": []}
    if not folder.exists():
        print(f"  [skip] mts_dialog: folder not found at {folder}")
        return out
    for path in sorted(folder.glob("**/*.csv")):
        name = path.name.lower()
        if "augmented" in name:
            if augmented == "none":
                bucket = None
            elif "set-3" in name or "fr-and-es" in name:
                bucket = "aug" if augmented == "all" else None
            else:
                bucket = "aug"
        elif "trainingset" in name:
            bucket = "train"
        elif "validationset" in name:
            bucket = "val"
        elif "testset" in name:
            bucket = "test"
        else:
            bucket = None
        if bucket is None:
            print(f"  [skip] {path.name}")
            continue
        rows = []
        for rec in _iter_records(path):
            pair = _pair_from_record(rec)
            if pair:
                pair["style"] = "section"
                rows.append(pair)
        print(f"  {path.name}: {len(rows)} pairs -> {bucket}")
        out[bucket].extend(rows)
    return out


def _read_aci_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        d_col = next((c for c in DIALOGUE_KEYS if c in cols), None)
        n_col = next((c for c in ("note", "summary") if c in cols), None)
        if not d_col or not n_col:
            sys.exit(f"ERROR: {path.name} has columns {cols}; expected a dialogue column "
                     f"({DIALOGUE_KEYS}) and a note column (note/summary).")
        rows = []
        for rec in reader:
            d, n = (rec.get(d_col) or "").strip(), (rec.get(n_col) or "").strip()
            if d and n:
                rows.append({"dialogue": d, "summary": n, "style": "note"})
    print(f"  {path.name}: columns {cols} -> {len(rows)} pairs")
    return rows


def load_aci(data_dir: Path):
    base = data_dir / "aci_bench" / "challenge_data"
    train_p, valid_p = base / "train.csv", base / "valid.csv"
    if not train_p.exists() or not valid_p.exists():
        print(f"  [skip] aci_bench: {train_p} / {valid_p} not found")
        return [], []
    return _read_aci_csv(train_p), _read_aci_csv(valid_p)


def prepare_data(data_dir: str, aci_repeat: int, mts_augmented: str) -> dict:
    root = Path(data_dir)
    print(f"Loading data from {root.resolve()}")
    mts = load_mts(root, mts_augmented)
    aci_train, aci_valid = load_aci(root)
    mts_train, mts_aug, mts_val, mts_test = mts["train"], mts["aug"], mts["val"], mts["test"]

    if not (mts_train or mts_aug or aci_train):
        sys.exit("ERROR: no usable training pairs found. Check --data-dir.")

    rng = random.Random(SEED)
    if not mts_val and mts_train:
        rng.shuffle(mts_train)
        cut = max(1, len(mts_train) // 10)
        mts_val, mts_train = mts_train[:cut], mts_train[cut:]
        print("  [warn] no official MTS validation file found; held out 10% of the original training set")
    if not mts_test:
        print("  [warn] no usable MTS test rows (test files have no reference notes?); "
              "eval will use the ACI-Bench test sets only")

    # Safety net: nothing that appears in validation/test may be in train
    held = {r["dialogue"] for r in mts_val + mts_test}
    mts_train = [r for r in mts_train if r["dialogue"] not in held]
    mts_aug = [r for r in mts_aug if r["dialogue"] not in held]

    # ACI-Bench is tiny, so repeat it so it is not drowned out.
    train = mts_train + mts_aug + aci_train * aci_repeat
    rng.shuffle(train)
    validation = aci_valid + mts_val   # ACI first, so the capped eval subset includes it

    splits = {"train": train, "validation": validation, "test": mts_test}
    print(f"\nTrain {len(train)} (MTS original {len(mts_train)} + MTS augmented {len(mts_aug)} "
          f"+ ACI {len(aci_train)} x {aci_repeat})")
    print(f"Val {len(validation)} (ACI {len(aci_valid)} + MTS {len(mts_val)}) | "
          f"MTS test {len(mts_test)}")

    split_dir = root.parent / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for key, rows in splits.items():
        with open(split_dir / f"{key}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved splits to {split_dir.resolve()}")
    return splits


# ============================================================================
# Training
# ============================================================================

def make_formatter(eos_token: str):
    def fmt(example):
        return {"text": build_prompt(example["dialogue"], example["style"])
                        + f"{example['summary']} {eos_token}"}
    return fmt


def train(splits: dict, output_dir: str, resume: bool):
    if not torch.cuda.is_available():
        sys.exit("ERROR: no CUDA GPU detected. Run this on the cloud GPU.")

    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB), "
          f"dtype={'bf16' if use_bf16 else 'fp16'}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    # FIX (v4): v3 used pad_token = eos_token. The SFT data collator sets every pad-token label to
    # -100, so the real end-of-note </s> was masked too and the model never learned to STOP
    # (v3 notes ran on, repeated themselves, or invented new dialogue). Use <unk> for padding.
    assert tokenizer.unk_token_id is not None and tokenizer.unk_token_id != tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"], torch_dtype=dtype, device_map="auto"
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["lora_targets"],
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    fmt = make_formatter(tokenizer.eos_token)
    cols = ["dialogue", "summary", "style"]
    train_ds = Dataset.from_list(splits["train"]).map(fmt, remove_columns=cols)
    val_ds = Dataset.from_list(splits["validation"][: CONFIG["max_eval_samples"]]).map(
        fmt, remove_columns=cols)
    print(f"Train: {len(train_ds)} samples | Val (eval subset): {len(val_ds)} samples")

    # Sanity check: the training text must end with the real EOS token, and EOS must not be the pad token.
    ids = tokenizer(train_ds[0]["text"])["input_ids"]
    print(f"Last 3 token ids of example 0: {ids[-3:]} | eos id = {tokenizer.eos_token_id} | "
          f"pad id = {tokenizer.pad_token_id}")
    if ids[-1] != tokenizer.eos_token_id:
        sys.exit("ERROR: training text does not end with the EOS token id -- the model would not "
                 "learn to stop. Fix make_formatter()/tokenizer before training.")
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        sys.exit("ERROR: pad token equals EOS token; the collator would mask the real EOS.")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        warmup_steps=CONFIG["warmup_steps"],
        weight_decay=CONFIG["weight_decay"],
        learning_rate=CONFIG["learning_rate"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        max_grad_norm=CONFIG["max_grad_norm"],
        logging_dir=f"{output_dir}/logs",
        logging_steps=CONFIG["logging_steps"],
        eval_strategy="steps",
        eval_steps=CONFIG["eval_steps"],
        save_strategy="steps",
        save_steps=CONFIG["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to=["tensorboard"],
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=args,
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        packing=False,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3,
                                         early_stopping_threshold=0.001)],
    )

    last_ckpt = get_last_checkpoint(output_dir) if resume and Path(output_dir).exists() else None
    if last_ckpt:
        print(f"Resuming from {last_ckpt}")
    result = trainer.train(resume_from_checkpoint=last_ckpt)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nTraining complete. Loss: {result.training_loss:.4f}")
    print(f"Adapter saved to: {output_dir}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune Llama-2 for AI Aftercare (v4: EOS fix)")
    p.add_argument("--data-dir", default="./data/raw")
    p.add_argument("--output-dir", default=CONFIG["output_dir"])
    p.add_argument("--epochs", type=int, default=CONFIG["num_epochs"])
    p.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    p.add_argument("--grad-accum", type=int, default=CONFIG["gradient_accumulation_steps"])
    p.add_argument("--max-seq-length", type=int, default=CONFIG["max_seq_length"])
    p.add_argument("--aci-repeat", type=int, default=3,
                   help="Times to repeat the small ACI-Bench train set (default 3; v3 used 8 and overfit)")
    p.add_argument("--eval-steps", type=int, default=CONFIG["eval_steps"],
                   help="Evaluate and save a checkpoint every N steps (default 200)")
    p.add_argument("--mts-augmented", choices=["none", "en", "all"], default="en",
                   help="Which augmented MTS-Dialog sets to add to TRAIN (default en = the two "
                        "English back-translation sets; 'all' adds the French/Spanish one too)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Load/split data and print examples, then exit (no GPU needed)")
    a = p.parse_args()

    CONFIG["num_epochs"] = a.epochs
    CONFIG["batch_size"] = a.batch_size
    CONFIG["gradient_accumulation_steps"] = a.grad_accum
    CONFIG["max_seq_length"] = a.max_seq_length
    CONFIG["eval_steps"] = CONFIG["save_steps"] = a.eval_steps

    splits = prepare_data(a.data_dir, a.aci_repeat, a.mts_augmented)

    if a.dry_run:
        for style in ("section", "note"):
            rows = [r for r in splits["train"] if r["style"] == style]
            if not rows:
                continue
            avg_d = sum(len(r["dialogue"]) for r in rows) // len(rows)
            avg_s = sum(len(r["summary"]) for r in rows) // len(rows)
            print(f"\n--- style '{style}': {len(rows)} train rows, avg dialogue {avg_d} chars, "
                  f"avg note {avg_s} chars ---")
            r = random.Random(0).choice(rows)
            print(f"DIALOGUE: {r['dialogue'][:300]}\nNOTE:     {r['summary'][:300]}")
        steps = (len(splits["train"]) // (a.batch_size * a.grad_accum)) * a.epochs
        print(f"\nApprox optimizer steps for this run: {steps}")
        sys.exit(0)

    train(splits, a.output_dir, resume=not a.no_resume)
