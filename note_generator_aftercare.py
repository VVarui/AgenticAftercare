"""
AI Aftercare: clinical note generator (uses the fine-tuned LoRA adapter).

Turns a finished check-in conversation into a Draft clinical note for nurse review.
"""

import json
import os
import re
import threading
from typing import Dict, List

BASE_MODEL = os.getenv("NOTE_BASE_MODEL", "NousResearch/Llama-2-7b-chat-hf")
ADAPTER_PATH = os.getenv("NOTE_ADAPTER_PATH", "")
QUANTIZE = os.getenv("NOTE_QUANTIZE", "").lower()
# The 7B model needs an NVIDIA GPU. On CPU it needs ~28 GB RAM and takes minutes per note, so it is
# off unless NOTE_ALLOW_CPU=1 (the API then answers 503 instead of freezing the machine).
ALLOW_CPU = os.getenv("NOTE_ALLOW_CPU", "").lower() in ("1", "true", "yes")
# Demo mode (NOTE_DEMO_MODE=1): no model is loaded. The note endpoint returns one of a few notes that the real
# v4 model wrote earlier on a GPU, so the review screen can be shown on a PC without an NVIDIA GPU.
# Every draft is labelled in model_version, and the text is NOT generated from the conversation.
DEMO_MODE = os.getenv("NOTE_DEMO_MODE", "").lower() in ("1", "true", "yes")
DEMO_LABEL = "DEMO: recorded v4 output, NOT generated from this conversation"
DEMO_NOTES = {
    "knee": ("The patient is recovering well from the knee replacement. The patient rates the pain at 4/10. "
             "The patient is taking acetaminophen twice a day and has stopped the stronger pain medications. "
             "The patient has some swelling but no redness or drainage at the incision site. The patient is doing "
             "the exercises from physical therapy in the morning and evening. The patient is able to bend the knee "
             "to about 90 degrees. The patient will call the office if the swelling gets worse or if the patient "
             "develops a fever."),
    "fever": ("The patient presents today with a fever and cough. The patient states that she has had a fever for "
              "about three days and a cough that won't go away. She states that her temperature was 101.5 this "
              "morning. She states that her chest feels a little tight when she coughs, but she can breathe okay. "
              "She states that she has body aches and is very tired. She denies any nausea. She states that she has "
              "asthma and uses an albuterol inhaler when she needs it. She states that she has taken ibuprofen for "
              "the fever, which helps for a few hours. She states that she will rest and drink fluids and come back "
              "if she does not get better in three days."),
    "heart": ("The patient reports that her breathing is worse today. She is having more difficulty going up the "
              "stairs. She has gained four pounds since Monday. She has some edema in her legs. She has missed her "
              "furosemide the last two days. She will restart it today and the nurse will call her this afternoon. "
              "She will go to the ER if she has difficulty breathing."),
}


def _demo_note(turns: List[Dict]) -> str:
    """Pick a recorded note by keywords in the patient's messages (default: the knee note)."""
    said = " ".join((t.get("content") or "") for t in turns if t.get("role") == "patient").lower()
    if any(w in said for w in ("breath", "furosemide", "weight", "ankle", "stairs", "swollen legs")):
        return DEMO_NOTES["heart"]
    if any(w in said for w in ("fever", "cough", "asthma")):
        return DEMO_NOTES["fever"]
    return DEMO_NOTES["knee"]

MAX_PROMPT_TOKENS = 3200
MAX_NEW_TOKENS = 600

# Keep identical to train_v3/train.py
INSTRUCTIONS = {
    "section": "Summarize the following patient conversation as a short clinical note section.",
    "note": "Write a complete clinical visit note for the following doctor-patient conversation.",
}

# Output guards: truncate repeated or continued dialogue from the model output.
_CUT_MARKERS = re.compile(r"\[/?INST\]|\n\s*(?:Patient:|Doctor:|\[doctor\]|\[patient\])")


def clean_note(text: str):
    """Return cleaned text and guard flags."""
    flags = []
    match = _CUT_MARKERS.search(text)
    if match:
        text = text[:match.start()]
        flags.append("continued_dialogue")

    seen, out_lines, dropped = set(), [], False
    for line in text.split("\n"):
        kept = []
        for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
            key = re.sub(r"\W+", " ", sentence.lower()).strip()
            if len(key.split()) >= 5:
                if key in seen:
                    dropped = True
                    continue
                seen.add(key)
            if sentence:
                kept.append(sentence)
        out_lines.append(" ".join(kept))
    if dropped:
        flags.append("repetition")

    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()
    if text and text[-1] not in ".!?:)":
        cut = max(text.rfind(char) for char in ".!?")
        if cut > len(text) * 0.5:
            text = text[:cut + 1]
        flags.append("truncated")
    if len(text.split()) < 8:
        flags.append("very_short")
    return text, flags


def _loop_stopper(prompt_len: int, window: int = 24):
    """Stop generation when a token window repeats."""
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _Loop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            generated = input_ids[0, prompt_len:].tolist()
            if len(generated) < window * 2:
                return False
            tail = generated[-window:]
            earlier = generated[:-window]
            return any(
                earlier[i:i + window] == tail
                for i in range(len(earlier) - window + 1)
            )

    return StoppingCriteriaList([_Loop()])


def format_dialogue(turns: List[Dict], style: str = "section") -> str:
    """Format conversation turns as the matching training dialogue style."""
    lines = []
    for turn in turns:
        text = (turn.get("content") or turn.get("text") or "").strip()
        if not text:
            continue
        who = "patient" if turn.get("role") == "patient" else "doctor"
        lines.append(f"{who.capitalize()}: {text}" if style == "section" else f"[{who}] {text}")
    return "\n".join(lines)


def build_prompt(dialogue: str, style: str = "section") -> str:
    """Build an inference prompt in the same format used during training."""
    return f"[INST] {INSTRUCTIONS[style]}\n\nDIALOGUE:\n{dialogue}\n\nNOTE: [/INST]"


class NoteGenerator:
    """Lazy-loads a LoRA model and serializes inference access."""

    def __init__(self, base_model: str = BASE_MODEL, adapter_path: str = ADAPTER_PATH):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self._model = None
        self._tok = None
        self._lock = threading.Lock()
        self.last_flags: List[str] = []

    @staticmethod
    def _gpu_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def disabled_reason(self):
        """None when note generation can run, otherwise a short reason (used in the 503 message)."""
        if DEMO_MODE:
            return None
        if not self.adapter_path:
            return "NOTE_ADAPTER_PATH is not set"
        if not os.path.exists(self.adapter_path):
            return f"adapter folder not found: {self.adapter_path}"
        if not ALLOW_CPU and not self._gpu_available():
            return "no NVIDIA GPU available in this container (set NOTE_ALLOW_CPU=1 to force CPU: very slow, ~28 GB RAM)"
        return None

    @property
    def enabled(self) -> bool:
        return self.disabled_reason() is None

    @property
    def demo_mode(self) -> bool:
        return DEMO_MODE

    @property
    def model_version(self) -> str:
        """Adapter folder name, e.g. llama-2-aftercare-v4 (stored with every draft)."""
        if DEMO_MODE:
            return DEMO_LABEL
        return os.path.basename(os.path.normpath(self.adapter_path)) if self.adapter_path else ""

    def _check_adapter(self):
        config_path = os.path.join(self.adapter_path, "adapter_config.json")
        if not os.path.exists(config_path):
            raise RuntimeError(f"adapter_config.json not found in {self.adapter_path}")
        weights = ("adapter_model.safetensors", "adapter_model.bin")
        if not any(os.path.exists(os.path.join(self.adapter_path, name)) for name in weights):
            raise RuntimeError(f"No adapter weights found in {self.adapter_path}")
        with open(config_path, encoding="utf-8") as config_file:
            trained_on = json.load(config_file).get("base_model_name_or_path")
        if trained_on and trained_on != self.base_model:
            print(f"[NoteGenerator] WARNING: adapter trained on {trained_on}, but NOTE_BASE_MODEL is {self.base_model}")

    def _load(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._check_adapter()
        print(f"[NoteGenerator] base={self.base_model} adapter={self.adapter_path} mode={QUANTIZE or 'bf16'}")
        self._tok = AutoTokenizer.from_pretrained(self.base_model)
        self._tok.pad_token = self._tok.eos_token

        if QUANTIZE == "4bit":
            from transformers import BitsAndBytesConfig
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                ),
            )
            self._model = PeftModel.from_pretrained(model, self.adapter_path).eval()
        else:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model, torch_dtype=dtype, device_map="auto"
            )
            self._model = PeftModel.from_pretrained(
                model, self.adapter_path
            ).merge_and_unload().eval()
        print("[NoteGenerator] model ready")

    def generate(self, turns: List[Dict], style: str = "section",
                 max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        """Return note text; flags are available from generate_with_flags."""
        return self.generate_with_flags(turns, style, max_new_tokens)[0]

    def generate_with_flags(self, turns: List[Dict], style: str = "section",
                            max_new_tokens: int = MAX_NEW_TOKENS):
        """Return (note text, guard flags); call via asyncio.to_thread from async code."""
        if style not in INSTRUCTIONS:
            raise ValueError(f"Unknown note style: {style}")
        if not self.enabled:
            raise RuntimeError(f"Note generation disabled: {self.disabled_reason()}")
        dialogue = format_dialogue(turns, style)
        if not dialogue:
            raise ValueError("Conversation is empty")
        if DEMO_MODE:
            return _demo_note(turns), []

        import torch
        with self._lock:
            if self._model is None:
                self._load()
            encoded = self._tok(build_prompt(dialogue, style), return_tensors="pt")
            prompt_length = encoded["input_ids"].shape[1]
            if prompt_length > MAX_PROMPT_TOKENS:
                raise ValueError(
                    f"Conversation too long for note generation ({prompt_length} tokens, limit {MAX_PROMPT_TOKENS})"
                )
            encoded = encoded.to(self._model.device)
            with torch.no_grad():
                output = self._model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tok.eos_token_id,
                    stopping_criteria=_loop_stopper(prompt_length),
                )
            raw = self._tok.decode(output[0][prompt_length:], skip_special_tokens=True).strip()
            text, flags = clean_note(raw)
            self.last_flags = flags
            return text, flags


note_generator = NoteGenerator()
