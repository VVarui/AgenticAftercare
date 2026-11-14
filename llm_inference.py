"""
AI Aftercare: Llama-2 LLM Inference Layer
Real-time conversational response generation with RAG context
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from typing import Optional, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Llama2Inference:
    """
    Llama-2 inference wrapper for AI Aftercare.
    Generates conversational responses with clinical context.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False,
    ):
        """
        Initialize Llama-2 model.
        
        
        """
        self.device = device
        self.model_name = model_name
        
        logger.info(f"Loading {model_name} on {device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if specified
        model_kwargs = {
            "device_map": device,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }
        
        if load_in_8bit and device == "cuda":
            model_kwargs["load_in_8bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        self.model.eval()
        logger.info(f"✓ Model loaded: {model_name}")
    
    def _build_prompt(
        self,
        patient_message: str,
        symptoms_summary: str,
        triage_level: str,
        patient_history: Dict,
        rag_context: Optional[str] = None,
    ) -> str:
        """
        Build clinical prompt with few-shot examples from MTS-Dialog.
        
        """
        
        system_prompt = """You are an AI nurse assistant for aftercare patient check-ins. 

GUIDELINES:
- Be empathetic and supportive
- Ask clarifying questions about symptoms
- Acknowledge patient concerns
- Do not provide diagnosis, just an assesment
- Do not prescribe medications
- Recommend nurse review for urgent cases
- Keep responses clear
- Always prioritize patient safety

PATIENT CONTEXT:
- Age: {age}
- Comorbidities: {comorbidities}
- Current Meds: {meds}

TRIAGE LEVEL: {triage_level}
SYMPTOMS: {symptoms}

{rag_context_str}

PATIENT MESSAGE: {message}

RESPONSE:"""
        
        rag_context_str = f"CLINICAL CONTEXT:\n{rag_context}" if rag_context else ""
        
        prompt = system_prompt.format(
            age=patient_history.get("age", "unknown"),
            comorbidities=", ".join(patient_history.get("comorbidities", [])) or "None",
            meds=", ".join(patient_history.get("current_meds", [])) or "None",
            triage_level=triage_level,
            symptoms=symptoms_summary,
            rag_context_str=rag_context_str,
            message=patient_message,
        )
        
        return prompt
    
    def generate_response(
        self,
        patient_message: str,
        symptoms_summary: str,
        triage_level: str,
        patient_history: Dict,
        rag_context: Optional[str] = None,
        max_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate LLM response to patient message.
        
        Returns:
            Generated response text
        """
        
        prompt = self._build_prompt(
            patient_message,
            symptoms_summary,
            triage_level,
            patient_history,
            rag_context,
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated part (after prompt)
        response = full_response[len(prompt):].strip()
        
        return response
    
    def generate_response_streaming(
        self,
        patient_message: str,
        symptoms_summary: str,
        triage_level: str,
        patient_history: Dict,
        rag_context: Optional[str] = None,
        max_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Stream response tokens in real-time.
        
        Yields:
            Token strings as they're generated
        """
        
        prompt = self._build_prompt(
            patient_message,
            symptoms_summary,
            triage_level,
            patient_history,
            rag_context,
        )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        # Use TextIteratorStreamer for token-by-token output
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
        )
        
        # Generate in thread
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens
        for token in streamer:
            if token and token != prompt:
                yield token
    
    def generate_summary(
        self,
        check_in_text: str,
        patient_history: Dict,
    ) -> str:
        """
        Generate clinical summary of check-in for nurse notes.
        """
        
        summary_prompt = f"""Summarize this patient check-in as a brief clinical note.

PATIENT: {patient_history.get('patient_id', 'N/A')} ({patient_history.get('age', '?')}yo)
HISTORY: {', '.join(patient_history.get('comorbidities', []))}

CHECK-IN:
{check_in_text}

SUMMARY (2-3 sentences):"""
        
        inputs = self.tokenizer(
            summary_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.5,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = summary[len(summary_prompt):].strip()
        
        return summary


class RAGContextBuilder:
    """
    Builds RAG context from clinical guidelines and patient history.
    """
    
    def __init__(self):
        # Clinical guidelines (in production, load from vector DB)
        self.guidelines = {
            "pneumonia": """
PNEUMONIA MANAGEMENT:
- Symptoms: Fever + cough + dyspnea
- Red flags: SpO2 < 90%, respiratory rate > 30
- Recommendation: Consider CXR, may need antibiotics
- Escalation: Any patient with dyspnea warrants clinical evaluation
            """,
            "COPD_exacerbation": """
COPD EXACERBATION:
- Symptoms: Increased dyspnea, cough, sputum change
- Red flags: SpO2 < 90%, severe distress, altered mental status
- Management: Bronchodilators, steroids, oxygen
- Escalation: Moderate/severe exacerbation = ED referral
            """,
            "chest_pain": """
CHEST PAIN ASSESSMENT:
- Red flags: Severe pain, dyspnea, diaphoresis, radiation to arm/jaw
- Vitals concern: Tachycardia, hypertension, tachypnea
- Escalation: Any atypical chest pain with risk factors = ECG + troponin
            """,
        }
    
    def build_context(
        self,
        symptoms: List[str],
        patient_history: Dict,
    ) -> str:
        """
        Build relevant clinical context based on symptoms.
        """
        
        context_parts = []
        
        # Match symptoms to guidelines
        for symptom in symptoms:
            if "pneumonia" in symptom.lower() or ("fever" in symptom.lower() and "cough" in symptom.lower()):
                context_parts.append(self.guidelines["pneumonia"])
            elif "dyspnea" in symptom.lower() or "copd" in str(patient_history.get("comorbidities", "")).lower():
                context_parts.append(self.guidelines["COPD_exacerbation"])
            elif "chest_pain" in symptom.lower():
                context_parts.append(self.guidelines["chest_pain"])
        
        return "\n".join(context_parts) if context_parts else ""


# Example usage
if __name__ == "__main__":
    import sys
    
    # Initialize model
    try:
        llm = Llama2Inference()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Install: huggingface-cli login && huggingface-cli download meta-llama/Llama-2-7b-chat-hf")
        sys.exit(1)
    
    # Example check-in
    patient_msg = "I've had a cough for 3 days and some chest pain"
    symptoms = "cough (3 days), chest pain"
    triage = "URGENT"
    history = {
        "patient_id": "P001",
        "age": 62,
        "comorbidities": ["COPD", "HTN"],
        "current_meds": ["Albuterol", "Lisinopril"],
    }
    
    # Build RAG context
    rag_builder = RAGContextBuilder()
    context = rag_builder.build_context(["cough", "chest_pain"], history)
    
    # Generate response
    print("\n=== LLM RESPONSE ===")
    response = llm.generate_response(
        patient_msg,
        symptoms,
        triage,
        history,
        context,
    )
    print(response)
    
    # Generate summary
    print("\n=== CLINICAL SUMMARY ===")
    summary = llm.generate_summary(patient_msg, history)
    print(summary)
