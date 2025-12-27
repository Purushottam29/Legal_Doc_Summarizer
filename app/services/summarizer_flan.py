import torch
import threading
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"
_tokenizer = None
_model = None
_lock = threading.Lock()

# MODEL LOADER
def load_model():
    global _tokenizer, _model

    with _lock:
        if _tokenizer is None or _model is None:
            print("[Summarizer] Loading FLAN-T5-BASE...")

            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

            _model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                torch_dtype=torch.float32
            )

            _model.eval()
            print("[Summarizer] FLAN-T5-BASE loaded successfully.")

    return _tokenizer, _model

# LEGAL SUMMARIZATION PROMPT
def build_summary_prompt(text: str) -> str:
    return f"""
Summarize the following legal document focusing on:

- The purpose of the agreement
- The involved parties
- Obligations and responsibilities
- Penalties or risks
- Termination conditions
- Important dates

Write a clear, factual summary with no hallucinations.

DOCUMENT:
{text}

SUMMARY:
"""

# GENERATE SUMMARY / ANSWER
def generate_summary(text: str, max_tokens: int = 256) -> str:
    """
    Used both for:
    • Legal document summarization
    • RAG answer generation (prompt already built by ask.py)
    """
    tokenizer, model = load_model()

    # Prepare input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.2,     # less hallucination
                top_p=0.95,
                num_beams=4,         # higher quality
                do_sample=False,     # deterministic output
                early_stopping=True
            )

        result = tokenizer.decode(output[0], skip_special_tokens=True)
        return result.strip()

    except Exception as e:
        print("[Summarizer ERROR]:", e)
        return "The system could not generate a response."



