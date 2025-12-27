import torch
import threading
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/gemma-2b-it"
_model_lock = threading.Lock()
_tokenizer = None
_model = None

# LOAD MODEL (lazy-load)
def load_model():
    global _tokenizer, _model

    with _model_lock:
        if _tokenizer is None or _model is None:
            print("[Summarizer] Loading Gemma-2B-IT model...")
            try:
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                _model = AutoModelForSeq2SeqLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float32,   # CPU-friendly
                    device_map="auto",           # Use GPU if available
                )
                _model.eval()
                print("[Summarizer] Gemma-2B-IT loaded successfully.")
            except Exception as e:
                print("[Summarizer] ERROR loading Gemma-2B-IT:", e)
                raise e

    return _tokenizer, _model

# BUILD PROMPT
def build_prompt(text: str) -> str:
    return f"""
You are a legal AI assistant.

Summarize the following legal document focusing on:
- Main purpose of the agreement
- Involved parties
- Obligations and responsibilities
- Risks / penalties
- Termination conditions
- Important dates

The summary must:
- Be accurate and legally precise
- Use full sentences
- Avoid hallucinations
- Only reflect the provided text

DOCUMENT:
{text}

SUMMARIZE BELOW:
"""

# SUMMARY FUNCTION
def generate_summary(text: str, max_tokens: int = 256) -> str:
    tokenizer, model = load_model()

    prompt = build_prompt(text)

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.2,
                top_p=0.95,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
            )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return summary

    except Exception as e:
        print("[Summarizer] Error during generation:", e)
        return "Summary unavailable due to an internal error."



