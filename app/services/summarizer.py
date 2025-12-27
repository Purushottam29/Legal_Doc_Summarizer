import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"


def query_ollama(prompt: str, max_tokens: int = 256) -> str:
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "options": {
                "temperature": 0.2,
                "num_predict": max_tokens
            }
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=120)

        if response.headers.get("Content-Type") == "application/x-ndjson":
            result_text = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode())
                    if "response" in data:
                        result_text += data["response"]
            return result_text.strip()

        data = response.json()
        return data.get("response", "").strip()

    except Exception as e:
        print("[Ollama ERROR]:", e)
        return "AI could not generate a response."


def build_summary_prompt(text: str) -> str:
    return f"""
Summarize the following legal document with focus on:
- Purpose of the agreement
- Parties involved
- Obligations and responsibilities
- Payment terms
- Risks / penalties
- Termination conditions
- Important dates

Be precise. No hallucinations.

DOCUMENT:
{text}

SUMMARY:
"""


def generate_summary(text: str, max_tokens: int = 256) -> str:
    return query_ollama(text, max_tokens=max_tokens)

