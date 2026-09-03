import json
import os
import requests

from dotenv import load_dotenv
load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3-32b")  # confirm/override via .env
OLLAMA_FALLBACK_MODEL = os.environ.get("MAKER_CHECKER_OLLAMA_MODEL", "phi4:latest")


def _build_prompt(tx, l3_analysis):
    verdict = l3_analysis.get("verdict", "unknown")
    confidence = l3_analysis.get("confidence", 0.0)
    citation = l3_analysis.get("citation", "No citation provided.")
    explanation = l3_analysis.get("explanation", "No explanation provided.")

    return f"""
    You are an independent second-opinion reviewer (Maker-Checker) for a
    merchant chargeback-response pipeline. A primary model has already
    analyzed a disputed transaction and recommended a response, citing
    supporting guidance.

    TRANSACTION FACTS:
    {json.dumps(tx, indent=2)}

    PRIMARY MODEL VERDICT: {verdict}
    PRIMARY MODEL CONFIDENCE: {confidence}
    CITED GUIDANCE: {citation}
    PRIMARY MODEL REASONING: {explanation}

    YOUR TASK:
    Write a short, clear paragraph (3-4 sentences maximum) checking this
    recommendation against the transaction facts and explaining, in plain
    language, whether the evidence actually supports it -- e.g. does the
    delivery status match what the recommendation assumes.
    - Avoid technical jargon (no internal codes like C1-C6, L1-L4).
    - Write it so a dispute-ops analyst can immediately understand the core issue.
    - Do NOT output any JSON, just the paragraph text.
    """


def _call_groq(prompt, model):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        },
        timeout=45,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_ollama(prompt, model):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
    }, timeout=90)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def run_maker_checker(tx, l3_analysis, model=None):
    """
    Independent second-opinion agent that validates L3's primary recommendation
    and explains it in plain language. Tries Groq (fast, cloud) first if
    GROQ_API_KEY is set, falls back to local Ollama, falls back to an honest
    offline message -- never blocks the rest of the pipeline.
    """
    prompt = _build_prompt(tx, l3_analysis)

    groq_model = model or GROQ_MODEL
    try:
        print(f"Maker-Checker: Connecting to Groq (model={groq_model})...")
        result = _call_groq(prompt, groq_model)
        if result is not None:
            print("Maker-Checker: Successfully validated transaction (Groq).")
            return result
        print("Maker-Checker: GROQ_API_KEY not set, trying local Ollama...")
    except requests.exceptions.RequestException as e:
        print(f"Maker-Checker: Groq call failed ({e}), trying local Ollama...")

    try:
        print(f"Maker-Checker: Connecting to Ollama (model={OLLAMA_FALLBACK_MODEL})...")
        result = _call_ollama(prompt, OLLAMA_FALLBACK_MODEL)
        print("Maker-Checker: Successfully validated transaction (Ollama).")
        return result
    except requests.exceptions.RequestException as e:
        print(f"Maker-Checker API error: {e}")
        return "Maker-Checker validation is currently offline (neither Groq nor local Ollama reachable)."
    except Exception as e:
        print(f"Maker-Checker unexpected error: {e}")
        return "Maker-Checker validation encountered an unexpected error."
