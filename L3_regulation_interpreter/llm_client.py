"""
Minimal LLM API client for L3 reasoning (Gemini API + Ollama fallback).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List
from urllib import request
from urllib.error import HTTPError

from dotenv import load_dotenv
load_dotenv()


def is_llm_configured() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def _extract_text_from_gemini_response(payload: Dict[str, Any]) -> str:
    try:
        candidates = payload.get("candidates") or []
        if not candidates:
            raise ValueError("No candidates returned from Gemini API.")
        parts = candidates[0].get("content", {}).get("parts", [])
        text_content = "".join(part.get("text", "") for part in parts).strip()
        
        # Clean up markdown code blocks if the model wrapped the JSON
        if text_content.startswith("```"):
            lines = text_content.splitlines()
            valid_lines = [line for line in lines if not line.strip().startswith("```")]
            text_content = "\n".join(valid_lines).strip()
            
        return text_content
    except Exception as e:
        raise ValueError(f"Unable to extract text content from Gemini API response: {e}")


def _extract_text_from_ollama_response(payload: Dict[str, Any]) -> str:
    message = payload.get("message") or {}
    content = message.get("content", "")
    text_content = str(content).strip()
    
    if text_content.startswith("```"):
        lines = text_content.splitlines()
        valid_lines = [line for line in lines if not line.strip().startswith("```")]
        text_content = "\n".join(valid_lines).strip()
        
    return text_content


def _call_gemini(
    api_key: str,
    chosen_model: str,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    """Attempt Gemini API call with exponential backoff on 429 errors."""
    max_retries = 3
    delays = [2, 4, 8]

    # Gemini REST API Format
    # https://ai.google.dev/api/rest/v1beta/models/generateContent
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{chosen_model}:generateContent?key={api_key}"
    
    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [{
            "parts": [{"text": user_prompt}]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.0
        }
    }

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with request.urlopen(req, timeout=120) as response:
                raw_payload = json.loads(response.read().decode("utf-8"))
            return json.loads(_extract_text_from_gemini_response(raw_payload))
            
        except HTTPError as e:
            last_error = e
            if e.code in (429, 503) and attempt < max_retries:
                delay = delays[attempt - 1]
                print(f"L3: Gemini API attempt {attempt}/{max_retries} failed ({e.code}). Retrying in {delay}s...")
                time.sleep(delay)
            else:
                if e.code == 429:
                    print(
                        f"L3: Gemini API attempt {attempt}/{max_retries} failed (429).\n"
                        f"    [TIP] You have hit the Gemini API rate limit.\n"
                    )
                raise
        except Exception:
            raise

    raise last_error  # type: ignore[misc]


def _call_ollama(
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    """Fallback to local Ollama server."""
    ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "phi4:latest")

    print(
        f"L3: Primary API call failed or unavailable. "
        f"Falling back to Ollama ({ollama_model})..."
    )

    augmented_system = system_prompt + "\n\nYou must return ONLY valid JSON."

    payload = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": augmented_system},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "stream": False
    }

    req = request.Request(
        f"{ollama_base.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with request.urlopen(req, timeout=300) as response:
        raw_payload = json.loads(response.read().decode("utf-8"))

    raw_text = _extract_text_from_ollama_response(raw_payload)
    return _parse_json_object(raw_text)


def _parse_json_object(raw_text: str) -> Dict[str, Any]:
    """Parse the model's JSON object, tolerating trailing text after it.

    Local models (phi4 observed doing this in practice) sometimes emit a
    complete, valid JSON object followed by extra commentary/whitespace --
    strict json.loads() rejects the whole response as "Extra data" in that
    case, which used to silently degrade the entire L3 analysis to an empty
    error dict with final_score missing (confidence ends up None downstream,
    which is worse than a slightly-noisy real answer). This finds the first
    balanced top-level {...} object and parses just that.
    """
    try:
        return json.loads(raw_text)
    except (json.JSONDecodeError, ValueError):
        pass

    start = raw_text.find("{")
    if start == -1:
        print("L3: Ollama response had no JSON object at all.")
        return {"raw_response": raw_text, "error": "ollama_json_parse_failed"}

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw_text)):
        ch = raw_text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw_text[start:i + 1]
                try:
                    print("L3: Ollama response had trailing extra data; parsed the leading JSON object instead.")
                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError) as exc:
                    print(f"L3: Ollama response was not valid JSON even after trimming: {exc}")
                    return {"raw_response": raw_text, "error": "ollama_json_parse_failed"}

    print("L3: Ollama response's JSON object was never closed.")
    return {"raw_response": raw_text, "error": "ollama_json_parse_failed"}


def generate_ollama_embedding(text: str) -> List[float]:
    """Generates an embedding vector using local Ollama model."""
    ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = "nomic-embed-text"
    
    import re
    # Strip corrupted PDF unicode (like \ufffd) and non-ASCII characters (like Hindi headers)
    # which cause the Ollama nomic-embed-text tokenizer to throw an HTTP 500.
    clean_text = text.replace('\ufffd', ' ')
    clean_text = re.sub(r'[^\x00-\x7F]+', ' ', clean_text)
    
    payload = {
        "model": model,
        "prompt": clean_text
    }
    
    try:
        req = request.Request(
            f"{ollama_base.rstrip('/')}/api/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=60) as response:
            raw_payload = json.loads(response.read().decode("utf-8"))
            return raw_payload.get("embedding", [])
    except Exception as exc:
        print(f"L3: Ollama embedding failed: {exc}")
        return []


def chat_json(
    system_prompt: str, user_prompt: str, model: str | None = None
) -> Dict[str, Any]:
    """
    Track 02 retrofit: local Ollama (phi4, per OLLAMA_MODEL) is now PRIMARY,
    matching the original project's design intent -- no dependency on a
    cloud key or its rate limits. Gemini is opt-in only, via
    L3_USE_GEMINI=true, for anyone who wants to compare against it.
    """
    if os.environ.get("L3_USE_GEMINI", "false").lower() == "true":
        api_key = os.environ.get("GEMINI_API_KEY")
        chosen_model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        if api_key:
            try:
                return _call_gemini(api_key, chosen_model, system_prompt, user_prompt)
            except Exception as e:
                print(f"L3: Gemini call failed: {e}")
        else:
            print("L3: L3_USE_GEMINI=true but GEMINI_API_KEY is not set. Using Ollama.")

    return _call_ollama(system_prompt, user_prompt)
