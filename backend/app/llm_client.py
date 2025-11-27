# backend/app/llm_client.py
import os
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional

import httpx
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type

from dotenv import load_dotenv
load_dotenv()

# Logger
logger = logging.getLogger("llm_client")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
logger.setLevel(os.getenv("LLM_CLIENT_LOGLEVEL", "INFO"))

# --- Config ---
LLM_USE_MOCK = os.getenv("LLM_USE_MOCK", "0") in ("1", "true", "True")
LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "8"))
HTTP_TIMEOUT = int(os.getenv("LLM_HTTP_TIMEOUT_SECS", "60"))
RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", "2"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# Default models
DEFAULT_EXPERT_MODEL = os.getenv("EXPERT_DEFAULT_MODEL", "llama-3.1-8b-instant")
DEFAULT_META_MODEL   = os.getenv("META_DEFAULT_MODEL", "mixtral-8x7b-32768")  # ← WORKS ON GROQ

# All real Groq models (Nov 2025)
KNOWN_GROQ_MODELS = {
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "gemma-7b-it",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-70b-8192",
    "llama3-8b-8192",
}

# Force chat format for all Groq models (they all support it)
CHAT_MODELS = KNOWN_GROQ_MODELS  # ← This is the key fix

# Concurrency
_llm_semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENCY)

def _approx_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)

# --- Core HTTP function ---
async def _http_post(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
    model = model or ""

    is_groq = model in KNOWN_GROQ_MODELS or model.startswith("groq/")

    if not is_groq:
        # You only want to use Groq-hosted models
        raise RuntimeError(
            f"Model '{model}' is not recognized as a Groq model. "
            f"Use one of: {', '.join(KNOWN_GROQ_MODELS)}"
        )

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in .env")

    # Normalize model name (if you ever use 'groq/llama-3.1-8b-instant' style)
    groq_model = model.split("/", 1)[-1] if "/" in model else model

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # payload already has model + messages etc from call_llm
    # ensure it uses the normalized groq_model
    payload = dict(payload)
    payload["model"] = groq_model

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()


# --- Public LLM call ---
async def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    stop: Optional[List[str]] = None,
) -> Dict[str, Any]:
    # if LLM_USE_MOCK:
    #     # Mock logic (unchanged)
    #     try:
    #         if "Aggregated candidates" in user_prompt:
    #             raw = user_prompt.split("Aggregated candidates:", 1)[-1].strip()
    #             compact = json.loads(raw)
    #             selected = [{"player_id": c.get("player_id")} for c in compact[:15]]
    #             text = json.dumps({"selected": selected, "bench": [], "justification": {"overall": "mock"}, "constraints_violated": []})
    #         else:
    #             text = json.dumps({"mock": True})
    #     except:
    #         text = json.dumps({"mock": True})

    #     return {"text": text, "meta": {"mock": True}}

    logger.info(f"REAL LLM CALL → model={model or DEFAULT_EXPERT_MODEL} | temp={temperature} | max_tokens={max_tokens}")

    model = model or DEFAULT_EXPERT_MODEL

    # Always use chat format for Groq models
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": 1,
    }
    if stop:
        payload["stop"] = stop

    prompt_tokens = _approx_tokens(system_prompt + user_prompt)

    async with _llm_semaphore:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(RETRY_ATTEMPTS),
            wait=wait_exponential(multiplier=0.5),
            retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
            reraise=True,
        ):
            with attempt:
                resp_json = await _http_post(payload, model=model)
                print(resp_json)
                # Extract text (Groq uses standard OpenAI format)
                text = resp_json["choices"][0]["message"]["content"]
                # print(text)
                meta = {
                    "model": model,
                    "prompt_tokens_approx": prompt_tokens,
                    "response_tokens_approx": _approx_tokens(text),
                    "usage": resp_json.get("usage", {})
                }
                logger.info(f"LLM SUCCESS → model={model} | tokens≈{meta['prompt_tokens_approx']}+{meta['response_tokens_approx']}")
                print("\n" + "═" * 80)
                print(f"LLM RESPONSE | model={model}")
                print("─" * 80)
                print(text.strip())
                print("═" * 80 + "\n")
                # ======================================
                return {"text": text.strip(), "meta": meta}