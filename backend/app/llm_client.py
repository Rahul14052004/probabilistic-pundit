# backend/app/llm_client.py
import os
import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from dotenv import load_dotenv
load_dotenv()

# ======================================================================
# LOGGING SETUP
# ======================================================================

logger = logging.getLogger("llm_client")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
logger.setLevel(os.getenv("LLM_CLIENT_LOGLEVEL", "INFO"))

# ======================================================================
# CONFIG
# ======================================================================

LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "8"))
HTTP_TIMEOUT = int(os.getenv("LLM_HTTP_TIMEOUT_SECS", "60"))
RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", "2"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
FALLBACK_SLEEP = int(os.getenv("LLM_FALLBACK_SLEEP", "3"))  # seconds

DEFAULT_EXPERT_MODEL = os.getenv("EXPERT_DEFAULT_MODEL", "llama-3.1-8b-instant")
DEFAULT_META_MODEL   = os.getenv("META_DEFAULT_MODEL", "mixtral-8x7b-32768")

# ======================================================================
# ALL MODELS SUPPORTED BY GROQ (UPDATED)
# ======================================================================

PRIMARY_MODELS = {
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

# FALLBACK MODELS (ON GROQ)
FALLBACK_MODELS = [
    "moonshotai/kimi-k2-instruct",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
]

ALL_GROQ_MODELS = PRIMARY_MODELS.union(FALLBACK_MODELS)

_llm_semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENCY)

# ======================================================================
# UTILS
# ======================================================================

def _approx_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)

async def _post_to_groq(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Low-level Groq HTTP call."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in .env")

    normalized_model = model.split("/", 1)[-1] if "/" in model else model

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = dict(payload)
    payload["model"] = normalized_model

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(url, json=payload, headers=headers)

        if r.status_code == 429:
            raise httpx.HTTPStatusError("Rate limit hit", request=r.request, response=r)

        r.raise_for_status()
        return r.json()

# ======================================================================
# MAIN DISPATCH LLM CALL
# ======================================================================

async def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    stop: Optional[List[str]] = None,
) -> Dict[str, Any]:

    model = model or DEFAULT_EXPERT_MODEL

    if model not in ALL_GROQ_MODELS:
        raise RuntimeError(f"Invalid model: {model}")

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
        # FIRST ATTEMPT
        try:
            logger.info(f"[LLM] PRIMARY MODEL = {model}")
            resp_json = await _post_to_groq(payload, model)
            return _extract_llm_response(resp_json, model, prompt_tokens)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning(f"[LLM] Rate limit for: {model}. Cooling {FALLBACK_SLEEP}s…")
                await asyncio.sleep(FALLBACK_SLEEP)
            else:
                logger.error(f"[LLM] HTTP error: {e}")

        except Exception as e:
            logger.error(f"[LLM] Primary model failed: {model}. Error: {e}")

        # FALLBACK MODELS
        for fb_model in FALLBACK_MODELS:
            try:
                logger.warning(f"[LLM] FALLBACK MODEL → {fb_model}")
                resp_json = await _post_to_groq(payload, fb_model)
                return _extract_llm_response(resp_json, fb_model, prompt_tokens)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning(f"[LLM] Fallback {fb_model} rate-limited. Sleeping {FALLBACK_SLEEP}s…")
                    await asyncio.sleep(FALLBACK_SLEEP)
                else:
                    logger.error(f"[LLM] Fallback HTTP error: {fb_model} | {e}")

            except Exception as e:
                logger.error(f"[LLM] Fallback model failed: {fb_model} | {e}")

        # FINAL RETRY LOOP
        logger.error("[LLM] ALL MODELS FAILED → last chance retry")
        await asyncio.sleep(FALLBACK_SLEEP)

        try:
            resp_json = await _post_to_groq(payload, FALLBACK_MODELS[-1])
            return _extract_llm_response(resp_json, FALLBACK_MODELS[-1], prompt_tokens)
        except Exception as e:
            raise RuntimeError(f"❌ ALL LLM MODELS FAILED EVEN AFTER FINAL RETRY | {e}")

# ======================================================================
# RESPONSE EXTRACTOR
# ======================================================================

def _extract_llm_response(resp_json: Dict[str, Any], model: str, prompt_tokens: int):
    text = resp_json["choices"][0]["message"]["content"]

    meta = {
        "model": model,
        "prompt_tokens_approx": prompt_tokens,
        "response_tokens_approx": _approx_tokens(text),
        "usage": resp_json.get("usage", {})
    }

    logger.info(f"LLM SUCCESS → model={model} | tokens≈{meta['prompt_tokens_approx']} + {meta['response_tokens_approx']}")
    print("\n" + "═" * 80)
    print(f"LLM RESPONSE | model={model}")
    print("─" * 80)
    print(text.strip())
    print("═" * 80 + "\n")

    return {"text": text.strip(), "meta": meta}
