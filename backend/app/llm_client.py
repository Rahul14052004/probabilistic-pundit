# backend/app/llm_client.py

import os
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
import streamlit as st
import httpx
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from dotenv import load_dotenv
load_dotenv()
import copy

# ======================================================================
# LOGGER
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

LLM_USE_MOCK = os.getenv("LLM_USE_MOCK", "0") in ("1", "true", "True")

LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "8"))
HTTP_TIMEOUT = int(os.getenv("LLM_HTTP_TIMEOUT_SECS", "60"))
RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", "2"))

PRIMARY_MODEL = os.getenv("EXPERT_DEFAULT_MODEL", "llama-3.1-8b-instant")
META_MODEL = os.getenv("META_DEFAULT_MODEL", "llama-3.1-8b-instant")

FALLBACK_MODELS = [
    os.getenv("FALLBACK_MODEL_1", "gemma-7b-it"),
    os.getenv("FALLBACK_MODEL_2", "llama-3.1-70b-versatile"),
    os.getenv("FALLBACK_MODEL_3", "mixtral-8x7b-32768"),
]

# ---------------- GROQ KEYS ----------------
API_KEYS = [
    os.getenv("GROQ_API_KEY", ""),
    os.getenv("GROQ_API_KEY_2", "")
]
API_KEYS = [k for k in API_KEYS if k.strip()]

if not API_KEYS:
    raise RuntimeError("No GROQ_API_KEY provided in .env")

# ======================================================================
# VALID GROQ MODELS (Nov 2025)
# ======================================================================
GROQ_MODELS = {
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "gemma-7b-it"
}

# ======================================================================
# CONTROL
# ======================================================================

_llm_semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENCY)

def _approx_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


# ======================================================================
# HTTP POST
# ======================================================================

async def _http_post(payload: Dict[str, Any], model: str, api_key: str) -> Dict[str, Any]:
    if model not in GROQ_MODELS:
        raise RuntimeError(f"Model '{model}' not supported by Groq.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()


# ======================================================================
# MASTER LLM CALL WITH FALLBACK + KEY ROTATION
# ======================================================================

async def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    stop: Optional[List[str]] = None
) -> Dict[str, Any]:

    model = model or PRIMARY_MODEL
    prompt_tokens = _approx_tokens(system_prompt + user_prompt)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": 1,
    }
    if stop:
        payload["stop"] = stop

    # ==========================================================
    # TRY PRIMARY MODEL WITH ALL API KEYS
    # ==========================================================
    for key in API_KEYS:
        try:
            async with _llm_semaphore:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(RETRY_ATTEMPTS),
                    wait=wait_exponential(multiplier=0.5),
                    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
                    reraise=True,
                ):
                    with attempt:
                        logger.info(f"[LLM] PRIMARY MODEL = {model}")
                        resp_json = await _http_post(payload, model, key)

                        text = resp_json["choices"][0]["message"]["content"]
                        meta = {
                            "model": model,
                            "prompt_tokens_approx": prompt_tokens,
                            "response_tokens_approx": _approx_tokens(text),
                            "usage": resp_json.get("usage", {}),
                        }

                        logger.info(f"LLM SUCCESS → model={model}")
                        return {"text": text.strip(), "meta": meta}

        except Exception as e:
            logger.warning(f"[LLM] Rate limit or HTTP error on primary → rotating key: {e}")

        await asyncio.sleep(2.5)

    # ==========================================================
    # FALLBACK MODELS
    # ==========================================================
    for fb_model in FALLBACK_MODELS:
        for key in API_KEYS:
            try:
                logger.warning(f"[LLM] FALLBACK MODEL → {fb_model}")

                fb_payload = copy.deepcopy(payload)
                fb_payload["model"] = fb_model

                resp_json = await _http_post(fb_payload, fb_model, key)
                text = resp_json["choices"][0]["message"]["content"]

                meta = {
                    "model": fb_model,
                    "prompt_tokens_approx": prompt_tokens,
                    "response_tokens_approx": _approx_tokens(text),
                    "usage": resp_json.get("usage", {}),
                }

                return {"text": text.strip(), "meta": meta}

            except Exception as e:
                logger.error(f"[LLM] Fallback error ({fb_model}): {e}")

            await asyncio.sleep(2)

    # ==========================================================
    # COMPLETELY FAILED
    # ==========================================================
    raise RuntimeError("❌ ALL LLM MODELS FAILED EVEN AFTER FINAL RETRY")
