# backend/app/agents/expert_agent.py
from typing import Any, Dict, List
import json
import os
from loguru import logger
from ..llm_client import call_llm

CHUNK_SIZE = int(os.getenv("EXPERT_CHUNK_SIZE", "30"))
EXPERT_DEFAULT_MODEL = os.getenv("EXPERT_DEFAULT_MODEL", "llama-3.1-8b-instant")

# Persona prompts (as requested)
PERSONA_PROMPTS = {
    "value_hunter": """You are the 'Value Hunter,' a fantasy football analyst who identifies undervalued players with high points-per-million potential.
Base recommendations on recent stats, fixture difficulty, and budget efficiency.

Return ONLY a pure JSON array like:
[{"player_id": <int>, "probs": {"Zeros":0.25,"Blanks":0.25,"Tickers":0.25,"Haulers":0.25}}, ...]
Probabilities MUST sum to 1.0. No commentary or extra text.""",

    "safe_bet": """You are the 'Safe Bet,' a fantasy football analyst who focuses on consistent, reliable returns.
Prioritize highly-owned players with favorable matchups and strong underlying metrics.

Return ONLY a pure JSON array like:
[{"player_id": <int>, "probs": {"Zeros":0.25,"Blanks":0.25,"Tickers":0.25,"Haulers":0.25}}, ...]
Probabilities MUST sum to 1.0. No commentary or extra text.""",

    "differentials_specialist": """You are the 'Differentials Specialist,' a fantasy football analyst who identifies low-owned, high-upside players likely to deliver breakout performances due to role changes, tactical shifts, or recent form.

Return ONLY a pure JSON array like:
[{"player_id": <int>, "probs": {"Zeros":0.25,"Blanks":0.25,"Tickers":0.25,"Haulers":0.25}}, ...]
Probabilities MUST sum to 1.0. No commentary or extra text.""",
}

# Default neutral distribution (fallback)
NEUTRAL_PROBS = {"Zeros": 0.25, "Blanks": 0.25, "Tickers": 0.25, "Haulers": 0.25}

class ExpertAgent:
    def __init__(self, name: str, persona: str = "value_hunter", model: str | None = None):
        self.name = name
        if persona not in PERSONA_PROMPTS:
            logger.warning("Unknown persona '%s', defaulting to value_hunter", persona)
            persona = "value_hunter"
        self.persona = persona
        self.system_prompt = PERSONA_PROMPTS[self.persona]
        self.model = model or EXPERT_DEFAULT_MODEL

    async def analyze(self, candidates: List[Dict[str, Any]], request: Dict[str, Any]) -> Dict[str, Any]:
        """
        candidates: list of dicts with at least 'player_id' or 'id' (and optional 'name')
        request: request metadata (budget, constraints, etc.)
        Returns: { "agent": <name>, "recommendations": [ {"player_id":..., "probs": {...}}, ... ], "reasoning": str }
        """
        results: List[Dict[str, Any]] = []
        # chunk candidates to avoid huge prompts
        # print(candidates)
        for i in range(0, len(candidates), CHUNK_SIZE):
            chunk = candidates[i:i + CHUNK_SIZE]
            # compact candidates for prompt
            compact = [{"player_id": int(c.get("player_id", c.get("id", -1))), "name": c.get("name", "")} for c in chunk]
            user_prompt = "Candidates: " + json.dumps(compact, ensure_ascii=False)

            try:
                resp = await call_llm(self.system_prompt, user_prompt, model=self.model, temperature=0.0, max_tokens=512)
                text = (resp.get("text") or "").strip()
                # Expecting JSON array
                parsed = json.loads(text)
                if not isinstance(parsed, list):
                    raise ValueError("Expert LLM did not return a JSON list")
                for it in parsed:
                    pid = int(it.get("player_id", -1))
                    probs = it.get("probs", {})
                    if not isinstance(probs, dict) or sum([float(v) for v in probs.values()]) <= 0:
                        probs = NEUTRAL_PROBS.copy()
                    else:
                        # normalize
                        s = sum([float(v) for v in probs.values()])
                        if s <= 0:
                            probs = NEUTRAL_PROBS.copy()
                        else:
                            probs = {k: float(v) / s for k, v in probs.items()}
                    results.append({"player_id": pid, "probs": probs})
            except Exception as e:
                # Log and fallback for this chunk
                logger.exception("ExpertAgent '%s' failed to parse LLM response; using neutral probs for chunk. err=%s", self.name, e)
                for c in chunk:
                    pid = int(c.get("player_id", c.get("id", -1)))
                    results.append({"player_id": pid, "probs": NEUTRAL_PROBS.copy()})

        return {"agent": self.name, "recommendations": results, "reasoning": f"persona={self.persona}, model={self.model}"}
