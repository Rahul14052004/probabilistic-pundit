# backend/app/agents/expert_agent.py

from typing import Any, Dict, List
import json
import os
from loguru import logger
from ..llm_client import call_llm

CHUNK_SIZE = int(os.getenv("EXPERT_CHUNK_SIZE", "25"))
EXPERT_DEFAULT_MODEL = os.getenv("EXPERT_DEFAULT_MODEL", "llama-3.1-8b-instant")
MAX_TOKENS_PER_PLAYER = 70  

# Persona prompts (justification added for Safe Bet + Differentials)
PERSONA_PROMPTS = {
    "value_hunter": """You are the 'Value Hunter,' an FPL analyst specializing in underpriced high-PPM players.

Analyze candidates using fields like:
value, xP, expected_goal_involvements, threat, minutes, fixture difficulty, bps.

Assign probabilities for each of the 4 outcomes:
Zeros (0 pts), Blanks (1–2 pts), Tickers (3–7 pts), Haulers (8+ pts)

For EACH player, include a SHORT justification (max 15–20 words).

Return ONLY:
[
  {
    "player_id": 123,
    "probs": {"Zeros":0.2,"Blanks":0.3,"Tickers":0.3,"Haulers":0.2},
    "justification": "high xGI + great fixture + underpriced"
  }
]
""",

    "safe_bet": """You are the 'Safe Bet,' an FPL analyst focused on consistency and reliability.

Use stability signals:
minutes, starts, ICT index, influence, xP, bps, team form, fixture.

Include a short justification for each player.

Return ONLY:
[
  {
    "player_id": 123,
    "probs": {"Zeros":0.25,"Blanks":0.25,"Tickers":0.25,"Haulers":0.25},
    "justification": "nailed starter, high influence, safe floor"
  }
]
""",

    "differentials_specialist": """You are the 'Differentials Specialist,' targeting low-owned explosive players.

Use fields like:
selected_by_percentage, expected_goal_involvements, threat, recent form, minutes, fixture swings.

Include a short justification for each player.

Return ONLY:
[
  {
    "player_id": 123,
    "probs": {"Zeros":0.1,"Blanks":0.2,"Tickers":0.4,"Haulers":0.3},
    "justification": "low ownership, rising form, strong xGI"
  }
]
""",
}

NEUTRAL_PROBS = {"Zeros": 0.25, "Blanks": 0.25, "Tickers": 0.25, "Haulers": 0.25}

class ExpertAgent:
    def __init__(self, name: str, persona: str = "value_hunter", model: str | None = None):
        self.name = name
        if persona not in PERSONA_PROMPTS:
            logger.warning(f"Unknown persona '{persona}', defaulting to value_hunter")
            persona = "value_hunter"

        self.persona = persona
        self.system_prompt = PERSONA_PROMPTS[persona]
        self.model = model or EXPERT_DEFAULT_MODEL

    async def analyze(self, candidates: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:

        results: List[Dict[str, Any]] = []

        for i in range(0, len(candidates), CHUNK_SIZE):
            chunk = candidates[i:i + CHUNK_SIZE]

            compact = [
                {
                    "player_id": int(c.get("player_id") or c.get("id")),
                    "name": c.get("name"),
                    "position": c.get("position"),
                    "team": c.get("team"),
                    "value": c.get("value"),
                    "xP": c.get("xP"),
                    "xGI": c.get("expected_goal_involvements"),
                    "minutes": c.get("minutes"),
                    "fixture": c.get("fixture"),
                    "selected_by": c.get("selected"),
                    "threat": c.get("threat"),
                    "ict": c.get("ict_index"),
                }
                for c in chunk
            ]

            user_prompt = "Candidates:\n" + json.dumps(compact, ensure_ascii=False)
            max_tokens = int(len(chunk) * MAX_TOKENS_PER_PLAYER * 1.1)
            try:
                resp = await call_llm(
                    self.system_prompt,
                    user_prompt,
                    model=self.model,
                    temperature=0,
                    max_tokens=max_tokens,
                )

                text = (resp.get("text") or "").strip()
                parsed = json.loads(text)

                if not isinstance(parsed, list):
                    raise ValueError("LLM must return a list")

                for entry in parsed:
                    pid = int(entry.get("player_id", -1))
                    raw_probs = entry.get("probs", {})
                    justification = entry.get("justification", "").strip()

                    # Validate & normalize probs
                    if not isinstance(raw_probs, dict):
                        probs = NEUTRAL_PROBS.copy()
                    else:
                        s = sum(float(v) for v in raw_probs.values())
                        if s <= 0:
                            probs = NEUTRAL_PROBS.copy()
                        else:
                            probs = {k: float(v) / s for k, v in raw_probs.items()}

                    if not justification:
                        justification = "No justification provided."

                    results.append({
                        "player_id": pid,
                        "probs": probs,
                        "justification": justification,
                    })

            except Exception as e:
                logger.exception(f"ExpertAgent {self.name} failed for chunk: {e}")
                for c in chunk:
                    pid = int(c.get("player_id") or c.get("id"))
                    results.append({
                        "player_id": pid,
                        "probs": NEUTRAL_PROBS.copy(),
                        "justification": "Fallback due to error."
                    })

        return {
            "agent": self.name,
            "persona": self.persona,
            "recommendations": results,
        }
