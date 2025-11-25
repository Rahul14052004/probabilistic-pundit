# backend/app/agents/meta_agent.py
from typing import Any, Dict, List, Optional
import json
import os
from loguru import logger
from ..llm_client import call_llm

# BEST MODEL FOR STRICT JSON (never wraps in markdown)
META_DEFAULT_MODEL = os.getenv("META_DEFAULT_MODEL", "llama-3.1-70b-versatile")
META_MAX_TOKENS = int(os.getenv("META_MAX_TOKENS", "3000"))  # Safe limit

PLACEHOLDER_SYSTEM = """
You are the Meta FPL Selector. Combine expert probability distributions to build a valid 15-player FPL squad
(2 GK, 5 DEF, 5 MID, 3 FWD), respecting budget and max 3 players per club.

Return ONLY a valid JSON object with these exact keys:
- selected: list of 15 player objects (from the input)
- bench: list of up to 4 players (optional, can be empty)
- justification: object with "overall" key explaining your logic
- constraints_violated: empty list if valid

DO NOT add any explanation, markdown, or text outside the JSON.
"""

def deterministic_fallback(compact_candidates: List[Dict[str, Any]], budget: float) -> Dict[str, Any]:
    scored = []
    for c in compact_candidates:
        probs = c.get("probs", {})
        score = float(probs.get("Tickers", 0.0)) + float(probs.get("Haulers", 0.0))
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [c for _, c in scored[:15]]
    return {
        "selected": selected,
        "bench": [],
        "justification": {"overall": "Deterministic fallback — top 15 by Tickers + Haulers"},
        "constraints_violated": []
    }

def _validate_team(selected: List[Dict[str, Any]], budget: float, max_per_club: int = 3) -> List[str]:
    violations: List[str] = []
    if not isinstance(selected, list):
        violations.append("selected_not_list")
        return violations
    if len(selected) != 15:
        violations.append(f"selected_count={len(selected)} != 15")
    total_price = sum(float(p.get("price", 0.0) or 0.0) for p in selected)
    if total_price > budget + 1e-6:
        violations.append(f"budget_exceeded={total_price:.1f} > {budget}")
    pos_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    club_counts: Dict[str, int] = {}
    for p in selected:
        pos = p.get("position")
        if pos in pos_counts:
            pos_counts[pos] += 1
        club = p.get("club")
        if club:
            club_counts[club] = club_counts.get(club, 0) + 1
    for pos, req in [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
        if pos_counts.get(pos, 0) != req:
            violations.append(f"pos_{pos}={pos_counts.get(pos,0)} != {req}")
    for club, cnt in club_counts.items():
        if cnt > max_per_club:
            violations.append(f"club_{club}={cnt} > {max_per_club}")
    return violations

class MetaAgent:
    def __init__(self, name: str = "meta_agent", model: Optional[str] = None):
        self.name = name
        self.model = model or META_DEFAULT_MODEL

    async def synthesize(self, expert_outputs: List[Dict[str, Any]], request: Dict[str, Any]) -> Dict[str, Any]:
        budget = float(request.get("budget", 100.0) or 100.0)
        max_per_club = int(request.get("max_per_club", 3) or 3)

        # Aggregate expert probabilities
        agg: Dict[int, Dict[str, Any]] = {}
        for out in expert_outputs:
            for r in out.get("recommendations", []):
                pid = int(r.get("player_id", -1))
                if pid not in agg:
                    agg[pid] = {"player_id": pid, "probs_list": []}
                agg[pid]["probs_list"].append(r.get("probs", {}))

        details_map = {}
        for c in request.get("candidates", []) or []:
            pid = int(c.get("player_id", c.get("id", -1)))
            details_map[pid] = c

        compact_candidates = []
        for pid, entry in agg.items():
            lists = entry.get("probs_list", [])
            keys = ["Zeros", "Blanks", "Tickers", "Haulers"]
            avg = {k: sum(float(d.get(k, 0.0) or 0.0) for d in lists) / max(1, len(lists)) for k in keys}
            total = sum(avg.values()) or 1.0
            avg = {k: v / total for k, v in avg.items()}
            info = details_map.get(pid, {})
            compact_candidates.append({
                "player_id": pid,
                "name": info.get("name"),
                "position": info.get("position"),
                "club": info.get("club"),
                "price": info.get("price"),
                "ev": info.get("ev"),
                "std": info.get("std"),
                "probs": avg
            })

        user_prompt = "Aggregated candidates: " + json.dumps(compact_candidates, ensure_ascii=False)

        try:
          
            resp = await call_llm(
                PLACEHOLDER_SYSTEM,
                user_prompt,
                model=self.model,
                temperature=0.0,
                max_tokens=META_MAX_TOKENS
            )
            print(resp)
            raw = resp.get("text", "")
            print("\n" + "█" * 90)
            print("META LLM RAW RESPONSE (model =", self.model, ")")
            print("█" * 90)
            print(raw)
            print("█" * 90 + "\n")

            # Clean markdown/code blocks
            json_str = raw.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json", 1)[1].split("```", 1)[0]
            elif json_str.startswith("```"):
                json_str = json_str.split("```", 1)[1].rsplit("```", 1)[0]
            json_str = json_str.strip()

            print("CLEANED JSON → attempting parse...")
            parsed = json.loads(json_str)

            selected = parsed.get("selected", [])
            if not isinstance(selected, list):
                raise ValueError("No valid 'selected' list")

            print(selected)
            violations = _validate_team(selected, budget, max_per_club)
            parsed["constraints_violated"] = violations

            if violations:
                logger.warning(f"Meta LLM gave valid JSON but violations: {violations}")
                fallback = deterministic_fallback(compact_candidates, budget)
                fallback["constraints_violated"] = violations
                return fallback

            logger.info("META LLM SUCCESS — PERFECT AI TEAM SELECTED!")
            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"JSON DECODE FAILED: {e}\nRaw:\n{r}")
        except Exception as e:
            logger.error(f"META AGENT FAILED: {e}\nRaw:\n{raw}")

        logger.warning("Using deterministic fallback")
        return deterministic_fallback(compact_candidates, budget)