# backend/app/agents/meta_agent.py

from typing import Any, Dict, List, Optional
import json
import os
from loguru import logger
from ..llm_client import call_llm

META_DEFAULT_MODEL = os.getenv("META_DEFAULT_MODEL", "llama-3.1-70b-versatile")
META_MAX_TOKENS = int(os.getenv("META_MAX_TOKENS", "3500"))

# ------------------------------------------------------------------------------------
# DETERMINISTIC FALLBACK (corrected, no player_id)
# ------------------------------------------------------------------------------------

def deterministic_fallback(candidates: List[Dict[str, Any]], budget: float):
    """
    Safe fallback if LLM fails.
    Greedy selection:
      - Score = Tickers + 2*Haulers
      - Respect FPL formation
      - Stay within budget
    """

    required = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    scored = []

    for c in candidates:
        p = c["probs"]
        score = float(p["Tickers"]) + 2 * float(p["Haulers"])
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    team = []
    spent = 0.0

    for _, c in scored:
        pos = c["position"]
        price = float(c["price"])

        if required[pos] <= 0:
            continue
        if spent + price > budget:
            continue

        team.append(c)
        required[pos] -= 1
        spent += price

        if all(v == 0 for v in required.values()):
            break

    return {
        "selected": team,
        "bench": [],
        "justification": {"overall": "Fallback greedy selection"},
        "constraints_violated": [],
    }


# ------------------------------------------------------------------------------------
# CONSENSUS REMOVAL (no player_id)
# ------------------------------------------------------------------------------------

def consensus_remover(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bad = []
    for c in candidates:
        p = c["probs"]
        if p["Tickers"] <= 0.10 and p["Haulers"] <= 0.05:
            bad.append(c)

    count = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for c in candidates:
        count[c["position"]] += 1

    min_req = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

    keep = []
    for c in candidates:
        if c not in bad:
            keep.append(c)
            continue

        pos = c["position"]
        if count[pos] - 1 < min_req[pos]:
            keep.append(c)
        else:
            count[pos] -= 1

    return keep


# ------------------------------------------------------------------------------------
# CONSENSUS PICKERS (no player_id)
# ------------------------------------------------------------------------------------

def consensus_pickers(candidates: List[Dict[str, Any]], per_expert_probs: Dict[str, List[Dict[str, float]]], budget: float):
    required = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    picked = []
    remaining_budget = budget
    filtered = []

    for c in candidates:
        name = c["name"]
        probs_by_expert = per_expert_probs.get(name, [])

        high_votes = 0
        for p in probs_by_expert:
            if p.get("Tickers", 0) >= 0.70 or p.get("Haulers", 0) >= 0.70:
                high_votes += 1

        if high_votes >= 2:
            picked.append(c)
            remaining_budget -= float(c["price"])
            required[c["position"]] -= 1
        else:
            filtered.append(c)

    for pos in required:
        if required[pos] < 0:
            required[pos] = 0

    return picked, filtered, remaining_budget, required


# ------------------------------------------------------------------------------------
# TEAM VALIDATION (unchanged except no player_id)
# ------------------------------------------------------------------------------------

def _validate_team(selected: List[Dict[str, Any]], budget: float, max_per_club: int = 3):
    violations = []

    if len(selected) != 15:
        violations.append(f"selected_count={len(selected)} != 15")

    spent = sum(float(p["price"]) for p in selected)
    if spent > budget:
        violations.append(f"budget_exceeded={spent:.1f} > {budget}")

    pos_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    club_counts = {}

    for p in selected:
        pos_counts[p["position"]] += 1
        club = p["club"]
        club_counts[club] = club_counts.get(club, 0) + 1

    required = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    for pos, req in required.items():
        if pos_counts[pos] != req:
            violations.append(f"pos_{pos}={pos_counts[pos]} != {req}")

    for club, cnt in club_counts.items():
        if cnt > max_per_club:
            violations.append(f"club_{club}={cnt} > {max_per_club}")

    return violations


# ------------------------------------------------------------------------------------
# META AGENT (NO PLAYER_ID VERSION)
# ------------------------------------------------------------------------------------

class MetaAgent:
    def __init__(self, name="meta_agent", model=None):
        self.name = name
        self.model = model or META_DEFAULT_MODEL

    async def synthesize(self, expert_outputs, request):

        budget = float(request.get("budget", 100.0))
        max_per_club = int(request.get("max_per_club", 3))

        # -------------------------------------------------------------
        # 1. Aggregate expert outputs by NAME ONLY
        # -------------------------------------------------------------
        agg = {}  # key: player name
        for out in expert_outputs:
            agent_name = out["agent"]
            for r in out["recommendations"]:
                name = r["name"]  # <<—— NOTE: expert_agent must return name now

                if name not in agg:
                    agg[name] = {"probs_list": [], "justifications": []}

                agg[name]["probs_list"].append(r["probs"])
                if "justification" in r:
                    agg[name]["justifications"].append(f"{agent_name}: {r['justification']}")

        # map metadata
        meta_map = {c["name"]: c for c in request["candidates"]}

        # compute final candidate representations
        compact = []
        for name, entry in agg.items():
            meta = meta_map[name]

            avg = {}
            keys = ["Zeros", "Blanks", "Tickers", "Haulers"]
            for k in keys:
                avg[k] = sum(p[k] for p in entry["probs_list"]) / len(entry["probs_list"])
            total = sum(avg.values()) or 1.0
            avg = {k: v / total for k, v in avg.items()}

            compact.append({
                "name": name,
                "position": meta["position"],
                "club": meta["club"],
                "price": meta["price"],
                "ev": meta["ev"],
                "std": meta["std"],
                "probs": avg,
                "expert_justifications": entry["justifications"],
            })

        # -------------------------------------------------------------
        # 2. CONSENSUS REMOVAL
        # -------------------------------------------------------------
        filtered = consensus_remover(compact)

        # -------------------------------------------------------------
        # 3. CONSENSUS PICKERS
        # -------------------------------------------------------------
        per_expert_probs = {name: agg[name]["probs_list"] for name in agg}
        picked, remaining_candidates, remaining_budget, required = consensus_pickers(
            filtered, per_expert_probs, budget
        )

        picked_names = {p["name"] for p in picked}
        remaining_candidates = [c for c in remaining_candidates if c["name"] not in picked_names]

        # -------------------------------------------------------------
        # 4. LLM PROMPT
        # -------------------------------------------------------------
        num_needed = sum(required.values())

        SYSTEM = f"""
You are the Meta FPL Selector.

You must fill the remaining {num_needed} players to complete a 15-player squad.

LOCKED PLAYERS (already selected):
{json.dumps(picked, ensure_ascii=False)}

Remaining budget: {remaining_budget}
Remaining positions needed:
{json.dumps(required)}

Formation rules:
- 2 GK
- 5 DEF
- 5 MID
- 3 FWD
Max 3 players per club.

You will receive the remaining candidates.
Choose EXACTLY the required slots per position.

Return ONLY strict JSON with these keys:
- selected: list of remaining players
- bench: []
- justification: {{"overall": "..."}}
- constraints_violated: []
"""

        user_prompt = "Remaining candidates:\n" + json.dumps(remaining_candidates, ensure_ascii=False)

        try:
            resp = await call_llm(
                SYSTEM, user_prompt, model=self.model, temperature=0.0, max_tokens=META_MAX_TOKENS
            )

            raw = resp["text"].strip()
            if "```" in raw:
                raw = raw.split("```")[1].strip()

            parsed = json.loads(raw)
            llm_selected = parsed["selected"]

            final_team = picked + llm_selected

            violations = _validate_team(final_team, budget, max_per_club)
            if violations:
                fb = deterministic_fallback(compact, budget)
                fb["constraints_violated"] = violations
                return fb

            parsed["selected"] = final_team
            parsed["constraints_violated"] = []
            return parsed

        except Exception as e:
            logger.error(f"MetaAgent exception: {e}")
            fb = deterministic_fallback(compact, budget)
            fb["constraints_violated"] = ["exception"]
            return fb
