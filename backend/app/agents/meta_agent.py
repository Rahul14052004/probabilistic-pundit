# backend/app/agents/meta_agent.py

from typing import Any, Dict, List, Optional
import json
import os
from loguru import logger
from ..llm_client import call_llm

META_DEFAULT_MODEL = os.getenv("META_DEFAULT_MODEL", "llama-3.1-70b-versatile")
META_MAX_TOKENS = int(os.getenv("META_MAX_TOKENS", "3500"))

# ------------------------------------------------------------------------------------
# DETERMINISTIC FALLBACK (corrected)
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
# CONSENSUS REMOVAL
# ------------------------------------------------------------------------------------

def consensus_remover(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bad_flags = []
    for c in candidates:
        p = c["probs"]
        if p["Tickers"] <= 0.10 and p["Haulers"] <= 0.05:
            bad_flags.append(c)

    # Count per position
    count = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for c in candidates:
        count[c["position"]] += 1

    min_req = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

    keep = []
    for c in candidates:
        if c not in bad_flags:
            keep.append(c)
            continue

        pos = c["position"]
        if count[pos] - 1 < min_req[pos]:
            keep.append(c)
        else:
            count[pos] -= 1

    return keep


# ------------------------------------------------------------------------------------
# CONSENSUS PICKERS (corrected for "2+ experts" rule)
# ------------------------------------------------------------------------------------

def consensus_pickers(candidates: List[Dict[str, Any]], prob_lists_map: Dict[int, List[Dict[str, float]]], budget: float):
    required = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    picked = []
    filtered = []
    remaining_budget = budget

    for c in candidates:
        pid = c["player_id"]
        per_expert_probs = prob_lists_map.get(pid, [])

        # count experts who believe tickers/haulers >= 0.7
        high_votes = 0
        for p in per_expert_probs:
            if p.get("Tickers", 0) >= 0.70 or p.get("Haulers", 0) >= 0.70:
                high_votes += 1

        if high_votes >= 2:  # actual intended rule
            picked.append(c)
            remaining_budget -= float(c["price"])
            required[c["position"]] -= 1
        else:
            filtered.append(c)

    # safety: prevent negatives
    for k in required:
        if required[k] < 0:
            required[k] = 0

    return picked, filtered, remaining_budget, required


# ------------------------------------------------------------------------------------
# TEAM VALIDATION (corrected formation)
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
# META AGENT
# ------------------------------------------------------------------------------------

class MetaAgent:
    def __init__(self, name="meta_agent", model=None):
        self.name = name
        self.model = model or META_DEFAULT_MODEL

    async def synthesize(self, expert_outputs, request):

        budget = float(request.get("budget", 100.0))
        max_per_club = int(request.get("max_per_club", 3))

        # -------------------------------------------------------------
        # 1. Aggregate expert outputs
        # -------------------------------------------------------------

        agg = {}
        for out in expert_outputs:
            agent_name = out["agent"]
            for r in out["recommendations"]:
                pid = r["player_id"]
                if pid not in agg:
                    agg[pid] = {"probs_list": [], "justifications": []}

                agg[pid]["probs_list"].append(r["probs"])
                if "justification" in r:
                    agg[pid]["justifications"].append(f"{agent_name}: {r['justification']}")

        details = {c["player_id"]: c for c in request["candidates"]}

        compact = []
        for pid, entry in agg.items():
            meta = details[pid]
            avg = {}
            keys = ["Zeros", "Blanks", "Tickers", "Haulers"]
            for k in keys:
                avg[k] = sum(p[k] for p in entry["probs_list"]) / len(entry["probs_list"])

            total = sum(avg.values()) or 1.0
            avg = {k: v / total for k, v in avg.items()}

            compact.append({
                "player_id": pid,
                "name": meta["name"],
                "position": meta["position"],
                "club": meta["club"],
                "price": meta["price"],
                "ev": meta["ev"],
                "std": meta["std"],
                "probs": avg,
                "expert_justifications": entry["justifications"],
            })

        # ------------------------------------------------------------------
        # 2. CONSENSUS REMOVAL
        # ------------------------------------------------------------------
        filtered = consensus_remover(compact)

        # ------------------------------------------------------------------
        # 3. CONSENSUS PICKERS — using per-expert probability lists
        # ------------------------------------------------------------------
        prob_map = {pid: agg[pid]["probs_list"] for pid in agg}
        picked, remaining, remaining_budget, required = consensus_pickers(filtered, prob_map, budget)

        # remove picked IDs from remaining
        remaining = [c for c in remaining if c["player_id"] not in {p["player_id"] for p in picked}]

        # ------------------------------------------------------------------
        # 4. LLM PROMPT — corrected + safe
        # ------------------------------------------------------------------

        num_needed = sum(required.values())

        SYSTEM = f"""
You are the Meta FPL Selector.

You MUST fill the remaining {num_needed} players to complete a 15-player squad.

Already selected (LOCKED):
{json.dumps(picked, ensure_ascii=False)}

Remaining budget: {remaining_budget}
Remaining slots:
{json.dumps(required)}

Formation rules:
- 2 GK
- 5 DEF
- 5 MID
- 3 FWD
Max 3 players per club.

You will receive the list of remaining candidates.
Pick ONLY the required counts for each position.
Never exceed remaining budget.
Never pick players already locked.

RETURN STRICT JSON:
{{
    "selected": [... exactly {num_needed} players ...],
    "bench": [],
    "justification": {{"overall": "your explanation"}},
    "constraints_violated": []
}}
"""

        user_prompt = "Remaining candidates:\n" + json.dumps(remaining, ensure_ascii=False)

        try:
            resp = await call_llm(SYSTEM, user_prompt, model=self.model, temperature=0.0, max_tokens=META_MAX_TOKENS)
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
            logger.error(f"MetaAgent error: {e}")
            fb = deterministic_fallback(compact, budget)
            fb["constraints_violated"] = ["exception"]
            return fb
