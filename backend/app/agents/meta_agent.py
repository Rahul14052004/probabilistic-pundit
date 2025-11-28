# backend/app/agents/meta_agent.py

from typing import Any, Dict, List, Optional
import json
import os
from loguru import logger
from ..llm_client import call_llm

META_DEFAULT_MODEL = os.getenv("META_DEFAULT_MODEL", "llama-3.1-70b-versatile")
META_MAX_TOKENS = int(os.getenv("META_MAX_TOKENS", "3500"))


# ------------------------------------------------------------------------------------
# DETERMINISTIC FALLBACK  (uses value instead of price)
# ------------------------------------------------------------------------------------

def deterministic_fallback(candidates: List[Dict[str, Any]], budget: float):
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
        cost = float(c["value"])

        if required[pos] <= 0:
            continue
        if spent + cost > budget:
            continue

        team.append(c)
        required[pos] -= 1
        spent += cost

        if all(v == 0 for v in required.values()):
            break

    return {
        "selected": team,
        "bench": [],
        "justification": {"overall": "Fallback greedy selection based on Tickers + Haulers"},
        "constraints_violated": [],
    }


# ------------------------------------------------------------------------------------
# CONSENSUS REMOVAL
# ------------------------------------------------------------------------------------

def consensus_remover(candidates: List[Dict[str, Any]]):
    bad = [
        c for c in candidates
        if c["probs"]["Tickers"] <= 0.10 and c["probs"]["Haulers"] <= 0.05
    ]

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
# CONSENSUS PICKERS (uses value, not price)
# ------------------------------------------------------------------------------------

def consensus_pickers(candidates: List[Dict[str, Any]], per_expert_probs: Dict[str, List[Dict[str, float]]], budget: float):
    required = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

    picked = []
    filtered = []
    remaining_budget = budget

    for c in candidates:
        key = f"{c['name']}_{c['team']}"
        probs_list = per_expert_probs.get(key, [])

        high_votes = sum(
            1 for p in probs_list
            if p.get("Tickers", 0) >= 0.70 or p.get("Haulers", 0) >= 0.70
        )

        if high_votes >= 2:
            picked.append(c)
            remaining_budget -= float(c["value"])
            required[c["position"]] -= 1
        else:
            filtered.append(c)

    for pos in required:
        required[pos] = max(required[pos], 0)

    return picked, filtered, remaining_budget, required


# ------------------------------------------------------------------------------------
# TEAM VALIDATION (value-based)
# ------------------------------------------------------------------------------------

def _validate_team(selected: List[Dict[str, Any]], budget: float, max_per_club: int = 3):
    violations = []

    if len(selected) != 15:
        violations.append(f"selected_count={len(selected)} != 15")

    spent = sum(float(p["value"]) for p in selected)
    if spent > budget:
        violations.append(f"budget_exceeded={spent:.1f} > {budget}")

    pos_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    club_counts = {}

    for p in selected:
        pos_counts[p["position"]] += 1
        club = p["team"]
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
# META AGENT  (NO EV, NO STD, USE VALUE)
# ------------------------------------------------------------------------------------

class MetaAgent:
    def __init__(self, name="meta_agent", model=None):
        self.name = name
        self.model = model or META_DEFAULT_MODEL

    async def synthesize(self, expert_outputs, request):

        budget = float(request.get("budget", 100.0))
        max_per_club = int(request.get("max_per_club", 3))

        # -------------------------------------------------------------
        # 1. Aggregate expert outputs (key = name_team)
        # -------------------------------------------------------------
        agg = {}

        for out in expert_outputs:
            agent_name = out["agent"]
            for r in out["recommendations"]:
                name = r["name"]

                # find real team from candidates
                team = next(c["team"] for c in request["candidates"] if c["name"] == name)

                key = f"{name}_{team}"

                if key not in agg:
                    agg[key] = {"name": name, "team": team, "probs_list": [], "justifications": []}

                agg[key]["probs_list"].append(r["probs"])
                agg[key]["justifications"].append(f"{agent_name}: {r['justification']}")

        # -------------------------------------------------------------
        # 2. Build compact list (NO EV, NO STD, USE VALUE)
        # -------------------------------------------------------------
        meta_map = {f"{c['name']}_{c['team']}": c for c in request["candidates"]}

        compact = []
        for key, entry in agg.items():
            meta = meta_map[key]

            name = entry["name"]
            team = entry["team"]
            position = meta["position"]
            value = float(meta["value"])

            # Average probabilities
            avg = {}
            keys = ["Zeros", "Blanks", "Tickers", "Haulers"]
            for k in keys:
                avg[k] = sum(p[k] for p in entry["probs_list"]) / len(entry["probs_list"])

            total = sum(avg.values()) or 1.0
            avg = {k: v / total for k, v in avg.items()}

            compact.append({
                "name": name,
                "team": team,
                "position": position,
                "value": value,
                "probs": avg,
                "expert_justifications": entry["justifications"],
            })

        # -------------------------------------------------------------
        # 3. Consensus removal
        # -------------------------------------------------------------
        filtered = consensus_remover(compact)

        # -------------------------------------------------------------
        # 4. Consensus picks
        # -------------------------------------------------------------
        per_expert_probs = {key: agg[key]["probs_list"] for key in agg}
        picked, remaining, remaining_budget, required = consensus_pickers(
            filtered, per_expert_probs, budget
        )

        picked_set = {f"{p['name']}_{p['team']}" for p in picked}
        remaining = [
            c for c in remaining
            if f"{c['name']}_{c['team']}" not in picked_set
        ]

        # -------------------------------------------------------------
        # 5. LLM Selection
        # -------------------------------------------------------------
        num_needed = sum(required.values())

        SYSTEM = f"""
You are the Meta FPL Selector.

Locked players:
{json.dumps(picked, ensure_ascii=False)}

Remaining budget: {remaining_budget}
Required slots:
{json.dumps(required)}

Formation:
2 GK, 5 DEF, 5 MID, 3 FWD
Max 3 per club.

Pick exactly {num_needed} players.
Return STRICT JSON.
"""

        user_prompt = "Remaining candidates:\n" + json.dumps(remaining, ensure_ascii=False)

        try:
            resp = await call_llm(
                SYSTEM, user_prompt,
                model=self.model,
                temperature=0.0,
                max_tokens=META_MAX_TOKENS
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
