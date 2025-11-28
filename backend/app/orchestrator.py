# backend/app/orchestrator.py

import asyncio
from loguru import logger

from .agents.expert_agent import ExpertAgent
from .agents.meta_agent import MetaAgent
from .data_loader import load_aggregated_players_for_season_gw
from .topsis_filter import select_top_players_with_topsis
from typing import Dict, Any, List
from .utils.logger_utils import save_json_log
import pandas as pd


class Orchestrator:
    def __init__(self):
        """
        Orchestrator wires together:
        - 3 expert agents (personas)
        - 1 meta agent (final squad construction + explanation)
        """
        # If you want generic experts: [ExpertAgent(name=f"expert_{i}") for i in range(3)]
        # Here we wire the three personas explicitly (value, safe, differential)
        self.experts = [
            ExpertAgent(name="value_hunter", persona="value_hunter"),
            ExpertAgent(name="safe_bet", persona="safe_bet"),
            ExpertAgent(name="differentials_specialist", persona="differentials_specialist"),
        ]
        self.meta = MetaAgent()

    async def generate_team(self, request: dict):
        """
        Main pipeline entrypoint.

        1. Get candidate pool (currently a hardcoded placeholder)
        2. Ask all experts to score candidates in parallel
        3. Pass expert outputs + candidates to the Meta LLM to pick the final 15
        4. Return (team, explanation) pair
        """
        # 1. deterministic candidate filter (later: TOPSIS / DB / Neon)
        season = request.get("season")
        gameweek = request.get("gameweek")

        if not season or not gameweek:
            raise ValueError("season and gameweek are required in request payload")

        candidates = self._filter_candidates(season, gameweek)
        logger.info(f"Orchestrator: using {len(candidates)} candidates")

        # 2. call experts in parallel
        expert_tasks = [e.analyze(candidates, request) for e in self.experts]
        expert_outputs = await asyncio.gather(*expert_tasks)

        for eo in expert_outputs:
                save_json_log(season, gameweek, f"expert_{eo['agent']}", eo)

        # 3. meta-synthesis
        #    Important: pass candidates into the request so MetaAgent can fill name/position/club/price
        meta_request = dict(request or {})
        meta_request["candidates"] = candidates

        team = await self.meta.synthesize(expert_outputs, meta_request)

        explanation = {
            "expert_outputs": expert_outputs,
            "meta_decision": "synthesized",
        }
        save_json_log(season, gameweek, "meta_output", team)
        return team, explanation

    
    def _filter_candidates(self, season: str, gameweek: int, n_candidates: int = 30) -> List[Dict[str, Any]]:
        """
        Use season + gameweek GW data to:

        1) Aggregate per-player stats up to (but not including) the target GW
        2) Run TOPSIS to pick the top n_candidates
        3) Return a clean list of player dicts using the new schema:

            name, position, team, price, xP, xGI, minutes,
            fixture, selected_by, threat, ict, topsis_score
        """
        # 1) aggregate
        df_agg = load_aggregated_players_for_season_gw(season, gameweek)

        # 2) topsis selection
        df_cand = select_top_players_with_topsis(df_agg, n_candidates=n_candidates)

        candidates: List[Dict[str, Any]] = []
        for _, row in df_cand.iterrows():
            candidates.append(
                {
                    "name": str(row.get("name", "")),
                    "position": str(row.get("position", "")),
                    "team": str(row.get("team", "")),
                    "value": float(row.get("value", 0.0) or 0.0),

                    "xP": float(row.get("xP", 0.0) or 0.0),
                    "xGI": float(row.get("xGI", 0.0) or 0.0),
                    "minutes": float(row.get("minutes", 0.0) or 0.0),
                    "fixture": str(row.get("fixture", "")),
                    "selected_by": float(row.get("selected_by", 0.0) or 0.0),
                    "threat": float(row.get("threat", 0.0) or 0.0),
                    "ict": float(row.get("ict", 0.0) or 0.0),

                    "topsis_score": float(row.get("topsis_score", 0.0) or 0.0),
                }
            )

        print("******************* CANDIDATES *******************")
        for c in candidates:
            print(c)
        print("**************************************************")

        save_json_log(season, gameweek, "candidates", candidates)
        save_json_log(season, gameweek, "topsis", df_cand.to_dict(orient="records"))
        return candidates


    def compare_players(self, payload: Dict[str, Any]):
        return {"comparison": "not implemented"}