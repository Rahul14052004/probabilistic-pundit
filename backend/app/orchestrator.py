# backend/app/orchestrator.py

import asyncio
from loguru import logger

from .agents.expert_agent import ExpertAgent
from .agents.meta_agent import MetaAgent
from .data_loader import load_aggregated_players_for_season_gw
from .topsis_filter import select_top_players_with_topsis
from typing import Dict, Any, List

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

        # 3. meta-synthesis
        #    Important: pass candidates into the request so MetaAgent can fill name/position/club/price
        meta_request = dict(request or {})
        meta_request["candidates"] = candidates

        team = await self.meta.synthesize(expert_outputs, meta_request)

        explanation = {
            "expert_outputs": expert_outputs,
            "meta_decision": "synthesized",
        }
        return team, explanation

    def _filter_candidates(self, season: str, gameweek: int, n_candidates: int = 30) -> List[Dict[str, Any]]:
        """
        Use Vaastav GW data for the given season up to (but not including) 'gameweek',
        aggregate stats per player, then run TOPSIS to select the top
        n_candidates players as the candidate pool for the agents.
        """
        df_agg = load_aggregated_players_for_season_gw(season, gameweek)
        df_cand = select_top_players_with_topsis(df_agg, n_candidates=n_candidates)

        candidates: List[Dict[str, Any]] = []
        for _, row in df_cand.iterrows():
            candidates.append(
                {
                    "name": str(row["name"]),
                    "position": str(row["position"]),
                    "club": str(row["club"]),
                    "price": float(row["price"]),
                    "ev": float(row.get("expected_points", 0.0) or 0.0),
                    "std": None,
                    "topsis_score": float(row.get("topsis_score", 0.0)),
                    "total_points": float(row.get("total_points", 0.0)),
                    "minutes": float(row.get("minutes", 0.0)),
                    "goals_scored": float(row.get("goals_scored", 0.0)),
                    "assists": float(row.get("assists", 0.0)),
                }
            )
        
        print("*******************")
        print(candidates)
        print("*******************")

        return candidates

    def compare_players(self, payload: Dict[str, Any]):
        return {"comparison": "not implemented"}