# backend/app/orchestrator.py

import asyncio
from loguru import logger

from .agents.expert_agent import ExpertAgent
from .agents.meta_agent import MetaAgent


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
        candidates = self._filter_candidates()
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

    def _filter_candidates(self):
        """
        Placeholder deterministic candidate filter returning correctly shaped dicts.
        Replace or extend this to query your DB / feature store later.

        Minimal fields used downstream:
        - player_id (int)
        - name (str)
        - position (str)   e.g. 'GK', 'DEF', 'MID', 'FWD'
        - club (str)
        - price (float)    FPL price in millions
        """
        return [
            {"player_id": 1, "name": "Player 1", "position": "MID", "club": "ClubA", "price": 5.0},
            {"player_id": 2, "name": "Player 2", "position": "FWD", "club": "ClubB", "price": 7.5},
            {"player_id": 3, "name": "Player 3", "position": "DEF", "club": "ClubC", "price": 4.5},
            {"player_id": 4, "name": "Player 4", "position": "GK",  "club": "ClubD", "price": 5.5},
            {"player_id": 5, "name": "Player 5", "position": "MID", "club": "ClubE", "price": 6.0},
            # Add more demo entries or replace with real data later
        ]

    def compare_players(self, payload: dict):
        """
        Placeholder for a future endpoint that compares two or more players.
        """
        return {"comparison": "not implemented"}
