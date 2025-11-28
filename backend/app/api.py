from fastapi import APIRouter
from pydantic import BaseModel
from .orchestrator import Orchestrator
from typing import Optional

router = APIRouter()

class TeamRequest(BaseModel):
    budget: float = 1000.0
    season: Optional[str] = None        # 🔹 NEW
    gameweek: Optional[int] = None      # was only on frontend before, now explicit
    chips: Optional[list[str]] = None
    constraints: Optional[dict] = None

orchestrator = Orchestrator()

@router.post('/generate_team')
async def generate_team(req: TeamRequest):
    team, explanation = await orchestrator.generate_team(req.dict())
    return {'team': team, 'explanation': explanation}

@router.post('/compare_players')
async def compare_players(payload: dict):
    return orchestrator.compare_players(payload)

@router.post('/explain_team')
async def explain_team(payload: dict):
    return {'explanation': 'Not implemented yet'}

from fastapi import FastAPI
app = FastAPI()
app.include_router(router)