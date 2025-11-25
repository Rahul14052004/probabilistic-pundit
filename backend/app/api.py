from fastapi import APIRouter
from pydantic import BaseModel
from .orchestrator import Orchestrator

router = APIRouter()

class TeamRequest(BaseModel):
    budget: float = 100.0
    chips: list[str] | None = None
    constraints: dict | None = None

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
