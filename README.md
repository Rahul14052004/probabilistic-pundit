# Probabilistic Pundit


A multi-agent LLM-powered Fantasy Premier League advisor with an MLOps pipeline.


## What this repository contains
- Backend: FastAPI service that orchestrates LLM agents
- Frontend: Streamlit chat UI
- CI/CD: GitHub Actions workflow for build, test, and deploy to Google Cloud Run
- Monitoring: Prometheus config and Grafana dashboard skeleton
- Data ingestion: weekly updater for FPL datasets


## Quickstart (local)
1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. `cd backend && uvicorn app.main:app --reload --port 8000`
4. `streamlit run frontend/streamlit_app.py`


## Deploy
See `deploy/cloudrun_deploy.sh` and `.github/workflows/ci-cd.yml`.


## Project structure
(See repo root file tree)


## Contact
Mehul Agarwal
Rahul Ramesh Omalur