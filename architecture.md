# Architecture Overview


- Frontend (Streamlit) <--> Backend (FastAPI)
- Backend orchestrates 3 Expert Agents + Meta Agent
- Agents use either Hugging Face models (local) or remote LLM APIs
- CI/CD: GitHub Actions builds Docker image, runs tests, and deploys to Google Cloud Run
- Monitoring: Prometheus scrapes FastAPI metrics endpoint; Grafana visualizes