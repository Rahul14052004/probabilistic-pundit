from fastapi import FastAPI
from .api import router as api_router
from prometheus_client import generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title='Probabilistic Pundit API')
app.include_router(api_router, prefix='/api')

@app.get('/metrics')
async def metrics():
    # Basic Prometheus exposition placeholder
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get('/')
async def root():
    return {'status': 'ok', 'service': 'probabilistic-pundit'}
