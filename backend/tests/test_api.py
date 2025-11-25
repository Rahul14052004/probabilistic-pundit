from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_root():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'

def test_generate_team():
    r = client.post('/api/generate_team', json={'budget': 100.0})
    assert r.status_code == 200
    assert 'team' in r.json()