from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict/", json={"tweet": "J'adore ce produit !"})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data

def test_feedback_valid():
    response = client.post("/predict/", json={"tweet": "This was a great movie !"})
    assert response.status_code == 200
    feedback_response = client.post("/feedback/", json={"correct": True})
    assert feedback_response.status_code == 200
    assert feedback_response.json() == {"message": "Feedback sent"}

def test_feedback_invalid():
    response = client.post("/feedback/", json={"correct": False})
    assert response.status_code == 200
    assert response.json() == {"message": "Feedback sent"}
