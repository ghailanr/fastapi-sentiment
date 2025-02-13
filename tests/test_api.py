from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict/", json={"tweet": "J'adore ce produit !"})
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "prediction" in data

def test_feedback_valid():
    response = client.post("/predict/", json={"tweet": "This was a great movie !"})
    assert response.status_code == 200
    prediction_id = response.json()["id"]

    feedback_response = client.post("/feedback/", json={"prediction_id": prediction_id, "correct": True})
    assert feedback_response.status_code == 200
    assert feedback_response.json() == {"message": "Feedback enregistré"}

def test_feedback_invalid():
    response = client.post("/feedback/", json={"prediction_id": 99999, "correct": False})
    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction ID not found"
