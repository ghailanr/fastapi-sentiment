from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_sentiment():
    response = client.post("/predict", json={"tweet": "J'adore ce produit !"})
    assert response.status_code == 200
    assert response.json()["sentiment"] in ["POSITIF", "NEGATIF"]