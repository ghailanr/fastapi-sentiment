from fastapi import FastAPI
import joblib
import pandas as pd

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/tfidf.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API de prédiction de sentiment avec régression logistique"}


@app.post("/predict/")
def predict_sentiment(tweet: str):
    # Transformer le tweet en vecteur
    tweet_vectorized = vectorizer.transform([tweet])

    prediction = int(model.predict(tweet_vectorized)[0])
    sentiment = "POSITIF" if prediction == 1 else "NEGATIF"
    return {"tweet": tweet, "sentiment": sentiment}