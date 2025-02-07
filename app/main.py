from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

model_path = os.path.join(os.path.dirname(__file__), "model/model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "model/tfidf.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

app = FastAPI()

class TweetInput(BaseModel):
    tweet: str

@app.get("/")
def home():
    return {"message": "API de prédiction de sentiment avec régression logistique"}


@app.post("/predict/")
def predict_sentiment(input: TweetInput):
    # Transformer le tweet en vecteur
    tweet_vectorized = vectorizer.transform([input.tweet])

    prediction = int(model.predict(tweet_vectorized)[0])
    sentiment = "POSITIF" if prediction == 1 else "NEGATIF"
    return {"tweet": input.tweet, "sentiment": sentiment}