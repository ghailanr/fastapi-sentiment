from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import os
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging
model_path = os.path.join(os.path.dirname(__file__), "model/model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "model/tfidf.pkl")

INSTRUMENTATION_KEY = "fc646e0f-4800-497c-ae30-69ab0005cb45"

app = FastAPI()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(AzureLogHandler(connection_string=f'InstrumentationKey={INSTRUMENTATION_KEY}'))
logger.info("L'API FastAPI a bien démarré et envoie des logs à Azure.")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


class TweetInput(BaseModel):
    tweet: str


class PredictionFeedback(BaseModel):
    correct: bool


@app.get("/")
def home():
    return {"message": "API de prédiction de sentiment avec régression logistique"}


@app.post("/predict/")
def predict_sentiment(userInput: TweetInput):
    # Transformer le tweet en vecteur
    tweet_vectorized = vectorizer.transform([userInput.tweet])
    prediction = int(model.predict(tweet_vectorized)[0])
    sentiment = "POSITIF" if prediction == 1 else "NEGATIF"
    return {"prediction": sentiment}


@app.post("/feedback/")
def feedback(userInput: PredictionFeedback):
    correct = userInput.correct
    log_message = {
        "feedback": correct
    }
    logger.info("User feedback", extra={"custom_dimensions": log_message})
    return {"message": "Feedback sent"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)