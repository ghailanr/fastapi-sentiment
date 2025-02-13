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
# logger.setLevel(logging.INFO)
logger.addHandler(AzureLogHandler(connection_string=f'InstrumentationKey={INSTRUMENTATION_KEY}'))

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
predictions = {}


class TweetInput(BaseModel):
    tweet: str


class PredictionFeedback(BaseModel):
    prediction_id: int
    correct: bool


@app.get("/")
def home():
    return {"message": "API de prédiction de sentiment avec régression logistique"}


@app.post("/predict/")
def predict_sentiment(userInput: TweetInput):
    # Transformer le tweet en vecteur
    tweet_vectorized = vectorizer.transform([userInput.tweet])
    prediction = int(model.predict(tweet_vectorized)[0])
    prediction_id = len(predictions) + 1
    predictions[prediction_id] = {"tweet": userInput.tweet, "prediction": prediction}
    sentiment = "POSITIF" if prediction == 1 else "NEGATIF"
    return {"id": prediction_id, "prediction": sentiment}


@app.post("/feedback/")
def feedback(userInput: PredictionFeedback):
    prediction_id = userInput.prediction_id
    correct = userInput.correct
    if prediction_id not in predictions:
        raise HTTPException(status_code=404, detail="Prediction ID not found")

    data = predictions[prediction_id]
    logger.warning("User feedback", extra={
        "custom_dimensions": {
            "tweet": data["tweet"],
            "prediction": data["prediction"],
            "correct": correct
        }
    })

    return {"message": "Feedback enregistré"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)