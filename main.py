from fastapi import FastAPI
from model import predict_values, predict_probas
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def get_response():
    return {"message": "Hello World!"}


class PredictionInput(BaseModel):
    x: list[list[float]]

@app.post("/predict")
def predict(batch: PredictionInput) -> dict:
    classes = predict_values(batch.x)
    probas = predict_probas(batch.x)
    return {
        "predictions": classes.tolist(),
        "probabilities": probas.tolist()
        }