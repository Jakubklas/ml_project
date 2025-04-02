from fastapi import FastAPI
from model import predict_class, predict_proba
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def get_response():
    return {"message": "Hello World!"}

# Checks for the right calling format
class InputFormat(BaseModel):
    features: list[list[float]]

# Predict classes
@app.post("/classes")
def predictions(input: InputFormat) -> dict:
    prediction = predict_class(input.features)[0]
    proba = max(predict_proba(input.features)[0])
    return {
        "predictions": [prediction, proba]
    }

# Predict probabilities