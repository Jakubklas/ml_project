from fastapi import FastAPI
from model import predict_value
from pydantic import BaseModel

app = FastAPI()
#hey bye
@app.get("/")
def read_root():
    return {"message": "Hello World!"}

# Define input data structure
class PredictionInput(BaseModel):
    value: float

@app.post("/predict"):
def predict(input_data: PredictionInput):
    prediction = predict_value(input_data.value)
    return {"prediction": prediction}
