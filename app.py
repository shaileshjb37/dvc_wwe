from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# Define input schema
class InputData(BaseModel):
    wrestler_1_win_rate: float
    wrestler_2_win_rate: float
    storyline_rivalry: int
    recent_form: int


# ✅ Root endpoint (fixes test_root 404 error)
@app.get("/")
def root():
    return {"status": "ok", "message": "WWE Match Prediction API is running"}


# ✅ Prediction endpoint
@app.post("/predict")
def predict(input_data: InputData):
    # Use model_dump() for Pydantic v2
    data = input_data.model_dump()

    # Example dummy logic (replace with your ML model later)
    if data["wrestler_1_win_rate"] > data["wrestler_2_win_rate"]:
        prediction = "Wrestler 1 wins"
        probability = 0.85
    else:
        prediction = "Wrestler 2 wins"
        probability = 0.65

    return {
        "input": data,
        "prediction": prediction,
        "probability": probability
    }
