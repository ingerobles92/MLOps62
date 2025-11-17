from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List

# FastAPI app initialization
app = FastAPI(title="Absenteeism Prediction API", version="1.0.0")

# Load model
with open('../../models/best_model_svr.pkl', 'rb') as f:
    model = pickle.load(f)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    Age: float
    Distance_from_Residence_to_Work: float
    Service_time: float
    Work_load_Average_day: float
    Transportation_expense: float
    Height: float
    Weight: float
    Body_mass_index: float
    Children: int
    Pet: int
    Disciplinary_failure: int
    Education: int
    Social_drinker: int
    Social_smoker: int

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str = "SVR_v1.0"

@app.get("/")
def root():
    return {"message": "Absenteeism Prediction API", "endpoints": ["/predict", "/health"]}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Convert request to DataFrame
        data = pd.DataFrame([request.dict()])

        # Make prediction
        prediction = model.predict(data)[0]

        return PredictionResponse(prediction=float(prediction))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
