from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.prediction_service import predict_stress_level

router = APIRouter()

class PredictionRequest(BaseModel):
    input_data: dict

class PredictionResponse(BaseModel):
    predicted_stress_level: str

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        predicted_level = predict_stress_level(request.input_data)
        return PredictionResponse(predicted_stress_level=predicted_level)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))