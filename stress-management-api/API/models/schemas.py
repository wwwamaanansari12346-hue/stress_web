from pydantic import BaseModel
from typing import Optional

class StressPredictionRequest(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: Optional[float] = None
    feature_4: Optional[str] = None
    # Add additional features as needed

class StressPredictionResponse(BaseModel):
    predicted_stress_level: str
    confidence: float