from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime

class CategoryPredictionRequest(BaseModel):
    merchant: Optional[str] = None
    amount: float
    transaction_text: Optional[str] = None

class CategoryPredictionResponse(BaseModel):
    predicted_category: str
    confidence: float
    all_probabilities: Dict[str, float]

class ExpensePredictionRequest(BaseModel):
    user_id: int
    category: Optional[str] = None
    month: int
    year: int

class ExpensePredictionResponse(BaseModel):
    predicted_amount: float
    confidence_interval: tuple[float, float]
    historical_average: float

class AnomalyDetectionRequest(BaseModel):
    user_id: int
    amount: float
    category: str
    merchant: Optional[str] = None

class AnomalyDetectionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    reason: Optional[str] = None