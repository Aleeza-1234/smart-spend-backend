from pydantic import BaseModel, Field
from datetime import datetime, date
from typing import Optional

class TravelPlanBase(BaseModel):
    destination: str
    duration_days: int = Field(..., gt=0)
    estimated_cost: float = Field(..., gt=0)
    travel_style: str = Field(default="mid", pattern="^(budget|mid|luxury)$")
    target_date: Optional[date] = None

class TravelPlanCreate(TravelPlanBase):
    pass

class TravelPlanUpdate(BaseModel):
    destination: Optional[str] = None
    duration_days: Optional[int] = None
    estimated_cost: Optional[float] = None
    current_savings: Optional[float] = None
    target_date: Optional[date] = None

class TravelPlanResponse(TravelPlanBase):
    id: int
    user_id: int
    current_savings: float
    is_completed: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class TravelTimelinePrediction(BaseModel):
    travel_plan_id: int
    destination: str
    duration_days: int
    travel_style: str
    total_cost: float
    current_savings: float
    remaining_needed: float
    months_needed: float
    target_date: str  # ISO format date string
    best_case_months: float
    worst_case_months: float
    best_case_date: str
    worst_case_date: str
    monthly_savings_needed: float
    weekly_savings_needed: float
    daily_savings_needed: float
    milestones: list[dict]
    savings_rate_info: dict
    confidence: float