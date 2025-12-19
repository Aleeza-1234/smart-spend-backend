from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

class BudgetBase(BaseModel):
    category: str
    monthly_limit: float = Field(..., gt=0)

class BudgetCreate(BudgetBase):
    pass

class BudgetUpdate(BaseModel):
    monthly_limit: Optional[float] = Field(None, gt=0)
    is_active: Optional[bool] = None

class BudgetResponse(BudgetBase):
    id: int
    user_id: int
    current_spent: float
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class DailyBudgetResponse(BaseModel):
    date: datetime
    food_budget: float
    discretionary_budget: float
    total_daily: float
    days_remaining: int
    monthly_income: float
    fixed_expenses: float
    savings_goal: float

class IncomeBase(BaseModel):
    source: str
    amount: float = Field(..., gt=0)
    date: datetime
    is_recurring: bool = False

class IncomeCreate(IncomeBase):
    pass

class IncomeResponse(IncomeBase):
    id: int
    user_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class NecessityBase(BaseModel):
    category: str
    amount: float = Field(..., gt=0)
    frequency: str = "monthly"

class NecessityCreate(NecessityBase):
    pass

class NecessityResponse(NecessityBase):
    id: int
    user_id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True