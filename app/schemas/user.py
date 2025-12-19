"""
User Pydantic schemas
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    email: str
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    monthly_salary: Optional[float] = None
    pocket_money: Optional[float] = None
    current_balance: Optional[float] = 0.0

class UserCreate(UserBase):
    password: str = Field(..., min_length=6, max_length=72, description="Password must be between 6-72 characters")

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True