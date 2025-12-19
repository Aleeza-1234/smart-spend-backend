from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Date
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

class TravelPlan(Base):
    __tablename__ = "travel_plans"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    destination = Column(String(100), nullable=False)
    duration_days = Column(Integer, nullable=False)
    
    estimated_cost = Column(Float, nullable=False)
    current_savings = Column(Float, default=0.0)
    
    travel_style = Column(String(20), default="mid")  # budget, mid, luxury
    target_date = Column(Date, nullable=True)
    
    is_completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="travel_plans")

class TravelCostPrediction(Base):
    __tablename__ = "travel_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    travel_plan_id = Column(Integer, ForeignKey("travel_plans.id"), nullable=False)
    
    predicted_months = Column(Float, nullable=False)
    confidence_lower = Column(Float, nullable=False)
    confidence_upper = Column(Float, nullable=False)
    
    monthly_savings_needed = Column(Float, nullable=False)
    predicted_date = Column(Date, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)