from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    full_name = Column(String(100))
    phone_number = Column(String(20), unique=True)
    
    current_balance = Column(Float, default=0.0)
    monthly_salary = Column(Float, nullable=True)
    pocket_money = Column(Float, nullable=True)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    transactions = relationship("Transaction", back_populates="user", cascade="all, delete-orphan")
    budgets = relationship("Budget", back_populates="user", cascade="all, delete-orphan")
    income = relationship("Income", back_populates="user", cascade="all, delete-orphan")
    necessities = relationship("Necessity", back_populates="user", cascade="all, delete-orphan")
    travel_plans = relationship("TravelPlan", back_populates="user", cascade="all, delete-orphan")