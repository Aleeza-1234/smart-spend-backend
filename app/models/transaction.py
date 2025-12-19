from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    amount = Column(Float, nullable=False)
    merchant = Column(String(200))
    category = Column(String(50), index=True)
    transaction_type = Column(String(10))  # DEBIT, CREDIT
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    raw_sms = Column(Text)
    account_balance = Column(Float, nullable=True)
    
    upi_id = Column(String(100), nullable=True)
    bank_name = Column(String(100), nullable=True)
    
    # Predicted fields (from ML)
    predicted_category = Column(String(50), nullable=True)
    confidence_score = Column(Float, nullable=True)
    is_anomaly = Column(Integer, default=0)  # 0 = normal, 1 = anomaly
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="transactions")