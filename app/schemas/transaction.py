from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class TransactionBase(BaseModel):
    amount: float = Field(..., gt=0)
    merchant: Optional[str] = None
    category: Optional[str] = None
    transaction_type: str = Field(..., pattern="^(DEBIT|CREDIT)$")
    raw_sms: Optional[str] = None
    timestamp: Optional[datetime] = None

class TransactionCreate(TransactionBase):
    pass

class TransactionUpdate(BaseModel):
    category: Optional[str] = None
    merchant: Optional[str] = None

class TransactionResponse(TransactionBase):
    id: int
    user_id: int
    predicted_category: Optional[str] = None
    confidence_score: Optional[float] = None
    is_anomaly: int
    account_balance: Optional[float] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class SMSParseRequest(BaseModel):
    sms_text: str

class SMSParseResponse(BaseModel):
    amount: Optional[float] = None
    merchant: Optional[str] = None
    transaction_type: Optional[str] = None
    account_balance: Optional[float] = None
    upi_id: Optional[str] = None
    bank_name: Optional[str] = None
    parsed_successfully: bool