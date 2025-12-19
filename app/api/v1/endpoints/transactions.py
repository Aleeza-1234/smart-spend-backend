"""
Transaction API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from typing import List, Optional
from datetime import datetime, timedelta

from app.api.deps import get_db, get_ml_models
from app.schemas.transaction import (
    TransactionCreate,
    TransactionResponse,
    TransactionUpdate,
    SMSParseRequest,
    SMSParseResponse
)
from app.models.transaction import Transaction
from app.services.sms_parser import SMSParser

router = APIRouter()
sms_parser = SMSParser()

@router.post("/parse-sms", response_model=SMSParseResponse)
async def parse_sms(
    request: SMSParseRequest,
    ml_models: dict = Depends(get_ml_models)
):
    """
    Parse SMS and extract transaction information
    """
    result = sms_parser.parse(request.sms_text)
    
    # If parsing successful and we have a classifier, predict category
    if result['parsed_successfully'] and result['merchant'] and ml_models['classifier']:
        try:
            predicted_cat, confidence, _ = ml_models['classifier'].predict(
                merchant=result['merchant'],
                amount=result['amount'],
                raw_sms=request.sms_text
            )
            result['predicted_category'] = predicted_cat
            result['confidence'] = confidence
        except Exception as e:
            print(f"Category prediction error: {e}")
    
    return SMSParseResponse(**result)

@router.post("/", response_model=TransactionResponse)
async def create_transaction(
    transaction: TransactionCreate,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db),
    ml_models: dict = Depends(get_ml_models)
):
    """
    Create a new transaction
    """
    # Create transaction object
    db_transaction = Transaction(
        user_id=user_id,
        **transaction.dict()
    )
    
    # Predict category if not provided
    if not db_transaction.category and ml_models['classifier'] and db_transaction.merchant:
        try:
            predicted_cat, confidence, _ = ml_models['classifier'].predict(
                merchant=db_transaction.merchant,
                amount=db_transaction.amount,
                raw_sms=db_transaction.raw_sms or ""
            )
            db_transaction.predicted_category = predicted_cat
            db_transaction.confidence_score = confidence
            
            # Auto-assign if confidence > 0.8
            if confidence > 0.8:
                db_transaction.category = predicted_cat
        except Exception as e:
            print(f"Category prediction error: {e}")
    
    # Check for anomaly
    if ml_models['detector'] and db_transaction.category:
        try:
            is_anomaly, score, reason = ml_models['detector'].detect(
                user_id=user_id,
                amount=db_transaction.amount,
                category=db_transaction.category,
                merchant=db_transaction.merchant
            )
            db_transaction.is_anomaly = 1 if is_anomaly else 0
        except Exception as e:
            print(f"Anomaly detection error: {e}")
    
    db.add(db_transaction)
    await db.commit()
    await db.refresh(db_transaction)
    
    return db_transaction

@router.get("/", response_model=List[TransactionResponse])
async def get_transactions(
    user_id: int = Query(..., description="User ID"),
    category: Optional[str] = None,
    transaction_type: Optional[str] = None,
    days: int = Query(30, description="Days to look back"),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user transactions with filters
    """
    cutoff_date = datetime.now() - timedelta(days=days)
    
    # Build query
    query = select(Transaction).where(
        and_(
            Transaction.user_id == user_id,
            Transaction.timestamp >= cutoff_date
        )
    )
    
    # Apply filters
    if category:
        query = query.where(Transaction.category == category)
    
    if transaction_type:
        query = query.where(Transaction.transaction_type == transaction_type)
    
    # Order and paginate
    query = query.order_by(desc(Transaction.timestamp)).offset(offset).limit(limit)
    
    result = await db.execute(query)
    transactions = result.scalars().all()
    
    return transactions

@router.get("/{transaction_id}", response_model=TransactionResponse)
async def get_transaction(
    transaction_id: int,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get specific transaction
    """
    stmt = select(Transaction).where(
        and_(
            Transaction.id == transaction_id,
            Transaction.user_id == user_id
        )
    )
    result = await db.execute(stmt)
    transaction = result.scalar_one_or_none()
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return transaction

@router.patch("/{transaction_id}", response_model=TransactionResponse)
async def update_transaction(
    transaction_id: int,
    update_data: TransactionUpdate,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Update transaction (e.g., correct category)
    """
    stmt = select(Transaction).where(
        and_(
            Transaction.id == transaction_id,
            Transaction.user_id == user_id
        )
    )
    result = await db.execute(stmt)
    transaction = result.scalar_one_or_none()
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    # Update fields
    update_dict = update_data.dict(exclude_unset=True)
    for key, value in update_dict.items():
        setattr(transaction, key, value)
    
    await db.commit()
    await db.refresh(transaction)
    
    return transaction

@router.delete("/{transaction_id}")
async def delete_transaction(
    transaction_id: int,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete transaction
    """
    stmt = select(Transaction).where(
        and_(
            Transaction.id == transaction_id,
            Transaction.user_id == user_id
        )
    )
    result = await db.execute(stmt)
    transaction = result.scalar_one_or_none()
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    await db.delete(transaction)
    await db.commit()
    
    return {"message": "Transaction deleted successfully"}