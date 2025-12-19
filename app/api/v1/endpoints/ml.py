"""
ML Model API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import Optional
from datetime import datetime

from app.api.deps import get_db, get_ml_models
from app.schemas.ml import (
    CategoryPredictionRequest,
    CategoryPredictionResponse,
    ExpensePredictionRequest,
    ExpensePredictionResponse,
    AnomalyDetectionRequest,
    AnomalyDetectionResponse
)
from app.models.transaction import Transaction
import pandas as pd

router = APIRouter()

@router.post("/predict-category", response_model=CategoryPredictionResponse)
async def predict_category(
    request: CategoryPredictionRequest,
    ml_models: dict = Depends(get_ml_models)
):
    """
    Predict transaction category using ML
    """
    classifier = ml_models['classifier']
    
    if not classifier:
        raise HTTPException(status_code=503, detail="Category classifier not loaded")
    
    try:
        predicted_cat, confidence, all_probs = classifier.predict(
            merchant=request.merchant or "",
            amount=request.amount,
            raw_sms=request.transaction_text or ""
        )
        
        return CategoryPredictionResponse(
            predicted_category=predicted_cat,
            confidence=confidence,
            all_probabilities=all_probs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/predict-expense", response_model=ExpensePredictionResponse)
async def predict_expense(
    request: ExpensePredictionRequest,
    ml_models: dict = Depends(get_ml_models),
    db: AsyncSession = Depends(get_db)
):
    """
    Predict future expenses using XGBoost
    """
    predictor = ml_models['predictor']
    
    if not predictor:
        raise HTTPException(status_code=503, detail="Expense predictor not loaded")
    
    try:
        # Get user's historical transactions
        stmt = select(Transaction).where(
            Transaction.user_id == request.user_id
        )
        result = await db.execute(stmt)
        transactions = result.scalars().all()
        
        if len(transactions) < 10:
            raise HTTPException(
                status_code=400, 
                detail="Need at least 10 historical transactions for prediction"
            )
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'user_id': t.user_id,
            'amount': t.amount,
            'category': t.category,
            'timestamp': t.timestamp
        } for t in transactions])
        
        # Predict
        prediction = predictor.predict(
            user_transactions=df,
            category=request.category,
            month=request.month,
            year=request.year
        )
        
        return ExpensePredictionResponse(
            predicted_amount=prediction['predicted_amount'],
            confidence_interval=(
                prediction['confidence_lower'],
                prediction['confidence_upper']
            ),
            historical_average=prediction['historical_average']
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/detect-anomaly", response_model=AnomalyDetectionResponse)
async def detect_anomaly(
    request: AnomalyDetectionRequest,
    ml_models: dict = Depends(get_ml_models)
):
    """
    Detect if transaction is anomalous
    """
    detector = ml_models['detector']
    
    if not detector:
        raise HTTPException(status_code=503, detail="Anomaly detector not loaded")
    
    try:
        is_anomaly, score, reason = detector.detect(
            user_id=request.user_id,
            amount=request.amount,
            category=request.category,
            merchant=request.merchant
        )
        
        return AnomalyDetectionResponse(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            reason=reason
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@router.get("/model-stats")
async def get_model_stats(
    ml_models: dict = Depends(get_ml_models)
):
    """
    Get statistics about loaded models
    """
    stats = {
        "classifier": {
            "loaded": ml_models['classifier'] is not None,
            "model_type": "RandomForest",
            "categories": list(ml_models['classifier'].label_encoder.classes_) if ml_models['classifier'] else []
        },
        "predictor": {
            "loaded": ml_models['predictor'] is not None,
            "model_type": "XGBoost",
            "features": len(ml_models['predictor'].feature_names) if ml_models['predictor'] else 0
        },
        "detector": {
            "loaded": ml_models['detector'] is not None,
            "model_type": "IsolationForest",
            "categories_trained": len(ml_models['detector'].category_stats) if ml_models['detector'] else 0
        }
    }
    
    return stats

@router.post("/retrain-models")
async def retrain_models(
    user_id: int = Query(..., description="User ID for personalized retraining"),
    db: AsyncSession = Depends(get_db),
    ml_models: dict = Depends(get_ml_models)
):
    """
    Trigger model retraining (for personalized models)
    """
    # Get user's transactions
    stmt = select(Transaction).where(Transaction.user_id == user_id)
    result = await db.execute(stmt)
    transactions = result.scalars().all()
    
    if len(transactions) < 50:
        raise HTTPException(
            status_code=400,
            detail="Need at least 50 transactions for retraining"
        )
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'user_id': t.user_id,
        'amount': t.amount,
        'merchant': t.merchant,
        'category': t.category,
        'timestamp': t.timestamp,
        'raw_sms': t.raw_sms
    } for t in transactions])
    
    results = {}
    
    # Retrain classifier
    if ml_models['classifier']:
        try:
            metrics = ml_models['classifier'].train(df)
            results['classifier'] = metrics
        except Exception as e:
            results['classifier'] = {"error": str(e)}
    
    # Retrain predictor
    if ml_models['predictor']:
        try:
            metrics = ml_models['predictor'].train(df)
            results['predictor'] = metrics
        except Exception as e:
            results['predictor'] = {"error": str(e)}
    
    # Retrain detector
    if ml_models['detector']:
        try:
            metrics = ml_models['detector'].train(df)
            results['detector'] = metrics
        except Exception as e:
            results['detector'] = {"error": str(e)}
    
    return {
        "message": "Retraining completed",
        "results": results
    }