"""
FastAPI Dependencies
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import async_session
from app.ml.inference.model_loader import model_loader

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Database session dependency"""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

def get_ml_models():
    """ML models dependency"""
    return {
        'classifier': model_loader.get_category_classifier(),
        'predictor': model_loader.get_expense_predictor(),
        'detector': model_loader.get_anomaly_detector()
    }