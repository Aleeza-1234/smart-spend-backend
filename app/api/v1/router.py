"""
API v1 Router
"""

from fastapi import APIRouter
from app.api.v1.endpoints import transactions, budget, travel, ml

api_router = APIRouter()

api_router.include_router(
    transactions.router,
    prefix="/transactions",
    tags=["transactions"]
)

api_router.include_router(
    budget.router,
    prefix="/budget",
    tags=["budget"]
)

api_router.include_router(
    travel.router,
    prefix="/travel",
    tags=["travel"]
)

api_router.include_router(
    ml.router,
    prefix="/ml",
    tags=["machine-learning"]
)