"""Database models initialization."""

# Import all models to ensure they're registered with SQLAlchemy
from .user import User
from .transaction import Transaction
from .budget import Budget, Income, Necessity, SavingsGoal
from .travel import TravelPlan, TravelCostPrediction

__all__ = [
    "User",
    "Transaction", 
    "Budget",
    "Income",
    "Necessity",
    "SavingsGoal",
    "TravelPlan",
    "TravelCostPrediction"
]