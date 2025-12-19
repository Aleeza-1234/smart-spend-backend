"""
Budget API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.api.deps import get_db
from app.schemas.budget import (
    BudgetCreate,
    BudgetResponse,
    BudgetUpdate,
    DailyBudgetResponse,
    IncomeCreate,
    IncomeResponse,
    NecessityCreate,
    NecessityResponse
)
from app.models.budget import Budget, Income, Necessity
from app.services.budget_calculator import BudgetCalculator
from sqlalchemy import select, and_

router = APIRouter()

@router.get("/daily", response_model=DailyBudgetResponse)
async def get_daily_budget(
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate daily budget for user
    """
    calculator = BudgetCalculator(db)
    budget_info = await calculator.calculate_daily_budget(user_id)
    
    return DailyBudgetResponse(**budget_info)

@router.get("/alerts")
async def get_budget_alerts(
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get budget alerts and warnings
    """
    calculator = BudgetCalculator(db)
    alerts = await calculator.check_budget_alerts(user_id)
    
    return {"alerts": alerts}

@router.get("/spending-by-category")
async def get_category_spending(
    user_id: int = Query(..., description="User ID"),
    days: int = Query(30, description="Days to analyze"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get spending breakdown by category
    """
    calculator = BudgetCalculator(db)
    spending = await calculator.get_category_spending(user_id, days)
    
    return {"spending_by_category": spending, "days": days}

@router.get("/trend")
async def get_spending_trend(
    user_id: int = Query(..., description="User ID"),
    months: int = Query(3, le=12, description="Months to analyze"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get monthly spending trend
    """
    calculator = BudgetCalculator(db)
    trend = await calculator.get_spending_trend(user_id, months)
    
    return {"trend": trend}

# Budget CRUD
@router.post("/budgets", response_model=BudgetResponse)
async def create_budget(
    budget: BudgetCreate,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """Create a category budget"""
    db_budget = Budget(user_id=user_id, **budget.dict())
    db.add(db_budget)
    await db.commit()
    await db.refresh(db_budget)
    return db_budget

@router.get("/budgets", response_model=List[BudgetResponse])
async def get_budgets(
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """Get all budgets"""
    stmt = select(Budget).where(Budget.user_id == user_id)
    result = await db.execute(stmt)
    return result.scalars().all()

# Income CRUD
@router.post("/income", response_model=IncomeResponse)
async def add_income(
    income: IncomeCreate,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """Add income source"""
    db_income = Income(user_id=user_id, **income.dict())
    db.add(db_income)
    await db.commit()
    await db.refresh(db_income)
    return db_income

@router.get("/income", response_model=List[IncomeResponse])
async def get_income(
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """Get all income sources"""
    stmt = select(Income).where(Income.user_id == user_id)
    result = await db.execute(stmt)
    return result.scalars().all()

# Necessities CRUD
@router.post("/necessities", response_model=NecessityResponse)
async def add_necessity(
    necessity: NecessityCreate,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """Add fixed expense"""
    db_necessity = Necessity(user_id=user_id, **necessity.dict())
    db.add(db_necessity)
    await db.commit()
    await db.refresh(db_necessity)
    return db_necessity

@router.get("/necessities", response_model=List[NecessityResponse])
async def get_necessities(
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """Get all fixed expenses"""
    stmt = select(Necessity).where(Necessity.user_id == user_id)
    result = await db.execute(stmt)
    return result.scalars().all()