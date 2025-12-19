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
    NecessityResponse,
    SavingsGoalCreate,
    SavingsGoalResponse,
    SavingsGoalUpdate
)
from app.models.budget import Budget, Income, Necessity, SavingsGoal
from app.services.budget_calculator import BudgetCalculator
from app.core.datetime_utils import convert_timezone_aware_datetimes
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
    income_data = income.dict()
    income_data = convert_timezone_aware_datetimes(income_data)
    
    db_income = Income(user_id=user_id, **income_data)
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

# Savings Goals CRUD
@router.post("/savings-goals", response_model=SavingsGoalResponse)
async def create_savings_goal(
    goal: SavingsGoalCreate,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """Create a savings goal"""
    goal_data = goal.dict()
    goal_data = convert_timezone_aware_datetimes(goal_data)
    
    db_goal = SavingsGoal(user_id=user_id, **goal_data)
    db.add(db_goal)
    await db.commit()
    await db.refresh(db_goal)
    
    # Calculate progress percentage and monthly required
    progress_percentage = (db_goal.current_amount / db_goal.target_amount) * 100 if db_goal.target_amount > 0 else 0
    monthly_required = 0
    if db_goal.target_date:
        from datetime import datetime
        months_left = max(1, (db_goal.target_date.year - datetime.now().year) * 12 + 
                         (db_goal.target_date.month - datetime.now().month))
        remaining = max(0, db_goal.target_amount - db_goal.current_amount)
        monthly_required = remaining / months_left
    
    # Add calculated fields to response
    response_data = db_goal.__dict__.copy()
    response_data['progress_percentage'] = round(progress_percentage, 1)
    response_data['monthly_required'] = round(monthly_required, 2) if monthly_required > 0 else None
    
    return SavingsGoalResponse(**response_data)

@router.get("/savings-goals", response_model=List[SavingsGoalResponse])
async def get_savings_goals(
    user_id: int = Query(..., description="User ID"),
    active_only: bool = Query(True, description="Show only active goals"),
    db: AsyncSession = Depends(get_db)
):
    """Get all savings goals"""
    query = select(SavingsGoal).where(SavingsGoal.user_id == user_id)
    if active_only:
        query = query.where(SavingsGoal.is_active == True)
    query = query.order_by(SavingsGoal.priority.asc(), SavingsGoal.created_at.desc())
    
    result = await db.execute(query)
    goals = result.scalars().all()
    
    # Calculate progress and monthly required for each goal
    response_goals = []
    for goal in goals:
        progress_percentage = (goal.current_amount / goal.target_amount) * 100 if goal.target_amount > 0 else 0
        monthly_required = 0
        if goal.target_date:
            from datetime import datetime
            months_left = max(1, (goal.target_date.year - datetime.now().year) * 12 + 
                             (goal.target_date.month - datetime.now().month))
            remaining = max(0, goal.target_amount - goal.current_amount)
            monthly_required = remaining / months_left
        
        response_data = goal.__dict__.copy()
        response_data['progress_percentage'] = round(progress_percentage, 1)
        response_data['monthly_required'] = round(monthly_required, 2) if monthly_required > 0 else None
        response_goals.append(SavingsGoalResponse(**response_data))
    
    return response_goals

@router.put("/savings-goals/{goal_id}/add-money")
async def add_money_to_goal(
    goal_id: int,
    amount: float = Query(..., gt=0, description="Amount to add"),
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """Add money to a savings goal"""
    stmt = select(SavingsGoal).where(
        and_(
            SavingsGoal.id == goal_id,
            SavingsGoal.user_id == user_id
        )
    )
    result = await db.execute(stmt)
    goal = result.scalar_one_or_none()
    
    if not goal:
        raise HTTPException(status_code=404, detail="Savings goal not found")
    
    goal.current_amount += amount
    goal.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(goal)
    
    return {"message": f"Added ₹{amount:.2f} to {goal.name}", "new_balance": goal.current_amount}