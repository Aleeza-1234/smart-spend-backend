"""
Travel Planning API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import List

from app.api.deps import get_db
from app.schemas.travel import (
    TravelPlanCreate,
    TravelPlanResponse,
    TravelPlanUpdate,
    TravelTimelinePrediction
)
from app.models.travel import TravelPlan
from app.services.travel_predictor import TravelPredictor

router = APIRouter()

@router.post("/", response_model=TravelPlanResponse)
async def create_travel_plan(
    plan: TravelPlanCreate,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new travel plan
    """
    # Estimate cost if not provided
    predictor = TravelPredictor(db)
    
    if not plan.estimated_cost:
        cost_estimate = await predictor.estimate_travel_cost(
            destination=plan.destination,
            duration_days=plan.duration_days,
            travel_style=plan.travel_style
        )
        estimated_cost = cost_estimate['total_estimated_cost']
    else:
        estimated_cost = plan.estimated_cost
    
    db_plan = TravelPlan(
        user_id=user_id,
        destination=plan.destination,
        duration_days=plan.duration_days,
        estimated_cost=estimated_cost,
        travel_style=plan.travel_style,
        target_date=plan.target_date
    )
    
    db.add(db_plan)
    await db.commit()
    await db.refresh(db_plan)
    
    return db_plan

@router.get("/", response_model=List[TravelPlanResponse])
async def get_travel_plans(
    user_id: int = Query(..., description="User ID"),
    active_only: bool = Query(True, description="Show only active plans"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all travel plans for user
    """
    query = select(TravelPlan).where(TravelPlan.user_id == user_id)
    
    if active_only:
        query = query.where(TravelPlan.is_completed == False)
    
    query = query.order_by(TravelPlan.target_date.asc())
    
    result = await db.execute(query)
    plans = result.scalars().all()
    
    return plans

@router.get("/{plan_id}", response_model=TravelPlanResponse)
async def get_travel_plan(
    plan_id: int,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get specific travel plan
    """
    stmt = select(TravelPlan).where(
        and_(
            TravelPlan.id == plan_id,
            TravelPlan.user_id == user_id
        )
    )
    result = await db.execute(stmt)
    plan = result.scalar_one_or_none()
    
    if not plan:
        raise HTTPException(status_code=404, detail="Travel plan not found")
    
    return plan

@router.patch("/{plan_id}", response_model=TravelPlanResponse)
async def update_travel_plan(
    plan_id: int,
    update_data: TravelPlanUpdate,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Update travel plan
    """
    stmt = select(TravelPlan).where(
        and_(
            TravelPlan.id == plan_id,
            TravelPlan.user_id == user_id
        )
    )
    result = await db.execute(stmt)
    plan = result.scalar_one_or_none()
    
    if not plan:
        raise HTTPException(status_code=404, detail="Travel plan not found")
    
    # Update fields
    update_dict = update_data.dict(exclude_unset=True)
    for key, value in update_dict.items():
        setattr(plan, key, value)
    
    await db.commit()
    await db.refresh(plan)
    
    return plan

@router.delete("/{plan_id}")
async def delete_travel_plan(
    plan_id: int,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete travel plan
    """
    stmt = select(TravelPlan).where(
        and_(
            TravelPlan.id == plan_id,
            TravelPlan.user_id == user_id
        )
    )
    result = await db.execute(stmt)
    plan = result.scalar_one_or_none()
    
    if not plan:
        raise HTTPException(status_code=404, detail="Travel plan not found")
    
    await db.delete(plan)
    await db.commit()
    
    return {"message": "Travel plan deleted successfully"}

@router.get("/{plan_id}/timeline", response_model=TravelTimelinePrediction)
async def predict_timeline(
    plan_id: int,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Predict when user can afford the trip
    """
    # Verify plan exists and belongs to user
    stmt = select(TravelPlan).where(
        and_(
            TravelPlan.id == plan_id,
            TravelPlan.user_id == user_id
        )
    )
    result = await db.execute(stmt)
    plan = result.scalar_one_or_none()
    
    if not plan:
        raise HTTPException(status_code=404, detail="Travel plan not found")
    
    # Get prediction
    predictor = TravelPredictor(db)
    prediction = await predictor.predict_timeline(user_id, plan_id)
    
    return TravelTimelinePrediction(**prediction)

@router.get("/{plan_id}/cost-estimate")
async def estimate_cost(
    plan_id: int,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed cost estimate for travel plan
    """
    stmt = select(TravelPlan).where(
        and_(
            TravelPlan.id == plan_id,
            TravelPlan.user_id == user_id
        )
    )
    result = await db.execute(stmt)
    plan = result.scalar_one_or_none()
    
    if not plan:
        raise HTTPException(status_code=404, detail="Travel plan not found")
    
    predictor = TravelPredictor(db)
    cost_breakdown = await predictor.estimate_travel_cost(
        destination=plan.destination,
        duration_days=plan.duration_days,
        travel_style=plan.travel_style
    )
    
    return cost_breakdown

@router.post("/{plan_id}/add-savings")
async def add_savings(
    plan_id: int,
    amount: float = Query(..., gt=0, description="Amount to add to savings"),
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Add money to travel savings
    """
    stmt = select(TravelPlan).where(
        and_(
            TravelPlan.id == plan_id,
            TravelPlan.user_id == user_id
        )
    )
    result = await db.execute(stmt)
    plan = result.scalar_one_or_none()
    
    if not plan:
        raise HTTPException(status_code=404, detail="Travel plan not found")
    
    # Update savings
    plan.current_savings += amount
    
    # Check if goal reached
    if plan.current_savings >= plan.estimated_cost:
        plan.is_completed = True
    
    await db.commit()
    await db.refresh(plan)
    
    return {
        "message": "Savings updated successfully",
        "current_savings": plan.current_savings,
        "remaining": plan.estimated_cost - plan.current_savings,
        "goal_reached": plan.is_completed
    }

@router.get("/{plan_id}/suggestions")
async def get_savings_suggestions(
    plan_id: int,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get personalized savings suggestions
    """
    stmt = select(TravelPlan).where(
        and_(
            TravelPlan.id == plan_id,
            TravelPlan.user_id == user_id
        )
    )
    result = await db.execute(stmt)
    plan = result.scalar_one_or_none()
    
    if not plan:
        raise HTTPException(status_code=404, detail="Travel plan not found")
    
    predictor = TravelPredictor(db)
    
    # Get timeline first to know target savings
    timeline = await predictor.predict_timeline(user_id, plan_id)
    target_savings = timeline['monthly_savings_needed']
    
    # Get suggestions
    suggestions = await predictor.suggest_savings_strategies(user_id, target_savings)
    
    return suggestions