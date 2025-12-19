"""
Travel Timeline Predictor Service
Predicts how many months needed to save for a trip
"""

from datetime import datetime, timedelta, date
from typing import Dict, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from app.models.transaction import Transaction
from app.models.budget import Income, Necessity
from app.models.travel import TravelPlan, TravelCostPrediction

class TravelPredictor:
    """
    Predicts travel timeline based on savings rate and travel costs
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def calculate_historical_savings_rate(
        self, 
        user_id: int, 
        months: int = 3
    ) -> float:
        """
        Calculate average monthly savings from historical data
        Savings = Income - Expenses
        """
        savings_by_month = []
        
        for i in range(months):
            # Calculate date range for each month
            target_date = datetime.now() - timedelta(days=30 * i)
            month = target_date.month
            year = target_date.year
            
            from calendar import monthrange
            days_in_month = monthrange(year, month)[1]
            
            start_date = datetime(year, month, 1)
            end_date = datetime(year, month, days_in_month, 23, 59, 59)
            
            # Get income for the month
            stmt = select(func.sum(Income.amount)).where(
                and_(
                    Income.user_id == user_id,
                    Income.date >= start_date,
                    Income.date <= end_date
                )
            )
            result = await self.db.execute(stmt)
            monthly_income = result.scalar() or 0.0
            
            # Get expenses for the month
            stmt = select(func.sum(Transaction.amount)).where(
                and_(
                    Transaction.user_id == user_id,
                    Transaction.transaction_type == 'DEBIT',
                    Transaction.timestamp >= start_date,
                    Transaction.timestamp <= end_date
                )
            )
            result = await self.db.execute(stmt)
            monthly_expenses = result.scalar() or 0.0
            
            # Calculate savings
            savings = monthly_income - monthly_expenses
            savings_by_month.append(savings)
        
        # Return average savings
        if savings_by_month:
            return sum(savings_by_month) / len(savings_by_month)
        return 0.0
    
    async def predict_monthly_savings_rate(
        self, 
        user_id: int
    ) -> Dict[str, float]:
        """
        Predict future savings rate considering:
        - Historical savings
        - Recurring income
        - Fixed expenses
        - Variable income patterns
        """
        # Historical savings rate
        historical_rate = await self.calculate_historical_savings_rate(user_id, months=3)
        
        # Get recurring income
        stmt = select(func.sum(Income.amount)).where(
            and_(
                Income.user_id == user_id,
                Income.is_recurring == True
            )
        )
        result = await self.db.execute(stmt)
        recurring_income = result.scalar() or 0.0
        
        # Get fixed expenses
        stmt = select(func.sum(Necessity.amount)).where(
            and_(
                Necessity.user_id == user_id,
                Necessity.is_active == True,
                Necessity.frequency == 'monthly'
            )
        )
        result = await self.db.execute(stmt)
        fixed_expenses = result.scalar() or 0.0
        
        # Calculate average variable expenses (last 3 months)
        cutoff_date = datetime.now() - timedelta(days=90)
        stmt = select(func.avg(Transaction.amount)).where(
            and_(
                Transaction.user_id == user_id,
                Transaction.transaction_type == 'DEBIT',
                Transaction.timestamp >= cutoff_date,
                Transaction.category.notin_(['rent', 'bills'])  # Exclude fixed categories
            )
        )
        result = await self.db.execute(stmt)
        avg_variable_expense = result.scalar() or 0.0
        
        # Count transactions per month
        stmt = select(func.count(Transaction.id)).where(
            and_(
                Transaction.user_id == user_id,
                Transaction.transaction_type == 'DEBIT',
                Transaction.timestamp >= cutoff_date
            )
        )
        result = await self.db.execute(stmt)
        total_transactions = result.scalar() or 1
        transactions_per_month = total_transactions / 3
        
        # Estimate monthly variable expenses
        estimated_variable = avg_variable_expense * transactions_per_month
        
        # Predicted savings rate
        predicted_rate = recurring_income - fixed_expenses - estimated_variable
        
        # Weighted average (70% historical, 30% predicted)
        if historical_rate > 0:
            final_rate = (0.7 * historical_rate) + (0.3 * predicted_rate)
        else:
            final_rate = predicted_rate
        
        return {
            'predicted_monthly_savings': round(max(0, final_rate), 2),
            'historical_rate': round(historical_rate, 2),
            'recurring_income': round(recurring_income, 2),
            'fixed_expenses': round(fixed_expenses, 2),
            'estimated_variable_expenses': round(estimated_variable, 2),
            'confidence': 0.75 if historical_rate > 0 else 0.5
        }
    
    async def estimate_travel_cost(
        self, 
        destination: str, 
        duration_days: int, 
        travel_style: str = 'mid'
    ) -> Dict[str, float]:
        """
        Estimate travel costs based on destination and style
        
        Travel Style Multipliers:
        - budget: 0.7x
        - mid: 1.0x
        - luxury: 1.5x
        """
        # Base costs per day for different destination types
        destination_lower = destination.lower()
        
        # Categorize destinations
        if any(city in destination_lower for city in ['goa', 'kerala', 'manali', 'shimla', 'rishikesh']):
            base_cost_per_day = 2000  # Domestic popular
        elif any(city in destination_lower for city in ['mumbai', 'delhi', 'bangalore', 'kolkata']):
            base_cost_per_day = 2500  # Metro cities
        elif any(city in destination_lower for city in ['ladakh', 'andaman', 'kashmir']):
            base_cost_per_day = 3000  # Premium domestic
        elif any(country in destination_lower for country in ['thailand', 'vietnam', 'bali', 'sri lanka']):
            base_cost_per_day = 3500  # Budget international
        elif any(country in destination_lower for country in ['dubai', 'singapore', 'malaysia']):
            base_cost_per_day = 5000  # Mid international
        else:
            base_cost_per_day = 2500  # Default
        
        # Apply style multiplier
        style_multipliers = {
            'budget': 0.7,
            'mid': 1.0,
            'luxury': 1.5
        }
        multiplier = style_multipliers.get(travel_style.lower(), 1.0)
        
        cost_per_day = base_cost_per_day * multiplier
        
        # Calculate components
        accommodation = cost_per_day * 0.4 * duration_days
        food = cost_per_day * 0.3 * duration_days
        transport_local = cost_per_day * 0.2 * duration_days
        activities = cost_per_day * 0.1 * duration_days
        
        # Add travel to destination (estimated)
        if 'international' in destination_lower or any(c in destination_lower for c in ['thailand', 'dubai', 'singapore']):
            travel_to_destination = 25000 * multiplier
        else:
            travel_to_destination = 8000 * multiplier
        
        total_cost = accommodation + food + transport_local + activities + travel_to_destination
        
        # Add 10% buffer for miscellaneous
        total_with_buffer = total_cost * 1.1
        
        return {
            'total_estimated_cost': round(total_with_buffer, 2),
            'accommodation': round(accommodation, 2),
            'food': round(food, 2),
            'local_transport': round(transport_local, 2),
            'activities': round(activities, 2),
            'travel_to_destination': round(travel_to_destination, 2),
            'miscellaneous': round(total_with_buffer - total_cost, 2),
            'cost_per_day': round(cost_per_day, 2)
        }
    
    async def predict_timeline(
        self, 
        user_id: int, 
        travel_plan_id: int
    ) -> Dict:
        """
        Predict how many months needed to save for a trip
        
        Returns:
            Timeline prediction with confidence intervals
        """
        # Get travel plan
        stmt = select(TravelPlan).where(TravelPlan.id == travel_plan_id)
        result = await self.db.execute(stmt)
        travel_plan = result.scalar_one_or_none()
        
        if not travel_plan:
            raise ValueError(f"Travel plan {travel_plan_id} not found")
        
        # Calculate remaining amount needed
        remaining = travel_plan.estimated_cost - travel_plan.current_savings
        
        # Get predicted savings rate
        savings_info = await self.predict_monthly_savings_rate(user_id)
        monthly_savings = savings_info['predicted_monthly_savings']
        
        if monthly_savings <= 0:
            return {
                'travel_plan_id': travel_plan_id,
                'destination': travel_plan.destination,
                'error': 'Current spending pattern shows no savings. Please reduce expenses or increase income.',
                'months_needed': None,
                'target_date': None
            }
        
        # Calculate months needed
        months_needed = remaining / monthly_savings
        
        # Calculate confidence interval (±20% based on income variability)
        confidence_range = 0.2
        lower_months = months_needed * (1 - confidence_range)
        upper_months = months_needed * (1 + confidence_range)
        
        # Calculate target date
        today = datetime.now().date()
        target_date = today + timedelta(days=int(months_needed * 30))
        lower_date = today + timedelta(days=int(lower_months * 30))
        upper_date = today + timedelta(days=int(upper_months * 30))
        
        # Calculate weekly/monthly savings breakdown
        weekly_savings = monthly_savings / 4
        daily_savings = monthly_savings / 30
        
        # Generate milestones
        milestones = []
        for pct in [25, 50, 75]:
            milestone_amount = travel_plan.estimated_cost * (pct / 100)
            if milestone_amount > travel_plan.current_savings:
                months_to_milestone = (milestone_amount - travel_plan.current_savings) / monthly_savings
                milestone_date = today + timedelta(days=int(months_to_milestone * 30))
                milestones.append({
                    'percentage': pct,
                    'amount': round(milestone_amount, 2),
                    'months_away': round(months_to_milestone, 1),
                    'date': milestone_date.isoformat()
                })
        
        prediction = {
            'travel_plan_id': travel_plan_id,
            'destination': travel_plan.destination,
            'duration_days': travel_plan.duration_days,
            'travel_style': travel_plan.travel_style,
            
            # Financial details
            'total_cost': travel_plan.estimated_cost,
            'current_savings': travel_plan.current_savings,
            'remaining_needed': round(remaining, 2),
            
            # Timeline prediction
            'months_needed': round(months_needed, 1),
            'target_date': target_date.isoformat(),
            
            # Confidence interval
            'best_case_months': round(lower_months, 1),
            'worst_case_months': round(upper_months, 1),
            'best_case_date': lower_date.isoformat(),
            'worst_case_date': upper_date.isoformat(),
            
            # Savings breakdown
            'monthly_savings_needed': round(monthly_savings, 2),
            'weekly_savings_needed': round(weekly_savings, 2),
            'daily_savings_needed': round(daily_savings, 2),
            
            # Milestones
            'milestones': milestones,
            
            # Additional context
            'savings_rate_info': savings_info,
            'confidence': savings_info['confidence']
        }
        
        # Save prediction to database
        db_prediction = TravelCostPrediction(
            travel_plan_id=travel_plan_id,
            predicted_months=months_needed,
            confidence_lower=lower_months,
            confidence_upper=upper_months,
            monthly_savings_needed=monthly_savings,
            predicted_date=target_date
        )
        self.db.add(db_prediction)
        await self.db.commit()
        
        return prediction
    
    async def suggest_savings_strategies(
        self, 
        user_id: int, 
        target_monthly_savings: float
    ) -> Dict:
        """
        Suggest ways to reach target savings
        """
        # Get current spending by category
        cutoff_date = datetime.now() - timedelta(days=30)
        stmt = select(
            Transaction.category,
            func.sum(Transaction.amount).label('total')
        ).where(
            and_(
                Transaction.user_id == user_id,
                Transaction.transaction_type == 'DEBIT',
                Transaction.timestamp >= cutoff_date
            )
        ).group_by(Transaction.category)
        
        result = await self.db.execute(stmt)
        category_spending = {row.category: row.total for row in result}
        
        # Get current savings rate
        savings_info = await self.predict_monthly_savings_rate(user_id)
        current_savings = savings_info['predicted_monthly_savings']
        
        gap = target_monthly_savings - current_savings
        
        if gap <= 0:
            return {
                'message': 'You are already saving enough!',
                'current_savings': current_savings,
                'target_savings': target_monthly_savings,
                'suggestions': []
            }
        
        # Generate suggestions
        suggestions = []
        
        # Analyze discretionary spending
        discretionary_categories = ['Food', 'Entertainment', 'Shopping']
        for category in discretionary_categories:
            if category in category_spending:
                amount = category_spending[category]
                # Suggest 20% reduction
                potential_savings = amount * 0.2
                if potential_savings >= gap * 0.3:  # If it covers 30% of gap
                    suggestions.append({
                        'category': category,
                        'current_spending': round(amount, 2),
                        'suggested_reduction': round(potential_savings, 2),
                        'tip': f"Reduce {category} spending by 20% (₹{potential_savings:.0f}/month)"
                    })
        
        # Suggest skipping certain expenses
        if 'Entertainment' in category_spending:
            suggestions.append({
                'category': 'Entertainment',
                'tip': f"Skip 2-3 restaurant visits or movie outings this month to save ₹{gap/2:.0f}"
            })
        
        # Generic tips
        suggestions.append({
            'tip': f"Pack lunch from home 3 days a week to save ₹{gap*0.3:.0f}/month"
        })
        
        suggestions.append({
            'tip': f"Use public transport instead of cabs to save ₹{gap*0.2:.0f}/month"
        })
        
        return {
            'current_savings': round(current_savings, 2),
            'target_savings': round(target_monthly_savings, 2),
            'gap': round(gap, 2),
            'suggestions': suggestions
        }