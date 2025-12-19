"""
Budget Calculator Service
Calculates daily budgets based on income, expenses, and savings goals
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from calendar import monthrange
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from app.models.budget import Budget, Income, Necessity
from app.models.transaction import Transaction
from app.models.travel import TravelPlan

class BudgetCalculator:
    """
    Calculates daily budgets and spending recommendations
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_monthly_income(self, user_id: int, month: int = None, year: int = None) -> float:
        """
        Calculate total monthly income from all sources
        """
        if month is None:
            month = datetime.now().month
        if year is None:
            year = datetime.now().year
        
        # Get recurring income (salary, pocket money)
        stmt = select(func.sum(Income.amount)).where(
            and_(
                Income.user_id == user_id,
                Income.is_recurring == True
            )
        )
        result = await self.db.execute(stmt)
        recurring_income = result.scalar() or 0.0
        
        # Get one-time income for this month (project payments)
        start_date = datetime(year, month, 1)
        end_date = datetime(year, month, monthrange(year, month)[1])
        
        stmt = select(func.sum(Income.amount)).where(
            and_(
                Income.user_id == user_id,
                Income.is_recurring == False,
                Income.date >= start_date,
                Income.date <= end_date
            )
        )
        result = await self.db.execute(stmt)
        one_time_income = result.scalar() or 0.0
        
        return recurring_income + one_time_income
    
    async def get_fixed_expenses(self, user_id: int) -> float:
        """
        Calculate total fixed monthly expenses (necessities)
        """
        stmt = select(func.sum(Necessity.amount)).where(
            and_(
                Necessity.user_id == user_id,
                Necessity.is_active == True,
                Necessity.frequency == 'monthly'
            )
        )
        result = await self.db.execute(stmt)
        monthly_necessities = result.scalar() or 0.0
        
        # Get weekly necessities and convert to monthly
        stmt = select(func.sum(Necessity.amount)).where(
            and_(
                Necessity.user_id == user_id,
                Necessity.is_active == True,
                Necessity.frequency == 'weekly'
            )
        )
        result = await self.db.execute(stmt)
        weekly_necessities = result.scalar() or 0.0
        weekly_to_monthly = weekly_necessities * 4
        
        return monthly_necessities + weekly_to_monthly
    
    async def get_active_savings_goal(self, user_id: int) -> Optional[Dict]:
        """
        Get the most urgent active travel savings goal
        """
        stmt = select(TravelPlan).where(
            and_(
                TravelPlan.user_id == user_id,
                TravelPlan.is_completed == False
            )
        ).order_by(TravelPlan.target_date.asc())
        
        result = await self.db.execute(stmt)
        travel_plan = result.scalar_one_or_none()
        
        if not travel_plan:
            return None
        
        remaining = travel_plan.estimated_cost - travel_plan.current_savings
        
        # Calculate months until target date
        if travel_plan.target_date:
            today = datetime.now().date()
            months_left = max(1, (travel_plan.target_date.year - today.year) * 12 + 
                            (travel_plan.target_date.month - today.month))
        else:
            months_left = 6  # Default to 6 months if no target date
        
        monthly_savings_needed = remaining / months_left
        
        return {
            'id': travel_plan.id,
            'destination': travel_plan.destination,
            'total_cost': travel_plan.estimated_cost,
            'current_savings': travel_plan.current_savings,
            'remaining': remaining,
            'months_left': months_left,
            'monthly_savings_needed': monthly_savings_needed
        }
    
    async def get_current_month_spending(self, user_id: int) -> float:
        """
        Get total spending for current month
        """
        now = datetime.now()
        start_of_month = datetime(now.year, now.month, 1)
        
        stmt = select(func.sum(Transaction.amount)).where(
            and_(
                Transaction.user_id == user_id,
                Transaction.transaction_type == 'DEBIT',
                Transaction.timestamp >= start_of_month
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar() or 0.0
    
    async def calculate_daily_budget(self, user_id: int) -> Dict:
        """
        Calculate daily budget breakdown
        
        Formula:
        Disposable Income = Monthly Income - Fixed Expenses - Monthly Savings Goal
        Daily Budget = Disposable Income / Days Remaining in Month
        """
        now = datetime.now()
        days_in_month = monthrange(now.year, now.month)[1]
        days_remaining = days_in_month - now.day + 1
        
        # Get income
        monthly_income = await self.get_monthly_income(user_id)
        
        # Get fixed expenses
        fixed_expenses = await self.get_fixed_expenses(user_id)
        
        # Get savings goal
        savings_goal = await self.get_active_savings_goal(user_id)
        monthly_savings = savings_goal['monthly_savings_needed'] if savings_goal else 0.0
        
        # Calculate disposable income
        disposable = monthly_income - fixed_expenses - monthly_savings
        
        # Already spent this month
        spent_this_month = await self.get_current_month_spending(user_id)
        
        # Adjust for overspending
        adjusted_disposable = max(0, disposable - spent_this_month)
        
        # Daily budgets
        daily_total = adjusted_disposable / days_remaining if days_remaining > 0 else 0
        food_budget = daily_total * 0.6  # 60% for food
        discretionary_budget = daily_total * 0.4  # 40% for other
        
        return {
            'date': now,
            'monthly_income': round(monthly_income, 2),
            'fixed_expenses': round(fixed_expenses, 2),
            'savings_goal': round(monthly_savings, 2),
            'disposable_income': round(disposable, 2),
            'spent_this_month': round(spent_this_month, 2),
            'remaining_for_month': round(adjusted_disposable, 2),
            'days_remaining': days_remaining,
            'daily_total': round(daily_total, 2),
            'food_budget': round(food_budget, 2),
            'discretionary_budget': round(discretionary_budget, 2),
            'savings_goal_details': savings_goal
        }
    
    async def get_category_spending(self, user_id: int, days: int = 30) -> Dict[str, float]:
        """
        Get spending breakdown by category for last N days
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
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
        
        spending_by_category = {}
        for row in result:
            if row.category:
                spending_by_category[row.category] = round(row.total, 2)
        
        return spending_by_category
    
    async def check_budget_alerts(self, user_id: int) -> List[Dict]:
        """
        Check if user is exceeding budgets and return alerts
        """
        alerts = []
        
        # Get active budgets
        stmt = select(Budget).where(
            and_(
                Budget.user_id == user_id,
                Budget.is_active == True
            )
        )
        result = await self.db.execute(stmt)
        budgets = result.scalars().all()
        
        for budget in budgets:
            percentage_used = (budget.current_spent / budget.monthly_limit) * 100
            
            if percentage_used >= 100:
                alerts.append({
                    'type': 'BUDGET_EXCEEDED',
                    'category': budget.category,
                    'limit': budget.monthly_limit,
                    'spent': budget.current_spent,
                    'percentage': round(percentage_used, 1),
                    'message': f"You've exceeded your {budget.category} budget by ₹{budget.current_spent - budget.monthly_limit:.2f}"
                })
            elif percentage_used >= 80:
                alerts.append({
                    'type': 'BUDGET_WARNING',
                    'category': budget.category,
                    'limit': budget.monthly_limit,
                    'spent': budget.current_spent,
                    'percentage': round(percentage_used, 1),
                    'message': f"You've used {percentage_used:.0f}% of your {budget.category} budget"
                })
        
        # Check daily budget
        daily_budget_info = await self.calculate_daily_budget(user_id)
        if daily_budget_info['daily_total'] < 0:
            alerts.append({
                'type': 'OVERSPENDING',
                'message': f"You've overspent this month by ₹{abs(daily_budget_info['remaining_for_month']):.2f}"
            })
        
        return alerts
    
    async def get_spending_trend(self, user_id: int, months: int = 3) -> List[Dict]:
        """
        Get monthly spending trend for last N months
        """
        trends = []
        
        for i in range(months):
            # Calculate date range
            now = datetime.now()
            target_month = now.month - i
            target_year = now.year
            
            if target_month <= 0:
                target_month += 12
                target_year -= 1
            
            start_date = datetime(target_year, target_month, 1)
            end_date = datetime(target_year, target_month, monthrange(target_year, target_month)[1])
            
            # Get spending for month
            stmt = select(func.sum(Transaction.amount)).where(
                and_(
                    Transaction.user_id == user_id,
                    Transaction.transaction_type == 'DEBIT',
                    Transaction.timestamp >= start_date,
                    Transaction.timestamp <= end_date
                )
            )
            result = await self.db.execute(stmt)
            total_spent = result.scalar() or 0.0
            
            trends.append({
                'month': target_month,
                'year': target_year,
                'month_name': start_date.strftime('%B'),
                'total_spent': round(total_spent, 2)
            })
        
        return list(reversed(trends))