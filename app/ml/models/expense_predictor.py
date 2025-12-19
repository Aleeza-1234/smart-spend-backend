import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
import os

class ExpensePredictor:
    """
    Predicts monthly expenses using XGBoost with time-series features
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.feature_names = None
        self.model_path = model_path
        
        if model_path and os.path.exists(f"{model_path}/expense_predictor.json"):
            self.load_model(model_path)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-series features from transaction data
        
        Features:
        - Lag features (previous months spending)
        - Rolling statistics (moving averages)
        - Trend features
        - Seasonal features (month, day of week)
        - Category-specific features
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Extract time components
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 7).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 23).astype(int)
        
        # Aggregate by month and category
        df['year_month'] = df['timestamp'].dt.to_period('M')
        
        monthly_agg = df.groupby(['year_month', 'category']).agg({
            'amount': ['sum', 'mean', 'count', 'std']
        }).reset_index()
        
        monthly_agg.columns = ['year_month', 'category', 'total_amount', 'avg_amount', 
                               'transaction_count', 'std_amount']
        
        # Fill NaN in std (when only 1 transaction)
        monthly_agg['std_amount'] = monthly_agg['std_amount'].fillna(0)
        
        # Create lag features (previous months)
        for lag in [1, 2, 3]:
            monthly_agg[f'lag_{lag}_total'] = monthly_agg.groupby('category')['total_amount'].shift(lag)
            monthly_agg[f'lag_{lag}_count'] = monthly_agg.groupby('category')['transaction_count'].shift(lag)
        
        # Rolling features (moving averages)
        for window in [2, 3]:
            monthly_agg[f'rolling_{window}_mean'] = (
                monthly_agg.groupby('category')['total_amount']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(drop=True)
            )
            
            monthly_agg[f'rolling_{window}_std'] = (
                monthly_agg.groupby('category')['total_amount']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(drop=True)
            )
        
        # Trend features
        monthly_agg['trend'] = monthly_agg.groupby('category').cumcount()
        
        # Growth rate (month-over-month)
        monthly_agg['mom_growth'] = (
            monthly_agg.groupby('category')['total_amount'].pct_change()
        )
        
        # Extract month and year for seasonal features
        monthly_agg['month'] = monthly_agg['year_month'].dt.month
        monthly_agg['year'] = monthly_agg['year_month'].dt.year
        
        # Fill NaN values in lag and rolling features with 0
        feature_cols = [col for col in monthly_agg.columns if 
                       col.startswith('lag_') or col.startswith('rolling_') or col == 'mom_growth']
        monthly_agg[feature_cols] = monthly_agg[feature_cols].fillna(0)
        
        return monthly_agg
    
    def prepare_training_data(self, transactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features and target for training
        
        Args:
            transactions: DataFrame with columns [user_id, amount, category, timestamp]
        
        Returns:
            X (features), y (target), feature_names
        """
        # Create features
        featured_df = self.create_features(transactions)
        
        # Define feature columns
        feature_cols = [
            'month', 'year', 'trend',
            'avg_amount', 'transaction_count', 'std_amount',
            'lag_1_total', 'lag_1_count',
            'lag_2_total', 'lag_2_count',
            'lag_3_total', 'lag_3_count',
            'rolling_2_mean', 'rolling_2_std',
            'rolling_3_mean', 'rolling_3_std',
            'mom_growth'
        ]
        
        # Add category one-hot encoding
        category_dummies = pd.get_dummies(featured_df['category'], prefix='cat')
        
        X = pd.concat([
            featured_df[feature_cols],
            category_dummies
        ], axis=1)
        
        y = featured_df['total_amount']
        
        self.feature_names = X.columns.tolist()
        
        return X, y, self.feature_names
    
    def train(self, transactions: pd.DataFrame, n_splits: int = 3) -> Dict[str, float]:
        """
        Train XGBoost model with time-series cross-validation
        
        Args:
            transactions: DataFrame with transaction history
            n_splits: Number of cross-validation splits
        
        Returns:
            Dictionary with training metrics
        """
        print(f"Training with {len(transactions)} transactions...")
        
        # Prepare data
        X, y, feature_names = self.prepare_training_data(transactions)
        
        print(f"Features shape: {X.shape}")
        print(f"Feature names: {feature_names[:10]}... ({len(feature_names)} total)")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = {
            'mae': [],
            'rmse': [],
            'mape': []
        }
        
        # Train final model on all data
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation for evaluation
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Predict
            y_pred = self.model.predict(X_val)
            
            # Calculate metrics
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mape = mean_absolute_percentage_error(y_val, y_pred) * 100
            
            cv_scores['mae'].append(mae)
            cv_scores['rmse'].append(rmse)
            cv_scores['mape'].append(mape)
            
            print(f"Fold {fold + 1}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
        
        # Train on full dataset for deployment
        self.model.fit(X, y, verbose=False)
        
        # Average metrics
        metrics = {
            'mae': np.mean(cv_scores['mae']),
            'rmse': np.mean(cv_scores['rmse']),
            'mape': np.mean(cv_scores['mape']),
            'n_samples': len(transactions),
            'n_features': len(feature_names)
        }
        
        print(f"\nFinal Metrics (avg across {n_splits} folds):")
        print(f"  MAE:  {metrics['mae']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        return metrics
    
    def predict(
        self, 
        user_transactions: pd.DataFrame,
        category: Optional[str] = None,
        month: int = None,
        year: int = None
    ) -> Dict[str, float]:
        """
        Predict expense for next month or specific month
        
        Args:
            user_transactions: Historical transactions for user
            category: Specific category to predict (if None, predicts total)
            month: Target month (if None, predicts next month)
            year: Target year
        
        Returns:
            Dictionary with prediction and confidence interval
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Create features from historical data
        featured_df = self.create_features(user_transactions)
        
        if featured_df.empty:
            return {
                'predicted_amount': 0.0,
                'confidence_lower': 0.0,
                'confidence_upper': 0.0,
                'historical_average': 0.0
            }
        
        # Filter by category if specified
        if category:
            featured_df = featured_df[featured_df['category'] == category]
        
        # Get the most recent data point for each category
        latest_data = featured_df.sort_values('year_month').groupby('category').tail(1)
        
        # If month/year not specified, predict next month
        if month is None or year is None:
            last_date = featured_df['year_month'].max()
            next_date = last_date + 1
            month = next_date.month
            year = next_date.year
        
        predictions = []
        
        for _, row in latest_data.iterrows():
            # Create prediction features
            pred_features = {
                'month': month,
                'year': year,
                'trend': row['trend'] + 1,  # Next time step
                'avg_amount': row['avg_amount'],
                'transaction_count': row['transaction_count'],
                'std_amount': row['std_amount'],
                'lag_1_total': row['total_amount'],  # Current becomes lag_1
                'lag_1_count': row['transaction_count'],
                'lag_2_total': row['lag_1_total'],
                'lag_2_count': row['lag_1_count'],
                'lag_3_total': row['lag_2_total'],
                'lag_3_count': row['lag_2_count'],
                'rolling_2_mean': row['rolling_2_mean'],
                'rolling_2_std': row['rolling_2_std'],
                'rolling_3_mean': row['rolling_3_mean'],
                'rolling_3_std': row['rolling_3_std'],
                'mom_growth': row['mom_growth']
            }
            
            # Create DataFrame with all required features
            pred_df = pd.DataFrame([pred_features])
            
            # Add category dummies
            for feature in self.feature_names:
                if feature.startswith('cat_'):
                    cat_name = feature.replace('cat_', '')
                    pred_df[feature] = 1 if cat_name == row['category'] else 0
            
            # Ensure all features are present
            for feature in self.feature_names:
                if feature not in pred_df.columns:
                    pred_df[feature] = 0
            
            # Reorder columns to match training
            pred_df = pred_df[self.feature_names]
            
            # Predict
            prediction = self.model.predict(pred_df)[0]
            predictions.append(max(0, prediction))  # Ensure non-negative
        
        # Aggregate predictions
        total_prediction = sum(predictions)
        
        # Calculate confidence interval (using historical std)
        historical_amounts = featured_df['total_amount'].values
        historical_avg = historical_amounts.mean() if len(historical_amounts) > 0 else 0
        historical_std = historical_amounts.std() if len(historical_amounts) > 1 else historical_avg * 0.2
        
        # 95% confidence interval (±1.96 * std)
        confidence_lower = max(0, total_prediction - 1.96 * historical_std)
        confidence_upper = total_prediction + 1.96 * historical_std
        
        return {
            'predicted_amount': round(total_prediction, 2),
            'confidence_lower': round(confidence_lower, 2),
            'confidence_upper': round(confidence_upper, 2),
            'historical_average': round(historical_avg, 2)
        }
    
    def predict_next_n_months(
        self, 
        user_transactions: pd.DataFrame, 
        n_months: int = 3,
        category: Optional[str] = None
    ) -> List[Dict]:
        """
        Predict expenses for next N months
        
        Returns:
            List of predictions for each month
        """
        predictions = []
        current_date = datetime.now()
        
        for i in range(1, n_months + 1):
            target_date = current_date + timedelta(days=30 * i)
            month = target_date.month
            year = target_date.year
            
            pred = self.predict(user_transactions, category, month, year)
            pred['month'] = month
            pred['year'] = year
            predictions.append(pred)
        
        return predictions
    
    def save_model(self, path: str):
        """Save XGBoost model and feature names"""
        os.makedirs(path, exist_ok=True)
        
        self.model.save_model(f"{path}/expense_predictor.json")
        joblib.dump(self.feature_names, f"{path}/expense_feature_names.pkl")
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load XGBoost model and feature names"""
        self.model = xgb.XGBRegressor()
        self.model.load_model(f"{path}/expense_predictor.json")
        self.feature_names = joblib.load(f"{path}/expense_feature_names.pkl")
        
        print(f"Model loaded from {path}")