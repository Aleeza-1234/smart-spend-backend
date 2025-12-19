import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple
import joblib
import os

class AnomalyDetector:
    """
    Detects unusual transactions using Isolation Forest
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = StandardScaler()
        self.category_stats = {}
        self.model_path = model_path
        
        if model_path and os.path.exists(f"{model_path}/anomaly_detector.pkl"):
            self.load_model(model_path)
    
    def calculate_category_stats(self, df: pd.DataFrame):
        """Calculate statistical parameters for each category"""
        self.category_stats = {}
        
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]['amount']
            
            self.category_stats[category] = {
                'mean': cat_data.mean(),
                'std': cat_data.std(),
                'median': cat_data.median(),
                'q1': cat_data.quantile(0.25),
                'q3': cat_data.quantile(0.75),
                'iqr': cat_data.quantile(0.75) - cat_data.quantile(0.25),
                'min': cat_data.min(),
                'max': cat_data.max()
            }
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for anomaly detection
        
        Features:
        - Amount (normalized)
        - Z-score within category
        - Distance from median
        - Time features
        - Transaction frequency
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        features = pd.DataFrame()
        
        # Amount features
        features['amount'] = df['amount']
        
        # Z-score within category
        features['z_score'] = df.apply(
            lambda row: (row['amount'] - self.category_stats.get(row['category'], {}).get('mean', 0)) / 
                       max(self.category_stats.get(row['category'], {}).get('std', 1), 0.01),
            axis=1
        )
        
        # Distance from median (normalized by IQR)
        features['median_distance'] = df.apply(
            lambda row: abs(row['amount'] - self.category_stats.get(row['category'], {}).get('median', 0)) / 
                       max(self.category_stats.get(row['category'], {}).get('iqr', 1), 0.01),
            axis=1
        )
        
        # Time features
        features['hour'] = df['timestamp'].dt.hour
        features['day_of_week'] = df['timestamp'].dt.dayofweek
        features['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
        features['is_night'] = ((df['timestamp'].dt.hour >= 22) | (df['timestamp'].dt.hour <= 6)).astype(int)
        
        # Transaction frequency (transactions per day for user)
        df['date'] = df['timestamp'].dt.date
        daily_counts = df.groupby(['user_id', 'date']).size().reset_index(name='daily_count')
        df = df.merge(daily_counts, on=['user_id', 'date'], how='left')
        features['daily_transaction_count'] = df['daily_count']
        
        # Category encoding (one-hot)
        category_dummies = pd.get_dummies(df['category'], prefix='cat')
        features = pd.concat([features, category_dummies], axis=1)
        
        return features
    
    def train(self, transactions: pd.DataFrame, contamination: float = 0.05) -> Dict[str, float]:
        """
        Train Isolation Forest for anomaly detection
        
        Args:
            transactions: Historical transactions
            contamination: Expected proportion of anomalies (default 5%)
        
        Returns:
            Training statistics
        """
        print(f"Training anomaly detector with {len(transactions)} transactions...")
        
        # Calculate category statistics
        self.calculate_category_stats(transactions)
        
        # Create features
        X = self.create_features(transactions)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        
        predictions = self.model.fit_predict(X_scaled)
        
        # Statistics
        n_anomalies = (predictions == -1).sum()
        anomaly_rate = n_anomalies / len(predictions)
        
        print(f"\nAnomalies detected: {n_anomalies} ({anomaly_rate*100:.2f}%)")
        
        return {
            'n_samples': len(transactions),
            'n_anomalies': int(n_anomalies),
            'anomaly_rate': float(anomaly_rate),
            'contamination': contamination
        }
    
    def detect(
        self, 
        user_id: int,
        amount: float, 
        category: str, 
        merchant: str = None,
        timestamp: pd.Timestamp = None
    ) -> Tuple[bool, float, str]:
        """
        Detect if a transaction is anomalous
        
        Returns:
            (is_anomaly, anomaly_score, reason)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        # Create DataFrame for single transaction
        df = pd.DataFrame([{
            'user_id': user_id,
            'amount': amount,
            'category': category,
            'merchant': merchant or "",
            'timestamp': timestamp
        }])
        
        # Get category stats
        cat_stats = self.category_stats.get(category)
        
        if cat_stats is None:
            # Unknown category - flag as potential anomaly
            return True, -0.8, f"Unknown category: {category}"
        
        # Rule-based checks first (faster)
        reasons = []
        
        # Check if amount is way outside normal range
        if amount > cat_stats['q3'] + 3 * cat_stats['iqr']:
            reasons.append(f"Amount ${amount:.2f} significantly higher than typical ${cat_stats['q3']:.2f}")
        
        if amount < cat_stats['q1'] - 3 * cat_stats['iqr'] and amount > 0:
            reasons.append(f"Amount ${amount:.2f} significantly lower than typical ${cat_stats['q1']:.2f}")
        
        # Z-score check
        z_score = (amount - cat_stats['mean']) / max(cat_stats['std'], 0.01)
        if abs(z_score) > 3:
            reasons.append(f"Z-score {z_score:.2f} indicates unusual amount for {category}")
        
        # ML-based detection
        X = self.create_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly score
        anomaly_score = self.model.score_samples(X_scaled)[0]
        prediction = self.model.predict(X_scaled)[0]
        
        is_anomaly = prediction == -1 or len(reasons) > 0
        
        # Combine reasons
        if reasons:
            reason = "; ".join(reasons)
        elif is_anomaly:
            reason = f"ML model flagged as anomalous (score: {anomaly_score:.3f})"
        else:
            reason = "Transaction appears normal"
        
        return is_anomaly, float(anomaly_score), reason
    
    def save_model(self, path: str):
        """Save model, scaler, and category stats"""
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(self.model, f"{path}/anomaly_detector.pkl")
        joblib.dump(self.scaler, f"{path}/anomaly_scaler.pkl")
        joblib.dump(self.category_stats, f"{path}/category_stats.pkl")
        
        print(f"Anomaly detector saved to {path}")
    
    def load_model(self, path: str):
        """Load model, scaler, and category stats"""
        self.model = joblib.load(f"{path}/anomaly_detector.pkl")
        self.scaler = joblib.load(f"{path}/anomaly_scaler.pkl")
        self.category_stats = joblib.load(f"{path}/category_stats.pkl")
        
        print(f"Anomaly detector loaded from {path}")