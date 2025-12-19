"""
Training script for all ML models
Run this to train models on sample or real data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.ml.models.category_classifier import CategoryClassifier
from app.ml.models.expense_predictor import ExpensePredictor
from app.ml.models.anomaly_detector import AnomalyDetector

def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate sample transaction data for training
    """
    np.random.seed(42)
    
    categories = ['Food', 'Transport', 'Entertainment', 'Shopping', 'Bills', 'Health', 'Education']
    merchants = {
        'Food': ['Swiggy', 'Zomato', 'McDonald', 'Dominos', 'Local Restaurant'],
        'Transport': ['Uber', 'Ola', 'Rapido', 'Petrol Pump', 'Metro', 'Bus'],
        'Entertainment': ['BookMyShow', 'Netflix', 'Amazon Prime', 'Gaming'],
        'Shopping': ['Amazon', 'Flipkart', 'Myntra', 'Local Store', 'Meesho', 'Skincare'],
        'Bills': ['Phone', 'College', 'Hostel'],
        'Health': ['Pharmacy', 'Hospital', 'Gym'],
        'Education': ['Udemy', 'Coursera', 'Books']
    }
    
    amount_ranges = {
        'Food': (50, 800),
        'Transport': (30, 500),
        'Entertainment': (100, 1500),
        'Shopping': (200, 5000),
        'Bills': (500, 4800),
        'Health': (100, 2000),
        'Education': (500, 5000)
    }
    
    data = []
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(n_samples):
        category = np.random.choice(categories)
        merchant = np.random.choice(merchants[category])
        
        # Amount with some variation
        min_amt, max_amt = amount_ranges[category]
        amount = np.random.uniform(min_amt, max_amt)
        
        # Random timestamp in the past year
        days_ago = np.random.randint(0, 365)
        timestamp = start_date + timedelta(days=days_ago)
        
        # Generate SMS text
        sms_templates = [
            f"Debited Rs {amount:.2f} from account to {merchant}",
            f"Rs {amount:.2f} paid to {merchant} via UPI",
            f"Transaction successful: {merchant} Rs {amount:.2f}",
            f"Spent Rs {amount:.2f} at {merchant}"
        ]
        raw_sms = np.random.choice(sms_templates)
        
        data.append({
            'user_id': 1,
            'amount': round(amount, 2),
            'merchant': merchant,
            'category': category,
            'transaction_type': 'DEBIT',
            'timestamp': timestamp,
            'raw_sms': raw_sms
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

def train_category_classifier(df: pd.DataFrame, save_path: str):
    """Train and save category classifier"""
    print("\n" + "="*60)
    print("TRAINING CATEGORY CLASSIFIER")
    print("="*60)
    
    classifier = CategoryClassifier()
    metrics = classifier.train(df)
    classifier.save_model(save_path)
    
    # Test prediction
    print("\nTest Prediction:")
    cat, conf, probs = classifier.predict("Swiggy", 350.0, "Paid Rs 350 to Swiggy")
    print(f"  Merchant: Swiggy, Amount: 350")
    print(f"  Predicted: {cat} (confidence: {conf:.2f})")
    
    return metrics

def train_expense_predictor(df: pd.DataFrame, save_path: str):
    """Train and save expense predictor"""
    print("\n" + "="*60)
    print("TRAINING EXPENSE PREDICTOR")
    print("="*60)
    
    predictor = ExpensePredictor()
    metrics = predictor.train(df, n_splits=3)
    predictor.save_model(save_path)
    
    # Test prediction
    print("\nTest Prediction (next month):")
    pred = predictor.predict(df, category='Food')
    print(f"  Category: Food")
    print(f"  Predicted: ₹{pred['predicted_amount']:.2f}")
    print(f"  Range: ₹{pred['confidence_lower']:.2f} - ₹{pred['confidence_upper']:.2f}")
    print(f"  Historical Avg: ₹{pred['historical_average']:.2f}")
    
    return metrics

def train_anomaly_detector(df: pd.DataFrame, save_path: str):
    """Train and save anomaly detector"""
    print("\n" + "="*60)
    print("TRAINING ANOMALY DETECTOR")
    print("="*60)
    
    detector = AnomalyDetector()
    metrics = detector.train(df, contamination=0.05)
    detector.save_model(save_path)
    
    # Test detection
    print("\nTest Detection:")
    # Normal transaction
    is_anom, score, reason = detector.detect(1, 300, 'Food', 'Swiggy')
    print(f"  ₹300 at Swiggy (Food): {'ANOMALY' if is_anom else 'NORMAL'}")
    print(f"  Reason: {reason}")
    
    # Anomalous transaction
    is_anom, score, reason = detector.detect(1, 15000, 'Food', 'Swiggy')
    print(f"  ₹15000 at Swiggy (Food): {'ANOMALY' if is_anom else 'NORMAL'}")
    print(f"  Reason: {reason}")
    
    return metrics

def main():
    # Paths
    model_save_path = "app/ml/data/saved_models"
    os.makedirs(model_save_path, exist_ok=True)
    
    print("="*60)
    print("SMARTSPEND ML MODEL TRAINING")
    print("="*60)
    
    # Generate or load data
    print("\nGenerating sample data...")
    df = generate_sample_data(n_samples=1000)
    
    # Save sample data
    df.to_csv("app/ml/data/sample_transactions.csv", index=False)
    print(f"Sample data saved: {len(df)} transactions")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Categories: {df['category'].unique().tolist()}")
    
    # Train models
    metrics = {}
    
    try:
        metrics['classifier'] = train_category_classifier(df, model_save_path)
    except Exception as e:
        print(f"Error training classifier: {e}")
    
    try:
        metrics['predictor'] = train_expense_predictor(df, model_save_path)
    except Exception as e:
        print(f"Error training predictor: {e}")
    
    try:
        metrics['detector'] = train_anomaly_detector(df, model_save_path)
    except Exception as e:
        print(f"Error training detector: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nModels saved to: {model_save_path}")
    print("\nMetrics Summary:")
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name.upper()}:")
        for key, value in model_metrics.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()