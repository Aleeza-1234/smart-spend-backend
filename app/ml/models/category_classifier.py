import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from typing import Tuple, Dict

class CategoryClassifier:
    def __init__(self, model_path: str = None):
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.label_encoder = LabelEncoder()
        self.model_path = model_path
        
        if model_path and os.path.exists(f"{model_path}/category_classifier.pkl"):
            self.load_model(model_path)
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess transaction text"""
        if not text:
            return ""
        
        text = text.lower()
        # Remove special characters but keep spaces
        text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Prepare features from transaction data
        Features: merchant name, amount bins, text vectorization
        """
        # Combine merchant and raw text
        df['combined_text'] = (
            df['merchant'].fillna('') + ' ' + 
            df['raw_sms'].fillna('')
        ).apply(self.preprocess_text)
        
        # Text features
        if fit:
            text_features = self.vectorizer.fit_transform(df['combined_text'])
        else:
            text_features = self.vectorizer.transform(df['combined_text'])
        
        # Amount features (binned)
        df['amount_bin'] = pd.cut(
            df['amount'], 
            bins=[0, 100, 500, 1000, 5000, float('inf')],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)
        
        amount_features = df[['amount_bin']].values
        
        # Combine features
        from scipy.sparse import hstack
        features = hstack([text_features, amount_features])
        
        return features
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the category classifier
        
        Args:
            df: DataFrame with columns: merchant, amount, category, raw_sms
        
        Returns:
            Dict with training metrics
        """
        print(f"Training with {len(df)} samples...")
        
        # Prepare features
        X = self.prepare_features(df, fit=True)
        
        # Encode labels
        y = self.label_encoder.fit_transform(df['category'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))
        
        return {
            "accuracy": accuracy,
            "n_samples": len(df),
            "n_features": X.shape[1],
            "n_classes": len(self.label_encoder.classes_)
        }
    
    def predict(self, merchant: str, amount: float, raw_sms: str = "") -> Tuple[str, float, Dict[str, float]]:
        """
        Predict category for a transaction
        
        Returns:
            (predicted_category, confidence, all_probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Create DataFrame for single prediction
        df = pd.DataFrame([{
            'merchant': merchant or "",
            'amount': amount,
            'raw_sms': raw_sms or ""
        }])
        
        # Prepare features
        X = self.prepare_features(df, fit=False)
        
        # Predict
        probabilities = self.model.predict_proba(X)[0]
        predicted_class = self.model.predict(X)[0]
        
        # Decode
        predicted_category = self.label_encoder.inverse_transform([predicted_class])[0]
        confidence = probabilities.max()
        
        # All probabilities
        all_probs = {
            cat: float(prob) 
            for cat, prob in zip(self.label_encoder.classes_, probabilities)
        }
        
        return predicted_category, confidence, all_probs
    
    def save_model(self, path: str):
        """Save model, vectorizer, and label encoder"""
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(self.model, f"{path}/category_classifier.pkl")
        joblib.dump(self.vectorizer, f"{path}/category_vectorizer.pkl")
        joblib.dump(self.label_encoder, f"{path}/category_label_encoder.pkl")
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model, vectorizer, and label encoder"""
        self.model = joblib.load(f"{path}/category_classifier.pkl")
        self.vectorizer = joblib.load(f"{path}/category_vectorizer.pkl")
        self.label_encoder = joblib.load(f"{path}/category_label_encoder.pkl")
        
        print(f"Model loaded from {path}")