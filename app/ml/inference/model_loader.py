"""
ML Model Loader - Singleton pattern for loading models once
"""

import os
from typing import Optional
from app.ml.models.category_classifier import CategoryClassifier
from app.ml.models.expense_predictor import ExpensePredictor
from app.ml.models.anomaly_detector import AnomalyDetector
from app.config import settings

class ModelLoader:
    """
    Singleton class to load and cache ML models
    """
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ModelLoader._models_loaded:
            self.category_classifier: Optional[CategoryClassifier] = None
            self.expense_predictor: Optional[ExpensePredictor] = None
            self.anomaly_detector: Optional[AnomalyDetector] = None
            self.load_models()
            ModelLoader._models_loaded = True
    
    def load_models(self):
        """Load all trained models"""
        model_path = settings.MODEL_PATH
        
        print("Loading ML models...")
        
        try:
            # Load category classifier
            if os.path.exists(f"{model_path}/category_classifier.pkl"):
                self.category_classifier = CategoryClassifier(model_path)
                print("✓ Category Classifier loaded")
            else:
                print("⚠ Category Classifier not found. Run training first.")
            
            # Load expense predictor
            if os.path.exists(f"{model_path}/expense_predictor.json"):
                self.expense_predictor = ExpensePredictor(model_path)
                print("✓ Expense Predictor loaded")
            else:
                print("⚠ Expense Predictor not found. Run training first.")
            
            # Load anomaly detector
            if os.path.exists(f"{model_path}/anomaly_detector.pkl"):
                self.anomaly_detector = AnomalyDetector(model_path)
                print("✓ Anomaly Detector loaded")
            else:
                print("⚠ Anomaly Detector not found. Run training first.")
            
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def get_category_classifier(self) -> Optional[CategoryClassifier]:
        return self.category_classifier
    
    def get_expense_predictor(self) -> Optional[ExpensePredictor]:
        return self.expense_predictor
    
    def get_anomaly_detector(self) -> Optional[AnomalyDetector]:
        return self.anomaly_detector
    
    def reload_models(self):
        """Reload models (useful after retraining)"""
        ModelLoader._models_loaded = False
        self.load_models()

# Global instance
model_loader = ModelLoader()