# SmartSpend Backend

Personal Finance Intelligence & Travel Planning API with ML

## Features

✨ **Transaction Management**
- SMS parsing for Indian banks (SBI, HDFC, ICICI, etc.)
- Automatic category classification (ML)
- Anomaly detection for unusual spending

🎯 **Budget Planning**
- Daily budget calculator
- Spending alerts and insights
- Category-wise budget tracking

✈️ **Travel Planning**
- Cost estimation for trips
- Timeline prediction (when can you afford it)
- Personalized savings suggestions

🤖 **Machine Learning**
- Category classifier (Random Forest)
- Expense predictor (XGBoost)
- Anomaly detector (Isolation Forest)

## Tech Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL (async)
- **Cache**: Redis
- **ML**: scikit-learn, XGBoost
- **NLP**: spaCy
- **MLOps**: MLflow

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone <repo-url>
cd smartspend-backend

# Start all services
docker-compose up -d

# Train ML models (first time only)
docker-compose exec backend python -m app.ml.training.train_all_models

# API available at http://localhost:8000