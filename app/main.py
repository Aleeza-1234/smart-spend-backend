"""
SmartSpend FastAPI Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.v1.router import api_router
from app.core.database import engine, Base
from app.ml.inference.model_loader import model_loader
import app.models

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Personal Finance Intelligence & Travel Planning API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print("Database initialized")
    
    # Load ML models
    print("Loading ML models...")
    model_loader.load_models()
    
    print("✓ Application started successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down...")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SmartSpend API",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ml_status = {
        "classifier": model_loader.get_category_classifier() is not None,
        "predictor": model_loader.get_expense_predictor() is not None,
        "detector": model_loader.get_anomaly_detector() is not None
    }
    
    return {
        "status": "healthy",
        "ml_models": ml_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )