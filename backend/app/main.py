from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.api import router
from app.database import engine, Base
from app.config import MODEL_DIR, DATASET_DIR
import os

app = FastAPI(
    title="AI Retail Demand Forecaster",
    description="AI-powered retail demand forecasting system with TFT, PPO, and Gemini AI",
    version="1.0.0",
)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:5173",
    "http://localhost:3000",
    "https://retail-forecaster-frontend.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "AI Retail Demand Forecaster API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_dir_exists": os.path.exists(MODEL_DIR),
        "dataset_dir_exists": os.path.exists(DATASET_DIR),
    }
