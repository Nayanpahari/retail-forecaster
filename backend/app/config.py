import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:password@localhost:3306/retail_forecaster")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
DATASET_DIR = os.getenv("DATASET_DIR", "./datasets")

TFT_MODEL_PATH = os.path.join(MODEL_DIR, "tft_model.ckpt")
PPO_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_pricing.zip")
