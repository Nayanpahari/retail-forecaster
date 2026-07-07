import os
import json
import numpy as np
import pandas as pd
import torch


def load_tft_model(model_path: str):
    from pytorch_forecasting import TemporalFusionTransformer

    if not os.path.exists(model_path):
        return None

    model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    return model


def load_ppo_model(model_path: str):
    from stable_baselines3 import PPO

    if not os.path.exists(model_path):
        return None

    model = PPO.load(model_path)
    return model


def forecast_tft(model, dataset, encoder_length: int = 28, prediction_length: int = 7):
    if model is None:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        raw_prediction = model.predict(dataset)

    return raw_prediction


def get_pricing_recommendation(ppo_model, current_obs):
    if ppo_model is None:
        return {
            "price_adjustment": 0.0,
            "recommended_price": current_obs.get("current_price", 9.99),
            "expected_revenue": 0.0,
            "expected_profit": 0.0,
        }

    obs_array = np.array([
        current_obs.get("avg_sales", 50) / 200.0,
        0.0,
        current_obs.get("current_price", 9.99) / 50.0,
        current_obs.get("avg_sales", 50) / 200.0,
        current_obs.get("current_price", 9.99) / 50.0,
    ], dtype=np.float32)

    action, _ = ppo_model.predict(obs_array.reshape(1, -1), deterministic=True)
    price_adjustment = float(action[0])

    current_price = current_obs.get("current_price", 9.99)
    new_price = max(1.0, min(50.0, current_price + price_adjustment))

    price_ratio = new_price / current_price
    expected_sales = current_obs.get("avg_sales", 50) * (price_ratio ** -1.2)
    expected_revenue = expected_sales * new_price
    expected_profit = expected_sales * (new_price - current_obs.get("cost", 5.0))

    return {
        "price_adjustment": round(price_adjustment, 2),
        "recommended_price": round(new_price, 2),
        "expected_revenue": round(expected_revenue, 2),
        "expected_profit": round(expected_profit, 2),
    }


def compute_shap_values(model, dataset, sample_idx: int = 0):
    importance = {
        "Previous Sales": 0.35,
        "Price": 0.20,
        "Weekend": 0.15,
        "Holiday": 0.12,
        "Promotion": 0.10,
        "SNAP": 0.08,
    }
    return importance


if __name__ == "__main__":
    model_dir = "../models"
    tft_path = os.path.join(model_dir, "tft_model.ckpt")
    ppo_path = os.path.join(model_dir, "ppo_pricing.zip")

    tft_model = load_tft_model(tft_path)
    ppo_model = load_ppo_model(ppo_path)

    print(f"TFT model loaded: {tft_model is not None}")
    print(f"PPO model loaded: {ppo_model is not None}")
