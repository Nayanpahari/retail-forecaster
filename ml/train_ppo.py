import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from gymnasium import Env
from gymnasium.spaces import Box


class PricingEnv(Env):
    def __init__(self, historical_sales=None, historical_prices=None, historical_promo=None):
        super().__init__()

        self.sales = historical_sales if historical_sales is not None else np.random.uniform(10, 100, 1826)
        self.prices = historical_prices if historical_prices is not None else np.random.uniform(10, 30, 1826)
        self.promo = historical_promo if historical_promo is not None else np.zeros(1826)
        self.cost = 8.0

        self.current_step = 0
        self.max_steps = len(self.sales) - 1

        self.observation_space = Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([200, 50, 50, 1, 100, 1000]),
            dtype=np.float32,
        )

        self.action_space = Box(
            low=np.array([-5.0]),
            high=np.array([5.0]),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        base_price = self.prices[self.current_step]
        base_sales = self.sales[self.current_step]
        is_promo = self.promo[self.current_step]
        price_change = 0
        if self.current_step > 0:
            price_change = (base_price - self.prices[self.current_step - 1]) / self.prices[self.current_step - 1] * 100

        recent_avg_sales = np.mean(self.sales[max(0, self.current_step - 7):self.current_step + 1]) if self.current_step > 0 else base_sales
        recent_avg_price = np.mean(self.prices[max(0, self.current_step - 7):self.current_step + 1]) if self.current_step > 0 else base_price

        return np.array([
            base_sales / 200.0,
            price_change / 50.0,
            base_price / 50.0,
            is_promo,
            recent_avg_sales / 200.0,
            recent_avg_price / 50.0,
        ], dtype=np.float32)

    def step(self, action):
        price_adjustment = float(action[0])
        current_price = self.prices[self.current_step]
        new_price = np.clip(current_price + price_adjustment, 1.0, 50.0)

        price_ratio = new_price / current_price
        elasticity = -1.2
        promo_boost = 1.3 if self.promo[self.current_step] > 0 else 1.0
        expected_sales = self.sales[self.current_step] * (price_ratio ** elasticity) * promo_boost
        expected_sales = max(0, expected_sales)

        revenue = expected_sales * new_price
        profit = expected_sales * (new_price - self.cost)

        price_penalty = 0
        if new_price < self.cost * 1.1:
            price_penalty = -10
        elif new_price > current_price * 1.5:
            price_penalty = -5

        reward = profit / 100.0 + price_penalty

        self.current_step += 1
        done = self.current_step >= self.max_steps
        obs = self._get_obs() if not done else np.zeros(6, dtype=np.float32)

        info = {
            "new_price": new_price,
            "expected_sales": expected_sales,
            "revenue": revenue,
            "profit": profit,
        }

        return obs, reward, done, False, info


def train_ppo(
    data_path: str = "../datasets/featured_sales.parquet",
    model_dir: str = "../models",
    total_timesteps: int = 50000,
):
    os.makedirs(model_dir, exist_ok=True)

    print("=== PPO Dynamic Pricing Training ===")

    sales_data = None
    price_data = None
    promo_data = None

    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        item_sales = df.groupby("date")["sales"].mean().values
        item_prices = df.groupby("date")["price"].mean().values
        item_promo = df.groupby("date")["promo"].mean().values
        if len(item_sales) > 0:
            sales_data = item_sales
            price_data = item_prices
            promo_data = item_promo

    print("Creating pricing environment...")
    env = PricingEnv(historical_sales=sales_data, historical_prices=price_data, historical_promo=promo_data)

    print("Training PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    model.learn(total_timesteps=total_timesteps)

    model_path = os.path.join(model_dir, "ppo_pricing.zip")
    model.save(model_path)
    print(f"PPO model saved to {model_path}")

    return model


def predict_price(model, obs):
    action, _ = model.predict(obs, deterministic=True)
    return float(action[0])


if __name__ == "__main__":
    train_ppo(total_timesteps=50000)
