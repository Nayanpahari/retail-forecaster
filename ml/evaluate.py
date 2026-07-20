"""
Model Accuracy Evaluation Script
Computes TFT demand metrics (MAE, RMSE, MAPE, SMAPE), PPO pricing metrics,
and baseline comparisons on the last 7 days of the dataset.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    GroupNormalizer,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "datasets", "featured_sales.parquet")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "tft_model.ckpt")
OUTPUT_PATH = os.path.join(ROOT_DIR, "models", "evaluation_results.json")

MAX_ENCODER_LENGTH = 28
MAX_PREDICTION_LENGTH = 7
LIMIT_DATA = True
LIMIT_ITEMS = 50
LIMIT_STORES = 3


def load_data():
    print("Loading featured_sales.parquet...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Full dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

    if LIMIT_DATA:
        items = df["item_id"].unique()[:LIMIT_ITEMS]
        stores = df["store_id"].unique()[:LIMIT_STORES]
        df = df[df["item_id"].isin(items) & df["store_id"].isin(stores)].copy()
        print(f"After sampling ({LIMIT_ITEMS} items x {LIMIT_STORES} stores): {df.shape[0]:,} rows")

    return df


def prepare_features(df):
    le_store = LabelEncoder()
    le_item = LabelEncoder()

    df["store_id_enc"] = le_store.fit_transform(df["store_id"])
    df["item_id_enc"] = le_item.fit_transform(df["item_id"])
    df["store_id_enc"] = df["store_id_enc"].astype(str)
    df["item_id_enc"] = df["item_id_enc"].astype(str)
    df["group_id"] = df["store_id_enc"] + "_" + df["item_id_enc"]

    categorical_cols = ["weekday", "month", "quarter", "is_weekend", "promo"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    real_cols = [
        "sales", "price", "price_change", "price_rolling_mean_7", "price_relative",
        "year", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
        "rolling_std_7", "rolling_std_14", "rolling_std_28",
        "rolling_min_7", "rolling_max_7",
        "sales_rolling_mean_28", "sales_rolling_mean_7",
    ]
    for col in real_cols:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    df = df.sort_values(["group_id", "time_idx"]).reset_index(drop=True)
    return df, le_store, le_item


def load_tft_model():
    print("Loading TFT model...")
    model = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model


def compute_metrics(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    nonzero_mask = actual != 0
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(np.abs((actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask])) * 100)
    else:
        mape = float("inf")

    denom = np.abs(actual) + np.abs(predicted)
    nonzero_denom = denom != 0
    if nonzero_denom.sum() > 0:
        smape = float(np.mean(np.abs(actual[nonzero_denom] - predicted[nonzero_denom]) / denom[nonzero_denom]) * 100)
    else:
        smape = float("inf")

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2), "SMAPE": round(smape, 2)}


def evaluate_tft(df, model, test_cutoff):
    print("\n=== TFT Demand Forecasting Evaluation ===")

    training = df[df["time_idx"] <= test_cutoff].copy()
    test = df[df["time_idx"] > test_cutoff].copy()

    test_groups = test.groupby("group_id")
    unique_groups = list(test_groups.groups.keys())
    print(f"Test groups: {len(unique_groups)} (last {MAX_PREDICTION_LENGTH} days each)")

    training_dataset = TimeSeriesDataSet(
        training,
        time_idx="time_idx",
        target="sales",
        group_ids=["group_id"],
        min_encoder_length=MAX_ENCODER_LENGTH // 2,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["store_id_enc", "item_id_enc"],
        static_reals=[],
        time_varying_known_categoricals=["weekday", "month", "quarter", "is_weekend", "promo"],
        time_varying_known_reals=[
            "price", "price_change", "price_rolling_mean_7", "price_relative", "year",
        ],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "sales", "lag_7", "lag_14", "lag_28",
            "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
            "rolling_std_7", "rolling_std_14", "rolling_std_28",
            "rolling_min_7", "rolling_max_7",
            "sales_rolling_mean_28", "sales_rolling_mean_7",
        ],
        target_normalizer=GroupNormalizer(
            groups=["group_id"],
            transformation="softplus",
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, df, predict=True, stop_randomization=True
    )
    dataloader = test_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    print("Running TFT predictions on test set...")
    with torch.no_grad():
        raw_predictions = model.predict(dataloader, mode="prediction")
        if isinstance(raw_predictions, (list, tuple)):
            raw_predictions = raw_predictions[0]
        if hasattr(raw_predictions, "numpy"):
            raw_predictions = raw_predictions.numpy()
        raw_predictions = np.array(raw_predictions)

    raw_predictions = np.maximum(raw_predictions, 0)

    all_actuals = []
    all_predicted = []
    all_store_item = []

    sorted_group_ids = sorted(unique_groups)

    for i, group_id in enumerate(sorted_group_ids):
        if i >= len(raw_predictions):
            break

        group_data = test_groups.get_group(group_id).sort_values("time_idx")
        actual_values = group_data["sales"].values

        group_predictions = raw_predictions[i]
        if group_predictions.ndim > 1:
            group_predictions = group_predictions.flatten()

        n = min(len(actual_values), len(group_predictions))
        if n == 0:
            continue

        all_actuals.extend(actual_values[:n])
        all_predicted.extend(group_predictions[:n])
        all_store_item.append({
            "group_id": group_id,
            "actual_total": float(sum(actual_values[:n])),
            "predicted_total": float(sum(group_predictions[:n])),
        })

    if not all_actuals:
        print("ERROR: No predictions matched test data")
        return None

    aggregate_metrics = compute_metrics(all_actuals, all_predicted)

    per_store_item = []
    for info in all_store_item:
        info["MAE"] = round(abs(info["actual_total"] - info["predicted_total"]), 2)
        per_store_item.append(info)

    print(f"\nAggregate TFT Metrics ({len(all_actuals)} data points):")
    print(f"  MAE:  {aggregate_metrics['MAE']:.4f}")
    print(f"  RMSE: {aggregate_metrics['RMSE']:.4f}")
    print(f"  MAPE: {aggregate_metrics['MAPE']:.2f}%")
    print(f"  SMAPE: {aggregate_metrics['SMAPE']:.2f}%")

    return {
        "aggregate": aggregate_metrics,
        "per_store_item": per_store_item,
        "num_data_points": len(all_actuals),
        "num_groups": len(unique_groups),
    }


def evaluate_baselines(df, test_cutoff):
    print("\n=== Baseline Comparison ===")

    training = df[df["time_idx"] <= test_cutoff].copy()
    test = df[df["time_idx"] > test_cutoff].copy()

    stats_actuals = []
    stats_predicted = []
    naive_actuals = []
    naive_predicted = []

    for group_id, group_test in test.groupby("group_id"):
        group_test = group_test.sort_values("time_idx")
        actual_values = group_test["sales"].values

        group_train = training[training["group_id"] == group_id].sort_values("time_idx")
        if len(group_train) < 28:
            continue

        last_28 = group_train.tail(28)["sales"].values
        avg_sales = np.mean(last_28)
        trend = (np.mean(last_28[-7:]) - np.mean(last_28[:7])) / 7

        stats_forecast = []
        for i in range(len(actual_values)):
            base = avg_sales + trend * (i + 1)
            stats_forecast.append(max(0, base))

        naive_forecast = [last_28[-1]] * len(actual_values)

        n = min(len(actual_values), len(stats_forecast), len(naive_forecast))
        stats_actuals.extend(actual_values[:n])
        stats_predicted.extend(stats_forecast[:n])
        naive_actuals.extend(actual_values[:n])
        naive_predicted.extend(naive_forecast[:n])

    stats_metrics = compute_metrics(stats_actuals, stats_predicted) if stats_actuals else None
    naive_metrics = compute_metrics(naive_actuals, naive_predicted) if naive_actuals else None

    print(f"\nStatistical Baseline (linear trend from last 28 days):")
    if stats_metrics:
        print(f"  MAE:  {stats_metrics['MAE']:.4f}")
        print(f"  RMSE: {stats_metrics['RMSE']:.4f}")
        print(f"  MAPE: {stats_metrics['MAPE']:.2f}%")
        print(f"  SMAPE: {stats_metrics['SMAPE']:.2f}%")

    print(f"\nNaive Baseline (repeat last value):")
    if naive_metrics:
        print(f"  MAE:  {naive_metrics['MAE']:.4f}")
        print(f"  RMSE: {naive_metrics['RMSE']:.4f}")
        print(f"  MAPE: {naive_metrics['MAPE']:.2f}%")
        print(f"  SMAPE: {naive_metrics['SMAPE']:.2f}%")

    return {
        "statistical": stats_metrics,
        "naive": naive_metrics,
    }


def evaluate_ppo(df, test_cutoff):
    print("\n=== PPO Pricing Model Evaluation ===")

    training = df[df["time_idx"] <= test_cutoff].copy()
    test = df[df["time_idx"] > test_cutoff].copy()

    try:
        from stable_baselines3.common.policies import ActorCriticPolicy
        from gymnasium import spaces
        import io
        import zipfile

        ppo_path = os.path.join(ROOT_DIR, "models", "ppo_pricing.zip")
        if not os.path.exists(ppo_path):
            print("PPO model not found, skipping pricing evaluation")
            return None

        print("Loading PPO model...")
        with zipfile.ZipFile(ppo_path, "r") as zf:
            policy_data = torch.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=False)
            pytorch_vars = torch.load(io.BytesIO(zf.read("pytorch_variables.pth")), map_location="cpu", weights_only=False)

        obs_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        act_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        policy = ActorCriticPolicy(obs_space, act_space, lr_schedule=lambda _: 3e-4)
        policy.load_state_dict(policy_data, strict=False)
        for k, v in pytorch_vars.items():
            setattr(policy, k, v)
        policy.eval()

        class PPOLoader:
            def __init__(self, pol):
                self.policy = pol

            def predict(self, obs, deterministic=True):
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32)
                    if obs_t.ndim == 1:
                        obs_t = obs_t.unsqueeze(0)
                    latent_pi, _ = self.policy.mlp_extractor(obs_t)
                    mean_actions = self.policy.action_net(latent_pi)
                    if deterministic:
                        action = mean_actions
                    else:
                        log_std = self.policy.log_std
                        action = mean_actions + torch.randn_like(mean_actions) * log_std.exp()
                    action_np = action.clamp(-5.0, 5.0).numpy()
                    return action_np, None

        ppo_model = PPOLoader(policy)
        print("PPO model loaded successfully")

    except Exception as e:
        print(f"Failed to load PPO model: {e}")
        return None

    price_elasticity = -1.2
    results = []

    for group_id, group_test in test.groupby("group_id"):
        group_test = group_test.sort_values("time_idx")
        actual_sales = group_test["sales"].values
        actual_prices = group_test["price"].values

        group_train = training[training["group_id"] == group_id].sort_values("time_idx")
        if len(group_train) < 28:
            continue

        last_28 = group_train.tail(28)
        avg_sales = last_28["sales"].mean()
        current_price = float(actual_prices[0]) if len(actual_prices) > 0 else 15.0

        recent_avg_sales = avg_sales
        recent_avg_price = current_price
        price_change = 0.0

        if len(group_train) > 7:
            last_7 = group_train.tail(7)
            recent_avg_sales = last_7["sales"].mean()
            recent_avg_price = last_7["price"].mean()
            if len(group_train) > 1 and group_train["price"].iloc[-2] > 0:
                price_change = (current_price - group_train["price"].iloc[-2]) / group_train["price"].iloc[-2]

        obs = np.array([
            recent_avg_sales / 200.0,
            price_change / 50.0,
            current_price / 50.0,
            0.0,
            recent_avg_sales / 200.0,
            recent_avg_price / 50.0,
        ], dtype=np.float32)

        action, _ = ppo_model.predict(obs.reshape(1, -1), deterministic=True)
        price_adjustment = float(action[0])
        ppo_price = max(1.0, min(50.0, current_price + price_adjustment))

        baseline_revenue = float(sum(actual_sales * current_price))
        ppo_revenue = float(sum(actual_sales * ppo_price))

        optimal_price = current_price * (price_elasticity / (price_elasticity + 1))
        optimal_revenue = float(sum(actual_sales * optimal_price))

        results.append({
            "group_id": group_id,
            "current_price": round(current_price, 2),
            "ppo_price": round(ppo_price, 2),
            "price_adjustment": round(price_adjustment, 2),
            "actual_total_sales": round(float(sum(actual_sales)), 2),
            "baseline_revenue": round(baseline_revenue, 2),
            "ppo_revenue": round(ppo_revenue, 2),
            "optimal_revenue": round(optimal_revenue, 2),
            "ppo_vs_baseline_pct": round(((ppo_revenue - baseline_revenue) / baseline_revenue * 100) if baseline_revenue > 0 else 0, 2),
        })

    if not results:
        print("No PPO evaluation results")
        return None

    avg_ppo_lift = np.mean([r["ppo_vs_baseline_pct"] for r in results])
    total_baseline = sum(r["baseline_revenue"] for r in results)
    total_ppo = sum(r["ppo_revenue"] for r in results)
    total_optimal = sum(r["optimal_revenue"] for r in results)

    print(f"\nPPO Pricing Results ({len(results)} store-item pairs):")
    print(f"  Avg price adjustment: {np.mean([r['price_adjustment'] for r in results]):.2f}")
    print(f"  Avg PPO vs baseline: {avg_ppo_lift:+.2f}%")
    print(f"  Total baseline revenue: ${total_baseline:,.2f}")
    print(f"  Total PPO revenue: ${total_ppo:,.2f}")
    print(f"  Total optimal revenue: ${total_optimal:,.2f}")

    return {
        "avg_price_adjustment": round(float(np.mean([r["price_adjustment"] for r in results])), 2),
        "avg_ppo_lift_pct": round(float(avg_ppo_lift), 2),
        "total_baseline_revenue": round(total_baseline, 2),
        "total_ppo_revenue": round(total_ppo, 2),
        "total_optimal_revenue": round(total_optimal, 2),
        "per_store_item": results,
    }


def main():
    df = load_data()
    df, le_store, le_item = prepare_features(df)

    max_time_idx = df["time_idx"].max()
    test_cutoff = max_time_idx - MAX_PREDICTION_LENGTH

    tft_model = load_tft_model()

    tft_results = evaluate_tft(df, tft_model, test_cutoff)
    baseline_results = evaluate_baselines(df, test_cutoff)
    ppo_results = evaluate_ppo(df, test_cutoff)

    final_results = {
        "dataset": {
            "total_rows": int(df.shape[0]),
            "num_store_item_pairs": int(df["group_id"].nunique()),
            "test_cutoff_time_idx": int(test_cutoff),
            "max_time_idx": int(max_time_idx),
            "test_days": MAX_PREDICTION_LENGTH,
            "limit_data": LIMIT_DATA,
        },
        "tft_demand_forecasting": tft_results,
        "baseline_comparison": baseline_results,
        "ppo_pricing": ppo_results,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("RESULTS SAVED TO:", OUTPUT_PATH)
    print(f"{'='*60}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if tft_results:
        m = tft_results["aggregate"]
        print(f"\nTFT Model:")
        print(f"  MAE:  {m['MAE']:.4f}  |  RMSE: {m['RMSE']:.4f}")
        print(f"  MAPE: {m['MAPE']:.2f}%  |  SMAPE: {m['SMAPE']:.2f}%")

    if baseline_results:
        print(f"\nBaselines:")
        if baseline_results.get("statistical"):
            b = baseline_results["statistical"]
            print(f"  Statistical: MAE={b['MAE']:.4f} | RMSE={b['RMSE']:.4f} | MAPE={b['MAPE']:.2f}%")
        if baseline_results.get("naive"):
            n = baseline_results["naive"]
            print(f"  Naive:       MAE={n['MAE']:.4f} | RMSE={n['RMSE']:.4f} | MAPE={n['MAPE']:.2f}%")

    if ppo_results:
        print(f"\nPPO Pricing:")
        print(f"  Avg price adjustment: {ppo_results['avg_price_adjustment']:.2f}")
        print(f"  Avg PPO vs baseline revenue: {ppo_results['avg_ppo_lift_pct']:+.2f}%")


if __name__ == "__main__":
    main()
