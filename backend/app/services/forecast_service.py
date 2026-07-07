import os
import json
import io
import zipfile
import numpy as np
import pandas as pd
from typing import Optional


class ForecastService:
    def __init__(self, model_dir: str, dataset_dir: str):
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.tft_model = None
        self.ppo_model = None
        self.encoders = None
        self.dataset_info = None
        self.sales_data = None
        self._load_data()
        self._load_models()

    def _load_data(self):
        possible_paths = [
            self.dataset_dir,
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets"),
            os.path.join(os.getcwd(), "datasets"),
            os.path.join(os.getcwd(), "..", "datasets"),
        ]

        for path in possible_paths:
            processed_path = os.path.join(path, "processed_sales.parquet")
            csv_path = os.path.join(path, "retail_sales.csv")
            print(f"Trying path: {path}")

            if os.path.exists(processed_path):
                self.sales_data = pd.read_parquet(processed_path)
                self.sales_data["date"] = pd.to_datetime(self.sales_data["date"])
                print(f"SUCCESS: Loaded {len(self.sales_data):,} rows from {processed_path}")
                return
            elif os.path.exists(csv_path):
                self.sales_data = pd.read_csv(csv_path, parse_dates=["date"])
                print(f"SUCCESS: Loaded {len(self.sales_data):,} rows from {csv_path}")
                return

        print("WARNING: No dataset found in any path. Using fallback product lists.")
        self.sales_data = None

    def _load_models(self):
        model_paths = [
            self.model_dir,
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"),
            os.path.join(os.getcwd(), "models"),
            os.path.join(os.getcwd(), "..", "models"),
        ]

        tft_loaded = False
        ppo_loaded = False
        encoders_loaded = False
        info_loaded = False

        for path in model_paths:
            tft_path = os.path.join(path, "tft_model.ckpt")
            ppo_path = os.path.join(path, "ppo_pricing.zip")
            encoder_path = os.path.join(path, "encoders.joblib")
            info_path = os.path.join(path, "dataset_info.json")

            if not tft_loaded and os.path.exists(tft_path):
                try:
                    import torch
                    from pytorch_forecasting import TemporalFusionTransformer
                    self.tft_model = TemporalFusionTransformer.load_from_checkpoint(tft_path)
                    self.tft_model.eval()
                    print(f"SUCCESS: Loaded TFT model from {tft_path}")
                    tft_loaded = True
                except Exception as e:
                    print(f"WARNING: Failed to load TFT model: {e}")

            if not ppo_loaded and os.path.exists(ppo_path):
                try:
                    self.ppo_model = self._load_ppo_from_zip(ppo_path)
                    if self.ppo_model is not None:
                        print(f"SUCCESS: Loaded PPO model from {ppo_path}")
                        ppo_loaded = True
                    else:
                        print(f"WARNING: Failed to load PPO model from {ppo_path}")
                except Exception as e:
                    print(f"WARNING: Failed to load PPO model: {e}")

            if not encoders_loaded and os.path.exists(encoder_path):
                try:
                    import joblib
                    self.encoders = joblib.load(encoder_path)
                    print(f"SUCCESS: Loaded encoders from {encoder_path}")
                    encoders_loaded = True
                except Exception as e:
                    print(f"WARNING: Failed to load encoders: {e}")

            if not info_loaded and os.path.exists(info_path):
                with open(info_path) as f:
                    self.dataset_info = json.load(f)
                print(f"SUCCESS: Loaded dataset info from {info_path}")
                info_loaded = True

            if tft_loaded and ppo_loaded and encoders_loaded and info_loaded:
                break

        if not tft_loaded:
            print("WARNING: TFT model not loaded. Using statistical forecast.")
        if not ppo_loaded:
            print("WARNING: PPO model not loaded. Using rule-based pricing.")

    def _load_ppo_from_zip(self, zip_path: str):
        import io
        import torch
        import numpy as np
        from gymnasium import spaces
        from stable_baselines3.common.policies import ActorCriticPolicy

        with zipfile.ZipFile(zip_path, 'r') as zf:
            policy_data = torch.load(io.BytesIO(zf.read('policy.pth')), map_location='cpu', weights_only=False)
            pytorch_vars = torch.load(io.BytesIO(zf.read('pytorch_variables.pth')), map_location='cpu', weights_only=False)

        obs_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        act_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)

        policy = ActorCriticPolicy(obs_space, act_space, lr_schedule=lambda _: 3e-4)
        policy.load_state_dict(policy_data, strict=False)
        for k, v in pytorch_vars.items():
            setattr(policy, k, v)
        policy.eval()

        class _PPOLoader:
            def __init__(self, policy):
                self.policy = policy

            def predict(self, obs, deterministic=True):
                with torch.no_grad():
                    if isinstance(obs, np.ndarray):
                        obs_t = torch.tensor(obs, dtype=torch.float32)
                    else:
                        obs_t = obs
                    if obs_t.ndim == 1:
                        obs_t = obs_t.unsqueeze(0)

                    latent_pi, latent_vf = self.policy.mlp_extractor(obs_t)
                    mean_actions = self.policy.action_net(latent_pi)
                    if deterministic:
                        action = mean_actions
                    else:
                        log_std = self.policy.log_std
                        action = mean_actions + torch.randn_like(mean_actions) * log_std.exp()

                    action_np = action.clamp(-5.0, 5.0).numpy()
                    return action_np, None

        return _PPOLoader(policy)

    def get_products(self):
        if self.sales_data is not None:
            products = (
                self.sales_data[["item_id", "store_id"]]
                .drop_duplicates()
                .sort_values(["item_id", "store_id"])
                .to_dict("records")
            )
            return products

        products = []
        for i in range(1, 51):
            for j in range(1, 51):
                products.append({"item_id": f"item_{i}", "store_id": f"store_{j}"})
        return products

    def get_stores(self):
        if self.sales_data is not None:
            stores = (
                self.sales_data[["store_id"]]
                .drop_duplicates()
                .sort_values("store_id")
                .to_dict("records")
            )
            for s in stores:
                s["state_id"] = "Main"
            return stores

        return [{"store_id": f"store_{i}", "state_id": "Main"} for i in range(1, 51)]

    def get_heatmap_data(self) -> dict:
        if self.sales_data is None:
            return self._generate_mock_heatmap()

        df = self.sales_data.copy()
        df["weekday"] = df["date"].dt.dayofweek

        store_samples = sorted(df["store_id"].unique())[:6]
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        heatmap = {}
        for store in store_samples:
            store_data = df[df["store_id"] == store]
            daily_avg = store_data.groupby("weekday")["sales"].mean()
            heatmap[store] = [round(daily_avg.get(d, 0), 1) for d in range(7)]

        return {
            "stores": store_samples,
            "days": day_names,
            "data": heatmap,
        }

    def _generate_mock_heatmap(self) -> dict:
        stores = [f"store_{i}" for i in [1, 5, 10, 20, 30, 50]]
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        data = {}
        for store in stores:
            data[store] = [round(np.random.uniform(30, 70), 1) for _ in range(7)]
        return {"stores": stores, "days": days, "data": data}

    def _get_item_data(self, item_id: str, store_id: str) -> Optional[pd.DataFrame]:
        if self.sales_data is None:
            return None
        mask = (self.sales_data["item_id"] == item_id) & (
            self.sales_data["store_id"] == store_id
        )
        df = self.sales_data[mask].copy()
        if df.empty:
            return None
        return df.sort_values("date").reset_index(drop=True)

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["week"] = df["date"].dt.isocalendar().week.astype(int)
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["quarter"] = df["date"].dt.quarter
        df["day_of_year"] = df["date"].dt.dayofyear

        df["weekday"] = df["day_of_week"]

        for lag in [7, 14, 28]:
            df[f"lag_{lag}"] = df.groupby(["store_id", "item_id"])["sales"].shift(lag)

        for window in [7, 14, 28]:
            df[f"rolling_mean_{window}"] = (
                df.groupby(["store_id", "item_id"])["sales"]
                .transform(lambda x: x.shift(1).rolling(window).mean())
            )
            df[f"rolling_std_{window}"] = (
                df.groupby(["store_id", "item_id"])["sales"]
                .transform(lambda x: x.shift(1).rolling(window).std())
            )
            df[f"rolling_min_{window}"] = (
                df.groupby(["store_id", "item_id"])["sales"]
                .transform(lambda x: x.shift(1).rolling(window).min())
            )
            df[f"rolling_max_{window}"] = (
                df.groupby(["store_id", "item_id"])["sales"]
                .transform(lambda x: x.shift(1).rolling(window).max())
            )

        df["price_change"] = df.groupby(["store_id", "item_id"])["price"].pct_change().fillna(0)
        df["price_rolling_mean_7"] = (
            df.groupby(["store_id", "item_id"])["price"]
            .transform(lambda x: x.rolling(7, min_periods=1).mean())
        )
        df["price_relative"] = df["price"] / df.groupby("date")["price"].transform("mean")
        df["sales_rolling_mean_28"] = (
            df.groupby(["store_id", "item_id"])["sales"]
            .transform(lambda x: x.rolling(28, min_periods=1).mean())
        )
        df["sales_rolling_mean_7"] = (
            df.groupby(["store_id", "item_id"])["sales"]
            .transform(lambda x: x.rolling(7, min_periods=1).mean())
        )

        unique_dates = sorted(df["date"].unique())
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        df["time_idx"] = df["date"].map(date_to_idx)

        df = df.dropna(subset=["lag_28", "rolling_mean_28"])

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    def _predict_with_tft(self, df: pd.DataFrame, forecast_days: int) -> Optional[dict]:
        if self.tft_model is None or self.encoders is None:
            return None

        try:
            import torch
            from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer

            le_store = self.encoders["store_encoder"]
            le_item = self.encoders["item_encoder"]

            df = df.copy()

            try:
                df["store_id_enc"] = le_store.transform(df["store_id"])
            except ValueError:
                return None
            try:
                df["item_id_enc"] = le_item.transform(df["item_id"])
            except ValueError:
                return None

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

            max_encoder_length = self.dataset_info.get("max_encoder_length", 28)
            max_prediction_length = self.dataset_info.get("max_prediction_length", 7)

            df = df.sort_values(["group_id", "time_idx"]).reset_index(drop=True)

            min_required = max_encoder_length + max_prediction_length
            if len(df) < min_required:
                print(f"TFT: Not enough data ({len(df)} < {min_required})")
                return None

            training_dataset = TimeSeriesDataSet(
                df,
                time_idx="time_idx",
                target="sales",
                group_ids=["group_id"],
                min_encoder_length=max_encoder_length // 2,
                max_encoder_length=max_encoder_length,
                min_prediction_length=1,
                max_prediction_length=max_prediction_length,
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

            prediction_dataset = TimeSeriesDataSet.from_dataset(
                training_dataset, df, predict=True, stop_randomization=True
            )
            dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = self.tft_model.to(device)

            with torch.no_grad():
                raw_prediction = model.predict(dataloader, mode="prediction")
                if isinstance(raw_prediction, (list, tuple)):
                    raw_prediction = raw_prediction[0]

                if hasattr(raw_prediction, 'numpy'):
                    predictions = raw_prediction.numpy()
                else:
                    predictions = np.array(raw_prediction)

                if predictions.ndim > 1:
                    predictions = predictions.flatten()

                predictions = np.maximum(predictions, 0)

                return {
                    "forecast_values": predictions.tolist()[:forecast_days],
                    "model_used": "tft",
                }

        except Exception as e:
            print(f"TFT prediction failed: {e}")
            return None

    def _extend_forecast(self, base_values: list, target_days: int) -> list:
        if len(base_values) >= target_days:
            return base_values[:target_days]

        extended = list(base_values)
        n_base = len(base_values)

        if n_base > 1:
            recent_trend = (base_values[-1] - base_values[0]) / (n_base - 1)
        else:
            recent_trend = 0

        last_val = base_values[-1]
        for i in range(target_days - n_base):
            decay = 0.95 ** (i + 1)
            next_val = last_val + recent_trend * decay
            extended.append(max(0, round(next_val, 1)))

        return extended

    def _predict_with_stats(self, df: pd.DataFrame, forecast_days: int) -> dict:
        recent = df.tail(28)
        avg_sales = recent["sales"].mean()
        std_sales = recent["sales"].std()
        trend = (recent["sales"].tail(7).mean() - recent["sales"].head(7).mean()) / 7

        forecast_values = []
        for i in range(forecast_days):
            base = avg_sales + trend * (i + 1)
            forecast_values.append(max(0, base))

        return {
            "forecast_values": forecast_values,
            "model_used": "statistical",
            "avg_sales": avg_sales,
            "std_sales": std_sales,
        }

    def _compute_feature_importance(self, df: pd.DataFrame, avg_sales: float) -> dict:
        if len(df) < 28:
            return {
                "Previous Sales": 0.30,
                "Price": 0.20,
                "Rolling Mean": 0.18,
                "Promotion": 0.12,
                "Weekend": 0.10,
                "Day of Week": 0.10,
            }

        recent = df.tail(28)
        prev_sales_autocorr = abs(recent["sales"].autocorr(lag=1)) if len(recent) > 1 else 0.3

        if avg_sales > 0:
            price_cv = recent["price"].std() / (recent["price"].mean() + 1e-8)
            price_impact = min(0.35, price_cv * 0.5)
        else:
            price_impact = 0.2

        promo_effect = 0.0
        if "promo" in recent.columns:
            promo_sales = recent[recent["promo"] == 1]["sales"].mean()
            non_promo_sales = recent[recent["promo"] == 0]["sales"].mean()
            if non_promo_sales > 0 and not np.isnan(promo_sales):
                promo_effect = min(0.25, abs(promo_sales - non_promo_sales) / (non_promo_sales + 1))

        weekend_effect = 0.0
        if "weekday" in recent.columns:
            weekend_sales = recent[recent["weekday"].isin([5, 6])]["sales"].mean()
            weekday_sales = recent[~recent["weekday"].isin([5, 6])]["sales"].mean()
            if weekday_sales > 0 and not np.isnan(weekend_sales):
                weekend_effect = min(0.20, abs(weekend_sales - weekday_sales) / (weekday_sales + 1))

        rolling_importance = min(0.25, prev_sales_autocorr * 0.4)

        total = prev_sales_autocorr + price_impact + promo_effect + weekend_effect + rolling_importance + 0.1
        if total > 0:
            importance = {
                "Previous Sales": round(prev_sales_autocorr / total, 2),
                "Price": round(price_impact / total, 2),
                "Rolling Mean": round(rolling_importance / total, 2),
                "Promotion": round(promo_effect / total, 2),
                "Weekend": round(weekend_effect / total, 2),
                "Day of Week": round(0.1 / total, 2),
            }
        else:
            importance = {
                "Previous Sales": 0.30,
                "Price": 0.20,
                "Rolling Mean": 0.18,
                "Promotion": 0.12,
                "Weekend": 0.10,
                "Day of Week": 0.10,
            }

        total_check = sum(importance.values())
        if abs(total_check - 1.0) > 0.01:
            scale = 1.0 / total_check
            importance = {k: round(v * scale, 2) for k, v in importance.items()}

        return importance

    def _get_pricing_recommendation(
        self, avg_sales: float, current_price: float, forecast_days: int,
        recent_df: Optional[pd.DataFrame] = None, promotion: bool = False,
    ) -> dict:
        recent_avg_sales = avg_sales
        recent_avg_price = current_price
        price_change = 0.0
        is_promo = 1.0 if promotion else 0.0

        if recent_df is not None and len(recent_df) >= 7:
            last_7 = recent_df.tail(7)
            recent_avg_sales = last_7["sales"].mean()
            recent_avg_price = last_7["price"].mean()
            if len(recent_df) > 1 and recent_df["price"].iloc[-2] > 0:
                price_change = (current_price - recent_df["price"].iloc[-2]) / recent_df["price"].iloc[-2]

        if self.ppo_model is not None:
            try:
                obs = np.array([
                    recent_avg_sales / 200.0,
                    price_change / 50.0,
                    current_price / 50.0,
                    is_promo,
                    recent_avg_sales / 200.0,
                    recent_avg_price / 50.0,
                ], dtype=np.float32)

                action, _ = self.ppo_model.predict(obs.reshape(1, -1), deterministic=True)
                price_adjustment = float(action[0])
                new_price = max(1.0, min(50.0, current_price + price_adjustment))

                price_elasticity = -1.2
                price_ratio = new_price / current_price if current_price > 0 else 1.0
                expected_sales = avg_sales * (price_ratio ** price_elasticity)
                expected_revenue = expected_sales * forecast_days * new_price
                cost = current_price * 0.4
                expected_profit = expected_sales * forecast_days * (new_price - cost)

                return {
                    "suggested_price": round(new_price, 2),
                    "price_elasticity": round(price_elasticity, 2),
                    "expected_revenue": round(expected_revenue, 2),
                    "expected_profit": round(expected_profit, 2),
                    "cost": round(cost, 2),
                    "pricing_model": "ppo",
                }
            except Exception as e:
                print(f"PPO pricing failed: {e}")

        price_elasticity = -1.2
        cost = current_price * 0.4
        expected_revenue = avg_sales * forecast_days * current_price
        expected_profit = avg_sales * forecast_days * (current_price - cost)

        return {
            "suggested_price": round(current_price, 2),
            "price_elasticity": round(price_elasticity, 2),
            "expected_revenue": round(expected_revenue, 2),
            "expected_profit": round(expected_profit, 2),
            "cost": round(cost, 2),
            "pricing_model": "rule_based",
        }

    def predict(
        self,
        item_id: str,
        store_id: str,
        forecast_days: int,
        current_inventory: int,
        price: Optional[float] = None,
        promotion: bool = False,
        holiday: bool = False,
    ):
        df = self._get_item_data(item_id, store_id)
        if df is None or len(df) < 50:
            return self._generate_mock_prediction(
                item_id, store_id, forecast_days, current_inventory, price
            )

        df = self._create_features(df)

        if len(df) < 30:
            return self._generate_mock_prediction(
                item_id, store_id, forecast_days, current_inventory, price
            )

        base_price = price if price else df["price"].iloc[-1]
        recent = df.tail(28)
        avg_sales = recent["sales"].mean()
        std_sales = recent["sales"].std()

        tft_result = self._predict_with_tft(df, forecast_days)

        if tft_result and tft_result.get("forecast_values"):
            tft_values = tft_result["forecast_values"]
            if len(tft_values) < forecast_days:
                tft_values = self._extend_forecast(tft_values, forecast_days)
            forecast_values = tft_values[:forecast_days]
            model_used = "tft"
        else:
            stats_result = self._predict_with_stats(df, forecast_days)
            forecast_values = stats_result["forecast_values"]
            model_used = "statistical"

        if promotion:
            forecast_values = [v * 1.15 for v in forecast_values]
        if holiday:
            forecast_values = [v * 1.25 for v in forecast_values]

        predicted_demand = sum(forecast_values)
        confidence = min(0.95, max(0.6, 1 - (std_sales / (avg_sales + 1)) * 0.5))

        pricing = self._get_pricing_recommendation(
            avg_sales, base_price, forecast_days, recent_df=df, promotion=promotion
        )

        suggested_price = pricing["suggested_price"]
        revenue = predicted_demand * suggested_price

        safety_stock = int(1.65 * std_sales * np.sqrt(7))
        daily_sales = predicted_demand / forecast_days
        days_until_stockout = (
            int(current_inventory / daily_sales) if daily_sales > 0 else 999
        )
        reorder_qty = max(0, int(predicted_demand + safety_stock - current_inventory))

        shap_values = self._compute_feature_importance(df, avg_sales)

        last_date = df["date"].iloc[-1]
        forecast_dates = [
            str(last_date + pd.Timedelta(days=i + 1))
            for i in range(forecast_days)
        ]

        return {
            "item_id": item_id,
            "store_id": store_id,
            "forecast_days": forecast_days,
            "predicted_demand": round(predicted_demand, 1),
            "confidence": round(confidence, 3),
            "revenue": round(revenue, 2),
            "suggested_price": suggested_price,
            "current_inventory": current_inventory,
            "safety_stock": safety_stock,
            "days_until_stockout": days_until_stockout,
            "reorder_quantity": reorder_qty,
            "price_elasticity": pricing["price_elasticity"],
            "expected_profit": pricing["expected_profit"],
            "cost": pricing["cost"],
            "pricing_model": pricing["pricing_model"],
            "model_used": model_used,
            "forecast_data": {
                "dates": forecast_dates,
                "values": [round(v, 1) for v in forecast_values],
            },
            "shap_values": shap_values,
        }

    def _generate_mock_prediction(
        self, item_id, store_id, forecast_days, current_inventory, price
    ):
        base_demand = 50.0
        forecast_values = [max(0, base_demand) for _ in range(forecast_days)]
        predicted_demand = sum(forecast_values)
        current_price = price if price else 15.0
        confidence = 0.75
        safety_stock = int(base_demand * 7)
        daily_sales = predicted_demand / forecast_days
        days_until_stockout = (
            int(current_inventory / daily_sales) if daily_sales > 0 else 999
        )
        reorder_qty = max(0, int(predicted_demand + safety_stock - current_inventory))

        cost = current_price * 0.4
        price_elasticity = -1.2
        expected_profit = predicted_demand * (current_price - cost)

        now = pd.Timestamp.now()

        return {
            "item_id": item_id,
            "store_id": store_id,
            "forecast_days": forecast_days,
            "predicted_demand": round(predicted_demand, 1),
            "confidence": round(confidence, 3),
            "revenue": round(predicted_demand * current_price, 2),
            "suggested_price": round(current_price, 2),
            "current_inventory": current_inventory,
            "safety_stock": safety_stock,
            "days_until_stockout": days_until_stockout,
            "reorder_quantity": reorder_qty,
            "price_elasticity": round(price_elasticity, 2),
            "expected_profit": round(expected_profit, 2),
            "cost": round(cost, 2),
            "pricing_model": "mock",
            "model_used": "mock",
            "forecast_data": {
                "dates": [str(now + pd.Timedelta(days=i + 1)) for i in range(forecast_days)],
                "values": [round(v, 1) for v in forecast_values],
            },
            "shap_values": {
                "Previous Sales": 0.30,
                "Price": 0.20,
                "Rolling Mean": 0.18,
                "Promotion": 0.12,
                "Weekend": 0.10,
                "Day of Week": 0.10,
            },
        }
