# AI-Dynamic Retail Demand Forecaster — Complete Project Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Data Pipeline](#4-data-pipeline)
5. [ML Pipeline — Feature Engineering](#5-ml-pipeline--feature-engineering)
6. [ML Pipeline — TFT Model Training](#6-ml-pipeline--tft-model-training)
7. [ML Pipeline — PPO Pricing Agent](#7-ml-pipeline--ppo-pricing-agent)
8. [Backend — FastAPI Application](#8-backend--fastapi-application)
9. [Backend — Forecast Service (Core Engine)](#9-backend--forecast-service-core-engine)
10. [Backend — API Endpoints](#10-backend--api-endpoints)
11. [Backend — Database Schema](#11-backend--database-schema)
12. [Frontend — React Application](#12-frontend--react-application)
13. [Frontend — Page-by-Page Breakdown](#13-frontend--page-by-page-breakdown)
14. [Bugs Found and Fixed](#14-bugs-found-and-fixed)
15. [Deployment](#15-deployment)
16. [API Reference](#16-api-reference)
17. [File Inventory](#17-file-inventory)

---

## 1. Project Overview

**AI-Dynamic Retail Demand Forecaster** (RetailVerse) is a full-stack web application that predicts retail product demand using deep learning, optimizes pricing using reinforcement learning, and generates AI-powered business insights.

### What It Does
- **Predicts demand** for 50 products across 50 stores for up to 30 days ahead
- **Optimizes pricing** using a PPO reinforcement learning agent
- **Generates business insights** using Google Gemini AI
- **Monitors inventory** with stockout risk alerts and reorder recommendations
- **Visualizes analytics** with charts, heatmaps, and trend analysis
- **Generates reports** in PDF and CSV formats

### Key Numbers
- **Dataset:** 4,565,000 rows (50 stores × 50 items × 1,826 days)
- **ML Models:** Temporal Fusion Transformer (2.08 MB) + PPO (0.13 MB)
- **API Endpoints:** 10 REST endpoints
- **Frontend Pages:** 5 (Dashboard, Forecast, Analytics, Inventory, Reports)
- **Total Source Code:** ~3,400 lines across 30+ files

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER'S BROWSER                               │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              React SPA (Vite + Tailwind CSS)                  │  │
│  │                                                               │  │
│  │  Dashboard │ Forecast │ Analytics │ Inventory │ Reports      │  │
│  │                                                               │  │
│  │  api.js → Axios HTTP client (VITE_API_URL or /api proxy)     │  │
│  └───────────────────────────┬───────────────────────────────────┘  │
└──────────────────────────────┼──────────────────────────────────────┘
                               │ HTTP/REST (JSON)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Python 3.11)                    │
│                                                                     │
│  routes/api.py  ──→  services/forecast_service.py  (CORE ENGINE)   │
│                           │                                         │
│  ┌────────────────────────┼────────────────────────────────────┐   │
│  │                        ▼                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐ │   │
│  │  │ TFT Model│  │ PPO Agent│  │  SQLite   │  │  Gemini   │ │   │
│  │  │ (PyTorch)│  │ (SB3)    │  │  Database │  │  AI API   │ │   │
│  │  └──────────┘  └──────────┘  └───────────┘  └───────────┘ │   │
│  │                                                             │   │
│  │  datasets/processed_sales.parquet (12 MB)                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Deployment Modes

| Component | Render (Cloud) | Docker (Local) |
|-----------|---------------|----------------|
| Frontend | Render Static Site (CDN) | Nginx container (port 80) |
| Backend | Render Web Service | Python container (port 8000) |
| Database | SQLite (file-based) | MySQL 8.0 container (port 3306) |
| API Proxy | VITE_API_URL env var | Nginx reverse proxy `/api/` |

---

## 3. Technology Stack

### Frontend
| Technology | Version | Purpose |
|-----------|---------|---------|
| React | 19.2.7 | UI framework |
| React Router DOM | 7.18.1 | Client-side routing (5 pages) |
| Vite | 8.1.1 | Build tool and dev server |
| Tailwind CSS | 4.3.2 | Utility-first CSS framework |
| Chart.js | 4.5.1 | Charts (Line, Bar, Doughnut) |
| react-chartjs-2 | 5.3.1 | React wrapper for Chart.js |
| Axios | 1.18.1 | HTTP client |

### Backend
| Technology | Version | Purpose |
|-----------|---------|---------|
| FastAPI | 0.115.0 | REST API framework |
| SQLAlchemy | 2.0.35 | ORM and database toolkit |
| Pydantic | 2.9.0 | Request/response validation |
| Uvicorn | 0.30.0 | ASGI server |
| ReportLab | 4.2.2 | PDF report generation |
| Google Generative AI | 0.8.0 | Gemini AI insights |

### Machine Learning
| Technology | Version | Purpose |
|-----------|---------|---------|
| PyTorch | 2.12.1+cpu | Deep learning framework |
| PyTorch Forecasting | 1.8.0 | TFT implementation |
| PyTorch Lightning | 2.0+ | Training framework |
| Stable Baselines3 | 2.9.0 | PPO reinforcement learning |
| Pandas | 2.2.2 | Data manipulation |
| NumPy | 1.26.4 | Numerical computing |
| Scikit-learn | 1.5.1 | Label encoding |
| Joblib | 1.4.2 | Model serialization |

---

## 4. Data Pipeline

### Dataset Source
- **Name:** M5 Walmart Store Item Demand Forecasting (Kaggle)
- **Size:** 4,565,000 rows
- **Dimensions:** 50 stores × 50 items × 1,826 days (~5 years)
- **Date Range:** 2019-01-01 to 2023-12-31
- **Columns:** date, store_id, item_id, sales, price, promo, weekday, month

### Processing Pipeline

```
Step 1: data_processing.py
  Input:  Kaggle API → retail_sales.csv (186 MB)
  Process: Sort by store_id, item_id, date
  Output: processed_sales.parquet (12 MB, 93% compression)

Step 2: feature_engineering.py
  Input:  processed_sales.parquet
  Process: Add 30+ engineered features (lags, rolling windows, price features)
  Output: featured_sales.parquet (203 MB)

Step 3: train_tft.py
  Input:  featured_sales.parquet
  Process: Train TFT model for 10 epochs
  Output: tft_model.ckpt + encoders.joblib + dataset_info.json

Step 4: train_ppo.py
  Input:  featured_sales.parquet
  Process: Train PPO pricing agent for 50,000 timesteps
  Output: ppo_pricing.zip
```

### Why Parquet Over CSV?
- **Columnar storage** — reads only needed columns (10x faster for ML)
- **Built-in compression** — 186 MB CSV becomes 12 MB Parquet
- **Type preservation** — no need to re-infer data types on load
- **Pandas integration** — `pd.read_parquet()` is near-instant

---

## 5. ML Pipeline — Feature Engineering

**File:** `ml/feature_engineering.py` (150 lines)

Feature engineering transforms raw sales data into 30+ features that the TFT model can learn from. Each feature captures a different aspect of the demand signal.

### Feature Categories

#### A. Time Features (Calendar)
```python
year, month, day, week, day_of_week, is_weekend, quarter, day_of_year
```
**Why:** Retail demand has strong seasonality. Weekends sell more than weekdays. Q4 (holidays) sells more than Q1. These features let the model learn temporal patterns.

#### B. Lag Features (Autoregressive)
```python
lag_7   # Sales from 7 days ago (weekly cycle)
lag_14  # Sales from 14 days ago (biweekly cycle)
lag_28  # Sales from 28 days ago (monthly cycle)
```
**Why:** Yesterday's sales are the strongest predictor of today's sales. Lag features give the model "memory" of recent demand. The 7/14/28 windows capture weekly, biweekly, and monthly cycles.

#### C. Rolling Window Features (Trend & Volatility)
```python
rolling_mean_7, rolling_mean_14, rolling_mean_28   # Trend
rolling_std_7, rolling_std_14, rolling_std_28       # Volatility
rolling_min_7, rolling_max_7                         # Range extremes
```
**Why:** A 7-day rolling mean smooths out daily noise to reveal the underlying trend. Rolling standard deviation captures demand uncertainty (critical for safety stock calculations). Min/max capture short-term extremes.

**Critical detail:** All rolling features use `x.shift(1).rolling(window)` — the `shift(1)` prevents data leakage by ensuring we never look at the current day's value when computing the rolling statistic.

#### D. Price Features
```python
price_change           # Day-over-day percentage change
price_rolling_mean_7   # 7-day average price
price_relative         # Item price vs cross-item average
```
**Why:** Price changes affect demand (elasticity). A relative price feature tells the model whether this item is expensive or cheap compared to others on the same day.

#### E. Target Encoding Features
```python
sales_rolling_mean_28  # 28-day smoothed demand
sales_rolling_mean_7   # 7-day smoothed demand
```
**Why:** These are smoothed versions of the target variable itself. They help the model learn the baseline demand level for each product-store combination.

### Data Flow
```
processed_sales.parquet (4,565,000 rows × 8 columns)
  → add_time_features()        (+8 columns)
  → add_lag_features()         (+3 columns)
  → add_rolling_features()     (+10 columns)
  → add_price_features()       (+3 columns)
  → add_target_encoding()      (+2 columns)
  → create_time_idx()          (+1 column)
  → drop NaN rows              (-225,000 rows, first 28 days)
  → featured_sales.parquet     (4,495,000 rows × 36 columns)
```

---

## 6. ML Pipeline — TFT Model Training

**File:** `ml/train_tft.py` (337 lines)

### What is TFT?

The **Temporal Fusion Transformer** is a state-of-the-art attention-based architecture specifically designed for multi-horizon time-series forecasting. It was published by Google Research in 2021 and has become the gold standard for retail demand forecasting.

### Why TFT Over Alternatives?

| Model | Pros | Cons | Why Not |
|-------|------|------|---------|
| LSTM | Simple, fast | No attention, no known-future inputs | Can't handle promotions/prices |
| Prophet | Good for trends | No item-level granularity | Too slow for 2,500 items |
| ARIMA | Statistical rigor | No multivariate support | Can't use price/promo features |
| **TFT** | **Attention + known future + interpretability** | **Complex** | **Chosen** |

### Model Architecture

```
Input Features
  │
  ├── Static Categoricals: store_id, item_id (learned embeddings)
  │
  ├── Time-Varying Known: weekday, month, quarter, is_weekend, promo, price, year
  │   (These are known at prediction time — future promotions, prices)
  │
  └── Time-Varying Unknown: sales, lags, rolling stats
      (These are only known up to the present — historical observations)
          │
          ▼
  ┌─────────────────────────────────┐
  │    LSTM Encoder (hidden=32)     │
  │    Processes historical sequence │
  └──────────────┬──────────────────┘
                 │
  ┌──────────────▼──────────────────┐
  │  Variable Selection Networks    │
  │  (learns which features matter) │
  └──────────────┬──────────────────┘
                 │
  ┌──────────────▼──────────────────┐
  │  Multi-Head Attention (4 heads) │
  │  (captures long-range deps)     │
  └──────────────┬──────────────────┘
                 │
  ┌──────────────▼──────────────────┐
  │  Quantile Regression Output     │
  │  (10th, 50th, 90th percentiles) │
  └─────────────────────────────────┘
                 │
                 ▼
  Prediction: [demand_day1, demand_day2, ..., demand_day7]
```

### Training Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| hidden_size | 32 | Small enough for fast training, large enough for patterns |
| attention_head_size | 4 | 4 attention heads capture different pattern types |
| dropout | 0.1 | Prevents overfitting (10% of neurons randomly disabled) |
| learning_rate | 0.03 | Standard for TFT with QuantileLoss |
| max_encoder_length | 28 | 28 days = exactly 4 weeks (captures weekly seasonality) |
| max_prediction_length | 7 | 7 days = 1 week forecast (standard retail planning) |
| batch_size | 64 | Balance between speed and gradient quality |
| max_epochs | 10 | Enough to converge without overfitting |
| early_stopping | patience=5 | Stop if val_loss doesn't improve for 5 epochs |
| gradient_clip_val | 0.1 | Prevents exploding gradients |
| loss | QuantileLoss | Provides prediction intervals (uncertainty estimates) |

### Feature Roles in TFT

**Static Categoricals** (constant for each time series):
- `store_id_enc`, `item_id_enc` — The model learns embeddings for each store and item, capturing "this store sells more" or "this item is seasonal"

**Time-Varying Known Categoricals** (known at prediction time):
- `weekday`, `month`, `quarter`, `is_weekend`, `promo` — Calendar and promotion info. These are "known" because we know next week's calendar and planned promotions.

**Time-Varying Known Reals** (known at prediction time):
- `price`, `price_change`, `price_rolling_mean_7`, `price_relative`, `year` — Future price information. We know what we plan to charge.

**Time-Varying Unknown Reals** (only known historically):
- `sales`, `lag_7/14/28`, all rolling stats — These are observations that only exist up to today. The model must predict these from the encoder's memory.

### Training Process

```python
# 1. Load featured data
df = pd.read_parquet("featured_sales.parquet")

# 2. Sample for fast training (50 items × 3 stores = ~268K rows)
df = df[df["item_id"].isin(sample_items) & df["store_id"].isin(sample_stores)]

# 3. Encode categorical IDs
le_store.fit_transform(df["store_id"])  # "store_1" → "0"
le_item.fit_transform(df["item_id"])    # "item_1" → "0"

# 4. Split by time (not random) — no future data leakage
train_cutoff = df["time_idx"].max() - 7
training = df[df["time_idx"] <= train_cutoff]

# 5. Create TimeSeriesDataSet (TFT's data format)
dataset = TimeSeriesDataSet(training, time_idx="time_idx", target="sales", ...)

# 6. Train with Lightning
trainer = pl.Trainer(max_epochs=10, callbacks=[EarlyStopping, ModelCheckpoint])
trainer.fit(tft, train_dataloaders, val_dataloaders)

# 7. Save artifacts
trainer.save_checkpoint("tft_model.ckpt")
joblib.dump({"store_encoder": le_store, "item_encoder": le_item}, "encoders.joblib")
```

### Training Results

| Metric | Value |
|--------|-------|
| Best validation loss | 1.384 |
| Training samples | ~268,000 |
| Validation samples | ~268,000 |
| Training time | ~5 minutes (CPU) |
| Model size | 2.08 MB |

---

## 7. ML Pipeline — PPO Pricing Agent

**File:** `ml/train_ppo.py` (149 lines)

### What is PPO?

**Proximal Policy Optimization** is a reinforcement learning algorithm that learns an optimal policy through trial and error. It's the standard algorithm for continuous action spaces (like adjusting prices by a dollar amount).

### Why PPO for Pricing?

| Approach | Limitation |
|----------|-----------|
| Rule-based (fixed markup) | Doesn't adapt to demand changes |
| Price elasticity model | Requires manual calibration |
| Grid search | Doesn't generalize to new situations |
| **PPO (RL)** | **Learns optimal pricing from data** |

### Environment Design (`PricingEnv`)

The PPO agent interacts with a simulated retail environment:

**Observation Space (6 dimensions):**
```python
[
    base_sales / 200.0,        # Current day sales (normalized)
    price_change / 50.0,       # % price change from yesterday
    base_price / 50.0,         # Current price (normalized)
    is_promo,                  # 0 or 1
    recent_avg_sales / 200.0,  # 7-day average sales
    recent_avg_price / 50.0,   # 7-day average price
]
```

**Action Space (1 dimension):**
```python
action = price_adjustment  # Range: [-5.0, +5.0] dollars
new_price = clip(current_price + adjustment, 1.0, 50.0)
```

**Reward Function:**
```python
profit = expected_sales * (new_price - cost)
reward = profit / 100.0 + price_penalty

# Penalties:
if new_price < cost * 1.1:   reward -= 10   # Below-cost pricing
if new_price > price * 1.5:  reward -= 5    # Price gouging
```

**Demand Model (Constant Elasticity):**
```python
elasticity = -1.2  # Industry standard for consumer goods
expected_sales = base_sales * (new_price / old_price) ** elasticity * promo_boost

# Example: 10% price increase → 12% demand decrease
# Example: 30% promo boost when promotion active
```

### Why Elasticity = -1.2?

Price elasticity of demand measures how much quantity demanded changes when price changes. For retail consumer goods:
- Elasticity < -1: Elastic (demand drops more than price increases) — most retail goods
- Elasticity = -1.2 means: A 1% price increase causes a 1.2% demand decrease
- This is within the typical range for consumer packaged goods (-0.5 to -2.0)

### PPO Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| learning_rate | 3e-4 | Standard for PPO |
| n_steps | 2048 | Rollout length per update |
| batch_size | 64 | Minibatch size for gradient updates |
| n_epochs | 10 | Passes over each rollout |
| gamma | 0.99 | Discount factor (values future rewards) |
| gae_lambda | 0.95 | Generalized Advantage Estimation smoothing |
| clip_range | 0.2 | PPO clipping (prevents large policy updates) |
| total_timesteps | 50,000 | Total training steps |

### Training Output

```python
model.save("ppo_pricing.zip")  # 0.13 MB compressed
```

The saved zip contains:
- `policy.pth` — Neural network weights (41.5 KB)
- `pytorch_variables.pth` — log_std and other variables (1.23 KB)
- `policy.optimizer.pth` — Optimizer state (83.67 KB)
- `data` — Training metadata (10.24 KB)

---

## 8. Backend — FastAPI Application

**Directory:** `backend/app/` (9 source files, ~1,445 lines)

### File Structure

```
backend/app/
├── __init__.py          (empty)
├── main.py              (50 lines)   — App entry point, CORS, table creation
├── config.py            (13 lines)   — Environment variables
├── database.py          (22 lines)   — SQLAlchemy engine + session
├── models/
│   └── models.py        (56 lines)   — 4 ORM table definitions
├── schemas/
│   └── schemas.py       (104 lines)  — 9 Pydantic models
├── routes/
│   └── api.py           (221 lines)  — 10 API endpoints
├── services/
│   ├── forecast_service.py  (715 lines) — Core ML engine
│   ├── insights_service.py  (116 lines) — Gemini AI integration
│   └── report_service.py    (148 lines) — PDF/CSV generation
└── utils/
    └── __init__.py      (empty)
```

### Application Startup Flow

```
main.py
  │
  ├─ 1. Create FastAPI app (title, version, description)
  │
  ├─ 2. Configure CORS middleware
  │     - Allow: localhost:5173, localhost:3000, Render frontend URL
  │     - Allow: all methods (GET, POST, PUT, DELETE)
  │     - Allow: all headers
  │     - Allow: credentials (cookies, auth headers)
  │
  ├─ 3. Create database tables (idempotent)
  │     Base.metadata.create_all(bind=engine)
  │     → Creates products, forecast_history, inventory, product_aliases tables
  │
  ├─ 4. Include API router
  │     app.include_router(router)
  │     → At this point, routes/api.py is imported
  │
  ├─ 5. Module-level service initialization (happens once)
  │     forecast_service = ForecastService(MODEL_DIR, DATASET_DIR)
  │     → Loads processed_sales.parquet (12 MB) into memory
  │     → Loads TFT checkpoint (2.08 MB) into memory
  │     → Loads PPO zip (0.13 MB) into memory
  │     → Loads encoders.joblib and dataset_info.json
  │
  └─ 6. Server is ready to handle requests
```

### Configuration (`config.py`)

```python
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./retail_forecaster.db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
DATASET_DIR = os.getenv("DATASET_DIR", "./datasets")
```

| Variable | Default | Purpose |
|----------|---------|---------|
| DATABASE_URL | SQLite fallback | SQLAlchemy connection string |
| GEMINI_API_KEY | Empty | Google Gemini API key for AI insights |
| MODEL_DIR | ./models | Where trained models are stored |
| DATASET_DIR | ./datasets | Where training data is stored |

---

## 9. Backend — Forecast Service (Core Engine)

**File:** `backend/app/services/forecast_service.py` (715 lines)

This is the most critical file in the entire project. It orchestrates all ML inference, feature engineering, pricing optimization, and inventory calculations.

### Class Initialization

```python
class ForecastService:
    def __init__(self, model_dir, dataset_dir):
        self.tft_model = None      # PyTorch TFT model
        self.ppo_model = None      # Custom PPO wrapper
        self.encoders = None       # LabelEncoders for store/item
        self.dataset_info = None   # Training metadata
        self.sales_data = None     # Full historical DataFrame
        self._load_data()          # Search 4 paths for dataset
        self._load_models()        # Search 4 paths for models
```

### Multi-Path Data/Model Discovery

Both `_load_data()` and `_load_models()` search 4 different paths to handle different deployment layouts:

```python
paths = [
    self.dataset_dir,                              # From config (./datasets)
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets"),  # Relative to service file
    os.path.join(os.getcwd(), "datasets"),         # CWD
    os.path.join(os.getcwd(), "..", "datasets"),   # Parent of CWD
]
```

This allows the same code to work on:
- Local development (CWD = project root)
- Docker (CWD = /app)
- Render (CWD = /opt/render/project/src)

### Custom PPO Loader (`_load_ppo_from_zip`)

Standard Stable Baselines3 `PPO.load()` fails with torch 2.12+ due to `weights_only=True` default in `torch.load()`. The custom loader works around this:

```python
def _load_ppo_from_zip(self, zip_path):
    # 1. Extract .pth files from the zip
    with zipfile.ZipFile(zip_path, 'r') as zf:
        policy_data = torch.load(io.BytesIO(zf.read('policy.pth')), weights_only=False)
        pytorch_vars = torch.load(io.BytesIO(zf.read('pytorch_variables.pth')), weights_only=False)

    # 2. Create a fresh ActorCriticPolicy
    obs_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
    act_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
    policy = ActorCriticPolicy(obs_space, act_space, lr_schedule=lambda _: 3e-4)

    # 3. Load weights
    policy.load_state_dict(policy_data, strict=False)
    for k, v in pytorch_vars.items():
        setattr(policy, k, v)  # Sets log_std

    # 4. Wrap in custom loader class
    class _PPOLoader:
        def predict(self, obs, deterministic=True):
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32)
                latent_pi, _ = self.policy.mlp_extractor(obs_t)
                mean_actions = self.policy.action_net(latent_pi)
                if deterministic:
                    action = mean_actions
                else:
                    action = mean_actions + torch.randn_like(mean_actions) * self.policy.log_std.exp()
                return action.clamp(-5.0, 5.0).numpy(), None

    return _PPOLoader(policy)
```

**Why this is needed:** The `strict=False` and manual attribute setting (`log_std`) handle version mismatches between SB3 versions. The `_PPOLoader` class mimics SB3's `predict()` interface so it can be used interchangeably.

### On-the-Fly Feature Engineering (`_create_features`)

At prediction time, the service creates the same 30+ features that the TFT model was trained on, using only the available historical data for the specific item+store:

```python
def _create_features(self, df):
    # Calendar features (from date column)
    df["year"], df["month"], df["day"] = df["date"].dt.year, .month, .day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Lag features (sales shifted back 7, 14, 28 days)
    for lag in [7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby(["store_id", "item_id"])["sales"].shift(lag)

    # Rolling features (shifted to avoid leakage)
    for window in [7, 14, 28]:
        df[f"rolling_mean_{window}"] = (
            df.groupby(["store_id", "item_id"])["sales"]
            .transform(lambda x: x.shift(1).rolling(window).mean())
        )

    # Price features
    df["price_change"] = df.groupby(["store_id", "item_id"])["price"].pct_change()
    df["price_rolling_mean_7"] = (
        df.groupby(["store_id", "item_id"])["price"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    # Time index (sequential integer for TFT)
    df["time_idx"] = df["date"].map(date_to_idx)

    # Drop rows with NaN in critical features (first 28 rows)
    df = df.dropna(subset=["lag_28", "rolling_mean_28"])
    return df
```

### TFT Prediction (`_predict_with_tft`)

```python
def _predict_with_tft(self, df, forecast_days):
    # 1. Encode categorical IDs
    df["store_id_enc"] = le_store.transform(df["store_id"]).astype(str)
    df["item_id_enc"] = le_item.transform(df["item_id"]).astype(str)
    df["group_id"] = df["store_id_enc"] + "_" + df["item_id_enc"]

    # 2. Validate minimum data size
    if len(df) < max_encoder_length + max_prediction_length:
        return None  # Not enough data

    # 3. Create TimeSeriesDataSet (reconstructs training format)
    dataset = TimeSeriesDataSet(df, time_idx="time_idx", target="sales", ...)

    # 4. Create prediction dataloader
    dataloader = TimeSeriesDataSet.from_dataset(
        dataset, df, predict=True, stop_randomization=True
    ).to_dataloader(train=False, batch_size=1)

    # 5. Run inference
    with torch.no_grad():
        raw_prediction = self.tft_model.predict(dataloader, mode="prediction")
        forecast_values = raw_prediction[0].cpu().numpy().tolist()
        forecast_values = [max(0, v) for v in forecast_values]  # Clamp negatives

    return {"forecast_values": forecast_values[:forecast_days], "model_used": "tft"}
```

### Statistical Fallback (`_predict_with_stats`)

When TFT is unavailable or fails:

```python
def _predict_with_stats(self, df, forecast_days):
    recent = df.tail(28)  # Last 28 days
    avg_sales = recent["sales"].mean()
    std_sales = recent["sales"].std()

    # Linear trend: (last_7_avg - first_7_avg) / 7
    trend = (recent["sales"].tail(7).mean() - recent["sales"].head(7).mean()) / 7

    forecast_values = []
    for i in range(forecast_days):
        base = avg_sales + trend * (i + 1)
        forecast_values.append(max(0, base))

    return {"forecast_values": forecast_values, "model_used": "statistical"}
```

### Forecast Extension (`_extend_forecast`)

When TFT produces fewer days than requested (e.g., TFT gives 7 days but user wants 14):

```python
def _extend_forecast(self, base_values, target_days):
    n_base = len(base_values)
    recent_trend = (base_values[-1] - base_values[0]) / (n_base - 1)

    extended = base_values.copy()
    for i in range(target_days - n_base):
        decay = 0.95 ** (i + 1)  # Exponential decay
        next_val = extended[-1] + recent_trend * decay
        extended.append(max(0, round(next_val, 1)))

    return extended[:target_days]
```

**Why decay?** Linear extrapolation without decay would produce unrealistically large values far into the future. The 0.95 decay factor makes the trend gradually flatten, which is more realistic for demand forecasting.

### Pricing Recommendation (`_get_pricing_recommendation`)

```python
def _get_pricing_recommendation(self, avg_sales, base_price, forecast_days, recent_df, promotion):
    if self.ppo_model:
        # Build 6-dim observation
        obs = np.array([
            recent_avg_sales / 200.0,
            price_change / 50.0,
            current_price / 50.0,
            is_promo,
            recent_avg_sales / 200.0,
            recent_avg_price / 50.0,
        ], dtype=np.float32)

        # Get PPO action
        action, _ = self.ppo_model.predict(obs.reshape(1, -1), deterministic=True)
        price_adjustment = float(action[0])
        suggested_price = max(1.0, min(50.0, current_price + price_adjustment))
        pricing_model = "ppo"
    else:
        # Rule-based fallback
        suggested_price = current_price
        pricing_model = "rule_based"

    # Compute revenue and profit with elasticity
    price_ratio = suggested_price / current_price
    expected_sales = avg_sales * (price_ratio ** -1.2)
    expected_revenue = expected_sales * suggested_price
    cost = current_price * 0.4  # 40% of price
    expected_profit = expected_sales * (suggested_price - cost)

    return {
        "suggested_price": round(suggested_price, 2),
        "pricing_model": pricing_model,
        "expected_revenue": round(expected_revenue, 2),
        "expected_profit": round(expected_profit, 2),
        "price_elasticity": -1.2,
        "cost": round(cost, 2),
    }
```

### Main Predict Method (Orchestrator)

```python
def predict(self, item_id, store_id, forecast_days, current_inventory, price, promotion, holiday):
    # Step 1: Get historical data for this item+store
    df = self._get_item_data(item_id, store_id)
    if df is None or len(df) < 50:
        return self._generate_mock_prediction(...)

    # Step 2: Create features (30+ engineered features)
    df = self._create_features(df)
    if len(df) < 30:
        return self._generate_mock_prediction(...)

    # Step 3: Try TFT, fallback to statistical
    tft_result = self._predict_with_tft(df, forecast_days)
    if tft_result and tft_result.get("forecast_values"):
        forecast_values = tft_result["forecast_values"]
        model_used = "tft"
    else:
        stats_result = self._predict_with_stats(df, forecast_days)
        forecast_values = stats_result["forecast_values"]
        model_used = "statistical"

    # Step 4: Apply promotion/holiday multipliers
    if promotion:
        forecast_values = [v * 1.15 for v in forecast_values]   # +15%
    if holiday:
        forecast_values = [v * 1.25 for v in forecast_values]   # +25%

    # Step 5: Compute summary metrics
    predicted_demand = sum(forecast_values)
    confidence = min(0.95, max(0.6, 1 - (std_sales / (avg_sales + 1)) * 0.5))

    # Step 6: Get pricing recommendation
    pricing = self._get_pricing_recommendation(avg_sales, base_price, forecast_days, df, promotion)

    # Step 7: Compute inventory metrics
    safety_stock = int(1.65 * std_sales * np.sqrt(7))  # 95% service level, 7-day lead time
    daily_sales = predicted_demand / forecast_days
    days_until_stockout = int(current_inventory / daily_sales) if daily_sales > 0 else 999
    reorder_qty = max(0, int(predicted_demand + safety_stock - current_inventory))

    # Step 8: Compute feature importance
    shap_values = self._compute_feature_importance(df, avg_sales)

    return { ... complete result dict ... }
```

### Safety Stock Formula

```python
safety_stock = 1.65 * std_sales * sqrt(lead_time)
```

- **1.65** = Z-score for 95% service level (from normal distribution table)
- **std_sales** = Standard deviation of daily sales (demand variability)
- **sqrt(7)** = Square root of 7-day lead time

This is the standard formula used in supply chain management. It ensures that with 95% probability, the store won't run out of stock during the 7-day replenishment lead time.

### Mock/Fallback Prediction

When no data or models are available:

```python
def _generate_mock_prediction(self, item_id, store_id, forecast_days, current_inventory, price):
    base_demand = 50.0  # Flat 50 units/day
    current_price = price if price else 15.0
    confidence = 0.75
    model_used = "mock"
    pricing_model = "mock"

    # Generates flat forecast, hardcoded SHAP values, and current timestamps
    return { ... }
```

### Feature Importance (Data-Driven, Not True SHAP)

```python
def _compute_feature_importance(self, df, avg_sales):
    # 1. Autocorrelation: How correlated is today with yesterday?
    autocorr = abs(df["sales"].autocorr(lag=1))

    # 2. Price variance: How much does price fluctuate?
    price_cv = min(0.35, df["price"].std() / (df["price"].mean() + 1))

    # 3. Promotion effect: Sales lift during promotions
    promo_effect = min(0.25, abs(promo_avg_sales - non_promo_avg_sales) / (avg_sales + 1))

    # 4. Weekend effect: Sales difference on weekends
    weekend_effect = min(0.20, abs(weekend_avg - weekday_avg) / (avg_sales + 1))

    # Normalize to sum to 1.0
    total = autocorr + price_cv + promo_effect + weekend_effect + 0.1 + 0.1
    return {
        "Previous Sales": round(autocorr / total, 2),
        "Price": round(price_cv / total, 2),
        "Rolling Mean": round(autocorr * 0.4 / total, 2),
        "Promotion": round(promo_effect / total, 2),
        "Weekend": round(weekend_effect / total, 2),
        "Day of Week": round(0.1 / total, 2),
    }
```

---

## 10. Backend — API Endpoints

**File:** `backend/app/routes/api.py` (221 lines)

### Complete Endpoint Reference

| # | Method | Path | Input | Output | Description |
|---|--------|------|-------|--------|-------------|
| 1 | GET | `/` | — | `{message, version, status}` | Root status |
| 2 | GET | `/api/health` | — | `{status, model_dir_exists, dataset_dir_exists}` | Health check |
| 3 | POST | `/api/predict` | `PredictRequest` | `ForecastResult` | **Main prediction** |
| 4 | GET | `/api/products` | — | `List[ProductResponse]` | Products from DB |
| 5 | GET | `/api/products/all` | — | `list[dict]` | Products from dataset |
| 6 | GET | `/api/stores` | — | `List[StoreResponse]` | Stores |
| 7 | GET | `/api/history` | `?limit=20` | `List[HistoryResponse]` | Past predictions |
| 8 | GET | `/api/analytics` | — | `AnalyticsResponse` | Dashboard stats |
| 9 | GET | `/api/heatmap` | — | `{stores, days, data}` | Day-of-week heatmap |
| 10 | POST | `/api/report` | `ReportRequest` | `{file_path, format}` | Generate report |

### Endpoint 3: POST /api/predict (Detailed Flow)

```
Request:
{
  "item_id": "item_1",
  "store_id": "store_1",
  "forecast_days": 7,
  "current_inventory": 100,
  "price": 15.0,
  "promotion": false,
  "holiday": false
}

Processing:
  1. Pydantic validates request body
  2. forecast_service.predict() runs full ML pipeline
  3. insights_service.generate_insights() calls Gemini AI (or generates mock)
  4. Product lookup from DB for demo_name
  5. ForecastHistory record saved to database
  6. Response serialized as ForecastResult

Response:
{
  "prediction_id": 14,
  "item_id": "item_1",
  "store_id": "store_1",
  "forecast_days": 7,
  "predicted_demand": 337.7,
  "confidence": 0.889,
  "revenue": 5129.35,
  "suggested_price": 15.19,
  "current_inventory": 100,
  "safety_stock": 46,
  "days_until_stockout": 2,
  "reorder_quantity": 283,
  "price_elasticity": -1.2,
  "expected_profit": 2966.65,
  "cost": 6.0,
  "pricing_model": "ppo",
  "model_used": "tft",
  "forecast_data": {
    "dates": ["2024-01-01", "2024-01-02", ...],
    "values": [47.7, 54.7, 56.4, 51.7, 45.0, 40.3, 41.8]
  },
  "shap_values": {
    "Previous Sales": 0.17,
    "Price": 0.04,
    "Rolling Mean": 0.07,
    "Promotion": 0.33,
    "Weekend": 0.26,
    "Day of Week": 0.13
  },
  "ai_summary": "{...JSON string of insights...}"
}
```

### Endpoint 8: GET /api/analytics

Aggregates:
- **total_products**: Count from DB, or from dataset, or defaults to 50
- **total_stores**: Distinct store count
- **avg_daily_sales**: Average predicted_demand across all forecasts
- **total_forecasts**: Count of forecast_history records
- **latest_forecast**: Most recent prediction
- **inventory_health**: Buckets into healthy (>14 days), warning (7-14), critical (<7)

---

## 11. Backend — Database Schema

**File:** `backend/app/models/models.py` (56 lines)

### Tables

#### `products` — Master Product Catalog
```sql
CREATE TABLE products (
    product_id   INT AUTO_INCREMENT PRIMARY KEY,
    item_id      VARCHAR(50) NOT NULL,
    demo_name    VARCHAR(100),
    category     VARCHAR(50),
    department   VARCHAR(50),
    store_id     VARCHAR(20),
    state_id     VARCHAR(10)
);
```

#### `forecast_history` — Prediction Audit Log
```sql
CREATE TABLE forecast_history (
    prediction_id       INT AUTO_INCREMENT PRIMARY KEY,
    timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP,
    product_id          INT,
    item_id             VARCHAR(50),
    store_id            VARCHAR(20),
    forecast_days       INT,
    predicted_demand    FLOAT,
    confidence          FLOAT,
    revenue             FLOAT,
    suggested_price     FLOAT,
    current_inventory   INT,
    safety_stock        INT,
    days_until_stockout INT,
    reorder_quantity    INT,
    ai_summary          TEXT,
    forecast_data       JSON,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
```

#### `inventory` — Current Stock Levels
```sql
CREATE TABLE inventory (
    inventory_id  INT AUTO_INCREMENT PRIMARY KEY,
    product_id    INT,
    item_id       VARCHAR(50),
    store_id      VARCHAR(20),
    current_stock INT DEFAULT 100,
    safety_stock  INT DEFAULT 20,
    last_updated  DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
```

#### `product_aliases` — Human-Readable Names
```sql
CREATE TABLE product_aliases (
    alias_id  INT AUTO_INCREMENT PRIMARY KEY,
    item_id   VARCHAR(50) NOT NULL UNIQUE,
    demo_name VARCHAR(100) NOT NULL
);

-- 30 seeded items:
INSERT INTO product_aliases VALUES
(1, 'item_1', 'Rice (Basmati)'),
(2, 'item_2', 'Milk (Fresh)'),
(3, 'item_3', 'Bread (Whole Wheat)'),
...
(30, 'item_50', 'Canned Beans');
```

### Relationships
- `forecast_history.product_id` → `products.product_id` (many-to-one)
- `inventory.product_id` → `products.product_id` (many-to-one)
- `product_aliases.item_id` is a unique alternate key for product naming

---

## 12. Frontend — React Application

**Directory:** `frontend/` (~1,890 lines across 14 files)

### Technology Choices

| Technology | Why |
|-----------|-----|
| React 19 | Latest stable, good ecosystem |
| Vite 8 | Fastest build tool, instant HMR |
| Tailwind CSS 4 | Utility-first, no CSS files needed |
| Chart.js 4 | Lightweight, good for dashboards |
| React Router 7 | Client-side SPA routing |
| Axios | Promise-based HTTP client |
| No state mgmt lib | Simple enough for useState/useEffect |

### Application Shell (`App.jsx`)

```
┌─────────────┬──────────────────────────────────────────────┐
│             │  Header (sticky)                              │
│  Sidebar    │  "AI-Powered Retail Analytics"  [AI avatar]  │
│             ├──────────────────────────────────────────────┤
│  🏠 Home    │                                              │
│  📊 Forecast│         <Routes>                             │
│  📈 Analytics│          / → Dashboard                       │
│  📦 Inventory│          /forecast → Forecast                │
│  📄 Reports │          /analytics → Analytics               │
│             │          /inventory → Inventory                │
│  [◀/▶]     │          /reports → Reports                    │
│  collapse   │         </Routes>                             │
│             │                                              │
└─────────────┴──────────────────────────────────────────────┘
```

- **Sidebar:** Collapsible (w-64 expanded, w-16 collapsed), dark gray background
- **Header:** Sticky, shows current page name, "AI" avatar
- **Content:** Scrollable, padded area for page content

### API Layer (`api.js`)

```javascript
const API_BASE = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api`
  : '/api';

// 8 exported functions:
predict(data)          // POST /api/predict
getProducts()          // GET /api/products
getAllProducts()        // GET /api/products/all
getStores()            // GET /api/stores
getHistory(limit)      // GET /api/history?limit=N
getAnalytics()         // GET /api/analytics
getHeatmap()           // GET /api/heatmap
generateReport(data)   // POST /api/report
```

**Key design decision:** `VITE_API_URL` is injected at build time by Vite. If set, the frontend makes absolute requests to the backend URL. If not set (local dev), it uses relative `/api` which the Vite proxy forwards to localhost:8000.

---

## 13. Frontend — Page-by-Page Breakdown

### Page 1: Dashboard (`Dashboard.jsx`, 210 lines)

**Purpose:** Overview of system status and recent activity.

**Layout:**
1. **Hero Section** — Blue-to-purple gradient banner with CTA button
2. **Stats Grid** (4 cards) — Total Products (50), Available Stores (50), Avg Daily Sales, Total Forecasts
3. **Recent Forecasts** — Last 5 predictions with demand and confidence
4. **Inventory Health** — Green/Yellow/Red counts for stock health
5. **Features Grid** — TFT, PPO, Gemini AI feature cards

**Data Loading:**
```javascript
const [analyticsData, historyData] = await Promise.all([
  getAnalytics(),
  getHistory(5),
])
```

**Error Handling:** Falls back to hardcoded defaults if API fails:
```javascript
setAnalytics({
  total_products: 50,  // Fixed from 2500 (was showing product-store pairs)
  total_stores: 50,
  avg_daily_sales: 45.2,
  ...
})
```

### Page 2: Forecast (`Forecast.jsx`, 452 lines)

**Purpose:** Generate demand predictions for specific product-store combinations.

**Layout:**
1. **Configuration Form** — Product dropdown, Store dropdown, Forecast Period, Inventory, Price, Promotion, Holiday
2. **Results Section** (shown after prediction):
   - Key Metrics (5 cards): Demand, Confidence, Revenue, Price, Model
   - Forecast Chart (Line chart with area fill)
   - Feature Importance (SHAP-like horizontal bars)
   - Inventory Recommendation (stock levels, reorder qty)
   - Pricing Recommendation (suggested price, cost, profit, elasticity)
   - AI Business Insights (recommendations, risk analysis)

**Product Dropdown Deduplication:**
```javascript
// Backend returns 2500 product-store pairs (50 items × 50 stores)
// We need only 50 unique products for the dropdown
const seen = new Set()
const finalProducts = productsData.filter(p => {
  if (seen.has(p.item_id)) return false
  seen.add(p.item_id)
  return true
})
```

**Chart Configuration:**
```javascript
const forecastChart = {
  labels: dates.map(d => `${parts[1]}/${parts[2]}`),  // "01/15" format
  datasets: [{
    label: 'Predicted Demand',
    data: forecast_values,
    borderColor: '#3B82F6',           // Blue line
    backgroundColor: 'rgba(59, 130, 246, 0.1)',  // Light fill
    fill: true,                       // Area chart
    tension: 0.4,                     // Smooth curves
  }]
}
```

**SHAP Bar Component:**
```javascript
const SHAPBar = ({ name, value }) => (
  <div className="flex items-center gap-2">
    <span className="text-xs w-24 text-right">{name}</span>
    <div className="flex-1 bg-gray-100 rounded-full h-4 overflow-hidden">
      <div
        className="h-full bg-gradient-to-r from-blue-500 to-blue-600 rounded-full"
        style={{ width: `${value * 100}%` }}
      />
    </div>
    <span className="text-xs w-10">{(value * 100).toFixed(0)}%</span>
  </div>
)
```

### Page 3: Analytics (`Analytics.jsx`, 284 lines)

**Purpose:** Visual analytics dashboard with charts, heatmap, and history.

**Charts (5 total, all Chart.js):**
1. **Sales Trend** — Line chart of predicted demand over time
2. **Revenue Trend** — Bar chart of revenue over time
3. **Confidence Scores** — Line chart of model confidence
4. **Price Trends** — Line chart of suggested prices
5. **Inventory Health** — Doughnut chart (healthy/warning/critical)

**Heatmap (HTML table, not canvas):**
```javascript
// Renders a table where:
// - Rows = Days of week (Mon, Tue, ..., Sun)
// - Columns = Stores (store_1, store_2, ..., store_6)
// - Cell color = blue intensity based on avg sales
// - Higher sales = darker blue = more demand

const intensity = maxVal > 0 ? Math.min(val / maxVal, 1) : 0
<td style={{ backgroundColor: `rgba(59, 130, 246, ${intensity})` }}>
  {val.toFixed(0)}
</td>
```

**Data Source:** Real data from `GET /api/heatmap` endpoint, which computes average sales per store per day-of-week from the dataset.

### Page 4: Inventory (`Inventory.jsx`, 219 lines)

**Purpose:** Monitor stock levels and stockout risks.

**Data Derivation:**
```javascript
// Inventory data is derived from forecast history (not a dedicated API)
const inventoryMap = new Map()
history.forEach((h) => {
  const key = `${h.item_id}_${h.store_id}`
  if (!inventoryMap.has(key)) {
    inventoryMap.set(key, {
      item_id: h.item_id,
      store_id: h.store_id,
      current_inventory: h.current_inventory ?? Math.max(safetyStock, predicted * 2),
      safety_stock: h.safety_stock ?? Math.floor(predicted * 0.3),
      days_until_stockout: h.days_until_stockout ?? Math.floor(currentInventory / dailySales),
      reorder_quantity: h.reorder_quantity ?? Math.max(0, Math.floor(predicted * 0.5)),
    })
  }
})
```

**Status Logic:**
- **Critical:** `days_until_stockout < 7` — Red card border, red badge
- **Warning:** `days_until_stockout >= 7 && < 14` — Yellow border, yellow badge
- **Healthy:** `days_until_stockout >= 14` — No border, green badge

**StockBar Component:** Color-coded progress bar showing stock level relative to safety stock.

### Page 5: Reports (`Reports.jsx`, 241 lines)

**Purpose:** Generate and download PDF/CSV reports.

**Two Actions:**
1. **Generate PDF** — Calls `POST /api/report` with format="pdf"
2. **Export CSV** — Calls `POST /api/report` with format="csv"

**PDF Report Contents (from report_service.py):**
- Title page with project name
- Forecast summary text
- Key metrics table (12 rows)
- Demand forecast line chart (matplotlib-generated)
- Business recommendations
- Risk analysis

---

## 14. Bugs Found and Fixed

### Bug 1: `rolling_min` Used Wrong Aggregation

**File:** `ml/feature_engineering.py`

**Before (broken):**
```python
df[f"rolling_min_{window}"] = (
    df.groupby(["item_id", "store_id"])["sales"]
    .transform(lambda x: x.shift(1).rolling(window).std())  # BUG: .std() instead of .min()
)
```

**After (fixed):**
```python
df[f"rolling_min_{window}"] = (
    df.groupby(["item_id", "store_id"])["sales"]
    .transform(lambda x: x.shift(1).rolling(window).min())  # Correct: .min()
)
```

**Impact:** The `rolling_min_7` feature was actually computing rolling standard deviation, not rolling minimum. This gave the TFT model incorrect information about the lower bound of recent demand.

### Bug 2: `price_rolling_mean_7` Was Computing Sales Instead of Price

**File:** `ml/feature_engineering.py`

**Before (broken):**
```python
df["price_rolling_mean_7"] = (
    df.groupby(["item_id", "store_id"])["sales"]  # BUG: grouping by sales, not price
    .transform(lambda x: x.rolling(7, min_periods=1).mean())
)
```

**After (fixed):**
```python
df["price_rolling_mean_7"] = (
    df.groupby(["item_id", "store_id"])["price"]  # Correct: price column
    .transform(lambda x: x.rolling(7, min_periods=1).mean())
)
```

**Impact:** The price rolling mean was actually a sales rolling mean, duplicating information the model already had. The model couldn't learn price-demand relationships properly.

### Bug 3: PPO Observation Dimension Mismatch

**File:** `ml/predict.py`

**Before (broken):**
```python
obs_array = np.array([
    current_obs.get("avg_sales", 50) / 200.0,
    0.0,                                          # price_change = hardcoded 0
    current_obs.get("current_price", 9.99) / 50.0,
    current_obs.get("avg_sales", 50) / 200.0,    # DUPLICATE of first
    current_obs.get("current_price", 9.99) / 50.0, # DUPLICATE of third
], dtype=np.float32)  # Only 5 dimensions!
```

**After (fixed in forecast_service.py):**
```python
obs = np.array([
    recent_avg_sales / 200.0,    # Current sales
    price_change / 50.0,          # Price change
    current_price / 50.0,         # Current price
    is_promo,                     # Promotion flag
    recent_avg_sales / 200.0,     # 7-day avg sales
    recent_avg_price / 50.0,      # 7-day avg price
], dtype=np.float32)  # Full 6 dimensions
```

**Impact:** The PPO model expects 6-dimensional input. The 5-dimensional version would crash at runtime, falling back to rule-based pricing.

### Bug 4: Lightning Import Path

**File:** `ml/train_tft.py`

**Before:** `import pytorch_lightning as pl`
**After:** `import lightning.pytorch as pl`

**Reason:** PyTorch Forecasting 1.8.0 requires the new `lightning` package (renamed from `pytorch_lightning` in PyTorch Lightning 2.0+).

### Bug 5: Dashboard Showed 2500 Products

**File:** `frontend/src/pages/Dashboard.jsx`

**Before:**
```javascript
setAnalytics({
  total_products: 2500,  // Product-store pairs, not unique products
  ...
})
```

**After:**
```javascript
setAnalytics({
  total_products: 50,  // Actual unique product count
  ...
})
```

**Impact:** Users saw "2500 products" in the dashboard and "2500 products" label, but the dropdown only showed 50 products. This was confusing.

### Bug 6: Product Dropdown Showed 2500 Entries

**File:** `frontend/src/pages/Forecast.jsx` and `Reports.jsx`

**Before:** `getAllProducts()` returns all 2500 product-store pairs, creating 2500 dropdown entries.

**After:** Deduplication by `item_id`:
```javascript
const seen = new Set()
const finalProducts = productsData.filter(p => {
  if (seen.has(p.item_id)) return false
  seen.add(p.item_id)
  return true
})
```

**Impact:** Dropdown now shows exactly 50 unique products.

---

## 15. Deployment

### Option A: Render (Cloud, Free)

**Architecture:**
```
Frontend (Render Static Site) → Backend (Render Web Service) → SQLite (file-based)
```

**Steps:**
1. Push code to GitHub
2. Create Render Static Site for frontend
3. Create Render Web Service for backend
4. Set env vars: `VITE_API_URL`, `DATABASE_URL`

**Cost:** $0/month (free tier)
**Tradeoff:** Backend spins down after 15 min inactivity (30s cold start)

### Option B: Docker (Local/Server)

**Architecture:**
```
Nginx (port 80) → Backend (port 8000) → MySQL (port 3306)
```

**Commands:**
```bash
cd docker
docker-compose up --build
```

**What Docker Compose does:**
1. Builds backend image (Python 3.11-slim + pip install)
2. Builds frontend image (Node 20 build → Nginx)
3. Pulls MySQL 8.0 image
4. Starts MySQL → Backend → Frontend (in order)
5. Initializes database from `database/schema.sql`
6. Mounts models/ and datasets/ as bind mounts

**Nginx Configuration:**
```nginx
# Static files (React SPA)
location / {
    root /usr/share/nginx/html;
    try_files $uri $uri/ /index.html;  # SPA fallback
}

# API proxy
location /api/ {
    proxy_pass http://backend:8000/api/;  # Docker DNS
}
```

### Option C: Local Development

```bash
# Terminal 1: Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Terminal 2: Frontend
cd frontend
npm install
npm run dev
```

Vite dev server (port 5173) proxies `/api/*` to backend (port 8000).

---

## 16. API Reference

### Request/Response Schemas

#### PredictRequest
```json
{
  "item_id": "string (required)",
  "store_id": "string (required)",
  "forecast_days": "integer (default: 7)",
  "current_inventory": "integer (default: 100)",
  "promotion": "boolean (default: false)",
  "holiday": "boolean (default: false)",
  "price": "float (optional, auto-detect if null)"
}
```

#### ForecastResult
```json
{
  "prediction_id": "integer",
  "item_id": "string",
  "store_id": "string",
  "demo_name": "string or null",
  "forecast_days": "integer",
  "predicted_demand": "float (total units)",
  "confidence": "float (0.0-1.0)",
  "revenue": "float (dollars)",
  "suggested_price": "float (dollars)",
  "current_inventory": "integer",
  "safety_stock": "integer",
  "days_until_stockout": "integer",
  "reorder_quantity": "integer",
  "price_elasticity": "float",
  "expected_profit": "float (dollars)",
  "cost": "float (dollars)",
  "pricing_model": "string (ppo|rule_based|mock)",
  "model_used": "string (tft|statistical|mock)",
  "forecast_data": {
    "dates": ["2024-01-01", ...],
    "values": [47.7, 54.7, ...]
  },
  "shap_values": {
    "Previous Sales": 0.17,
    "Price": 0.04,
    ...
  },
  "ai_summary": "JSON string of insights"
}
```

---

## 17. File Inventory

### Complete Source Files

| File | Lines | Purpose |
|------|-------|---------|
| **ML Pipeline** | | |
| `ml/data_processing.py` | 66 | Dataset download and initial processing |
| `ml/feature_engineering.py` | 150 | 30+ feature creation |
| `ml/train_tft.py` | 337 | TFT model training |
| `ml/train_ppo.py` | 149 | PPO pricing agent training |
| `ml/predict.py` | 99 | Standalone prediction utilities |
| `ml/test_e2e.py` | 34 | End-to-end test |
| **Backend** | | |
| `backend/app/main.py` | 50 | App entry point |
| `backend/app/config.py` | 13 | Environment config |
| `backend/app/database.py` | 22 | SQLAlchemy setup |
| `backend/app/models/models.py` | 56 | ORM table definitions |
| `backend/app/schemas/schemas.py` | 104 | Pydantic models |
| `backend/app/routes/api.py` | 221 | API endpoints |
| `backend/app/services/forecast_service.py` | 715 | Core ML engine |
| `backend/app/services/insights_service.py` | 116 | Gemini AI integration |
| `backend/app/services/report_service.py` | 148 | PDF/CSV generation |
| **Frontend** | | |
| `frontend/src/main.jsx` | 13 | React bootstrap |
| `frontend/src/App.jsx` | 91 | Layout + routing |
| `frontend/src/api.js` | 52 | API layer |
| `frontend/src/pages/Dashboard.jsx` | 210 | Dashboard |
| `frontend/src/pages/Forecast.jsx` | 452 | Prediction form + results |
| `frontend/src/pages/Analytics.jsx` | 284 | Charts + heatmap |
| `frontend/src/pages/Inventory.jsx` | 219 | Stock monitoring |
| `frontend/src/pages/Reports.jsx` | 241 | Report generation |
| **Configuration** | | |
| `render.yaml` | 19 | Render backend deployment |
| `render-frontend.yaml` | 13 | Render frontend deployment |
| `docker/Dockerfile.backend` | 12 | Backend Docker image |
| `docker/Dockerfile.frontend` | 14 | Frontend Docker image |
| `docker/docker-compose.yml` | 45 | Multi-service orchestration |
| `docker/nginx.conf` | 17 | Nginx routing config |
| `database/schema.sql` | 70 | Database schema + seeds |
| **Total** | **~3,400** | |

### Data Files

| File | Size | Purpose |
|------|------|---------|
| `datasets/retail_sales.csv` | 186 MB | Raw M5 dataset |
| `datasets/processed_sales.parquet` | 12 MB | Cleaned data |
| `datasets/featured_sales.parquet` | 203 MB | Feature-engineered data |
| `models/tft_model.ckpt` | 2.08 MB | Trained TFT model |
| `models/ppo_pricing.zip` | 0.13 MB | Trained PPO agent |
| `models/encoders.joblib` | ~1 KB | Label encoders |
| `models/dataset_info.json` | ~1 KB | Training metadata |

---

*Document generated for the AI-Dynamic Retail Demand Forecaster project.*
*Total project: ~3,400 lines of source code across 30+ files.*
*ML Models: TFT (2.08 MB) + PPO (0.13 MB).*
*Dataset: 4,565,000 rows across 50 stores and 50 products.*
