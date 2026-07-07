# AI-Dynamic Retail Demand Forecaster

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- MySQL 8.0+

### 1. Database Setup
```bash
mysql -u root -p < database/schema.sql
```

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your MySQL credentials and Gemini API key

# Start server
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 4. ML Pipeline (Optional - for training models)
```bash
cd ml
pip install -r requirements.txt
python data_processing.py
python feature_engineering.py
python train_tft.py
python train_ppo.py
```

### 5. Docker (Alternative)
```bash
cd docker
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/predict | Generate demand forecast |
| GET | /api/products | List all products |
| GET | /api/stores | List all stores |
| GET | /api/history | Get forecast history |
| GET | /api/analytics | Dashboard analytics |
| POST | /api/report | Generate PDF/CSV report |

## Project Structure

```
AI-Dynamic-Retail-Demand-Forecaster/
├── frontend/          # React + Vite + Tailwind
├── backend/           # FastAPI
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── routes/
│   │   └── services/
│   └── requirements.txt
├── ml/                # ML Pipeline
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train_tft.py
│   ├── train_ppo.py
│   └── predict.py
├── models/            # Saved models
├── datasets/          # M5 dataset
├── database/          # SQL schema
├── reports/           # Generated reports
├── docker/            # Docker configs
└── docs/              # Documentation
```

## Technology Stack

- **Frontend:** React, Vite, Tailwind CSS, Chart.js
- **Backend:** FastAPI, SQLAlchemy, Pydantic
- **ML:** Temporal Fusion Transformer, PPO Reinforcement Learning
- **AI Insights:** Google Gemini API
- **Database:** MySQL
- **Deployment:** Docker, Vercel, Render

## Features

- AI-powered demand forecasting (TFT)
- Dynamic pricing optimization (PPO RL)
- Natural language business insights (Gemini)
- SHAP feature importance explainability
- Interactive dashboard with charts
- Inventory management recommendations
- PDF/CSV report generation
- Responsive design
