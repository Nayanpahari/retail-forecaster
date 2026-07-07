# Deployment Guide

## Architecture

```
Frontend (Render Static Site) → Backend (Render Web Service) → MySQL (Railway)
```

## Step 1: Push to GitHub

```bash
cd "D:\major project v2"
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/AI-Dynamic-Retail-Demand-Forecaster.git
git push -u origin main
```

## Step 2: Set Up Railway MySQL

1. Go to https://railway.com and sign up with GitHub
2. Click **Deploy a New Project** → **MySQL**
3. Wait for MySQL to be provisioned (~1 min)
4. Go to the MySQL service → **Variables** tab
5. Set `MYSQL_ROOT_PASSWORD` to a secure password
6. Set `MYSQL_DATABASE` to `retail_forecaster`
7. Go to **Settings** → **Volumes** → Add a volume at `/var/lib/mysql`
8. Go to **Connect** tab → Copy the **TCP Proxy** host and port
9. Note down: Host, Port, User (root), Password, Database

## Step 3: Deploy Backend on Render

1. Go to https://dashboard.render.com
2. Click **New** → **Web Service**
3. Connect your GitHub repo
4. Configure:

| Field | Value |
|-------|-------|
| Name | `retail-forecaster-api` |
| Region | Oregon (or closest) |
| Branch | `main` |
| Runtime | Python 3 |
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Instance Type | Free |

5. Go to **Environment** tab and add:

| Key | Value |
|-----|-------|
| DATABASE_URL | `mysql+pymysql://root:YOUR_PASSWORD@HOST:PORT/retail_forecaster` |
| GEMINI_API_KEY | Your Gemini API key |
| MODEL_DIR | `../models` |
| DATASET_DIR | `../datasets` |
| FRONTEND_URL | `https://retail-forecaster-frontend.onrender.com` |

6. Click **Create Web Service**
7. Wait for deployment (~5-10 min)

**Note:** The dataset (retail_sales.csv ~195MB) is too large to deploy with the app. The backend will use mock predictions when the dataset isn't available. To use real predictions, you'll need to download the dataset on startup or use a smaller sample.

## Step 4: Deploy Frontend on Render

1. Go to https://dashboard.render.com
2. Click **New** → **Static Site**
3. Connect your GitHub repo
4. Configure:

| Field | Value |
|-------|-------|
| Name | `retail-forecaster-frontend` |
| Branch | `main` |
| Root Directory | `frontend` |
| Build Command | `npm install && npm run build` |
| Publish Directory | `dist` |

5. Go to **Environment** tab and add:

| Key | Value |
|-----|-------|
| VITE_API_URL | `https://retail-forecaster-api.onrender.com` |

6. Go to **Redirects/Rewrites** tab and add:

| Type | Source | Destination |
|------|--------|-------------|
| Rewrite | `/*` | `/index.html` |

7. Click **Create Static Site**
8. Wait for deployment (~2-3 min)

## Step 5: Verify

1. Open the frontend URL
2. Check that the dashboard loads
3. Try a forecast prediction
4. Check that all pages work

## Troubleshooting

### "CORS Error" in Browser Console
- Make sure `FRONTEND_URL` is set correctly on the backend
- Make sure the frontend URL matches exactly (including https://)

### "Database Connection Error"
- Check that `DATABASE_URL` is correct on the backend
- Make sure Railway MySQL has a volume attached
- Check that the TCP Proxy host/port are correct

### Frontend Shows "0 Products" / "0 Stores"
- This is expected if the dataset isn't loaded
- The backend uses mock predictions when dataset is unavailable
- To use real data, you'd need to upload the dataset to cloud storage

### Backend Sleeps After Inactivity
- Render free tier spins down after 15 min of inactivity
- First request after sleep takes 30-60 seconds
- Consider upgrading to a paid plan for always-on service

## Cost Estimate

| Service | Plan | Cost |
|---------|------|------|
| Render Backend | Free | $0/month |
| Render Frontend | Free | $0/month |
| Railway MySQL | Free tier | ~$1-3/month |
| **Total** | | **~$1-3/month** |

## Optional: Keep-Alive Script

To prevent Render free tier from sleeping, you can use a cron job to ping the backend every 10 minutes:

```bash
# Add to cron (every 10 minutes)
*/10 * * * * curl -s https://retail-forecaster-api.onrender.com/api/health > /dev/null
```
