from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from app.database import get_db
from app.schemas.schemas import (
    PredictRequest, ForecastResult, ProductResponse, StoreResponse,
    HistoryResponse, AnalyticsResponse, ReportRequest
)
from app.services.forecast_service import ForecastService
from app.services.insights_service import InsightsService
from app.services.report_service import ReportService
from app.models.models import ForecastHistory, Product, Inventory
from app.config import MODEL_DIR, DATASET_DIR
import json

router = APIRouter()

forecast_service = ForecastService(MODEL_DIR, DATASET_DIR)
insights_service = InsightsService()
report_service = ReportService()


@router.post("/api/predict", response_model=ForecastResult)
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    prediction = forecast_service.predict(
        item_id=request.item_id,
        store_id=request.store_id,
        forecast_days=request.forecast_days,
        current_inventory=request.current_inventory,
        price=request.price,
        promotion=request.promotion,
        holiday=request.holiday,
    )

    if prediction is None:
        raise HTTPException(status_code=404, detail="No data found for this product/store combination")

    insights = insights_service.generate_insights(prediction)
    prediction["ai_summary"] = json.dumps(insights)

    demo_name = None
    product = db.query(Product).filter(
        Product.item_id == request.item_id,
        Product.store_id == request.store_id
    ).first()
    if product:
        demo_name = product.demo_name
    prediction["demo_name"] = demo_name

    history = ForecastHistory(
        product_id=product.product_id if product else None,
        item_id=request.item_id,
        store_id=request.store_id,
        forecast_days=request.forecast_days,
        predicted_demand=prediction["predicted_demand"],
        confidence=prediction["confidence"],
        revenue=prediction["revenue"],
        suggested_price=prediction["suggested_price"],
        current_inventory=request.current_inventory,
        safety_stock=prediction["safety_stock"],
        days_until_stockout=prediction["days_until_stockout"],
        reorder_quantity=prediction["reorder_quantity"],
        ai_summary=prediction["ai_summary"],
        forecast_data=json.dumps(prediction["forecast_data"]),
    )
    db.add(history)
    db.commit()
    db.refresh(history)
    prediction["prediction_id"] = history.prediction_id

    return ForecastResult(**prediction)


@router.get("/api/products", response_model=List[ProductResponse])
def get_products(db: Session = Depends(get_db)):
    products = db.query(Product).all()
    if not products:
        return []
    return [
        ProductResponse(
            product_id=p.product_id,
            item_id=p.item_id,
            demo_name=p.demo_name,
            category=p.category,
            department=p.department,
            store_id=p.store_id,
            state_id=p.state_id,
        )
        for p in products
    ]


@router.get("/api/products/all")
def get_all_products():
    products = forecast_service.get_products()
    return products


@router.get("/api/stores", response_model=List[StoreResponse])
def get_stores(db: Session = Depends(get_db)):
    stores = db.query(Product).with_entities(Product.store_id, Product.state_id).distinct().all()
    if not stores:
        ml_stores = forecast_service.get_stores()
        return [StoreResponse(store_id=s["store_id"], state_id=s["state_id"]) for s in ml_stores]
    return [StoreResponse(store_id=s.store_id, state_id=s.state_id) for s in stores]


@router.get("/api/history", response_model=List[HistoryResponse])
def get_history(limit: int = 20, db: Session = Depends(get_db)):
    records = (
        db.query(ForecastHistory)
        .order_by(ForecastHistory.timestamp.desc())
        .limit(limit)
        .all()
    )
    result = []
    for r in records:
        demo_name = None
        if r.product_id:
            product = db.query(Product).filter(Product.product_id == r.product_id).first()
            if product:
                demo_name = product.demo_name
        result.append(
            HistoryResponse(
                prediction_id=r.prediction_id,
                timestamp=r.timestamp,
                item_id=r.item_id,
                store_id=r.store_id,
                demo_name=demo_name,
                forecast_days=r.forecast_days,
                predicted_demand=r.predicted_demand,
                confidence=r.confidence,
                revenue=r.revenue,
                suggested_price=r.suggested_price,
                current_inventory=r.current_inventory,
                safety_stock=r.safety_stock,
                days_until_stockout=r.days_until_stockout,
                reorder_quantity=r.reorder_quantity,
                ai_summary=r.ai_summary,
            )
        )
    return result


@router.get("/api/analytics")
def get_analytics(db: Session = Depends(get_db)):
    total_products = db.query(Product).count()
    total_stores = db.query(Product.store_id).distinct().count()

    if total_products == 0:
        ml_products = forecast_service.get_products()
        if ml_products:
            total_products = len(set(p["item_id"] for p in ml_products))
            total_stores = len(set(p["store_id"] for p in ml_products))
        else:
            total_products = 50
            total_stores = 50

    total_forecasts = db.query(ForecastHistory).count()

    avg_sales = 0.0
    latest_forecast = None
    inventory_health = {"healthy": 0, "warning": 0, "critical": 0}

    if total_forecasts > 0:
        forecasts = db.query(ForecastHistory).all()
        avg_sales = sum(f.predicted_demand for f in forecasts) / total_forecasts

        latest = db.query(ForecastHistory).order_by(ForecastHistory.timestamp.desc()).first()
        if latest:
            latest_forecast = {
                "item_id": latest.item_id,
                "store_id": latest.store_id,
                "predicted_demand": latest.predicted_demand,
                "timestamp": str(latest.timestamp),
            }

        for f in forecasts:
            if f.days_until_stockout > 14:
                inventory_health["healthy"] += 1
            elif f.days_until_stockout > 7:
                inventory_health["warning"] += 1
            else:
                inventory_health["critical"] += 1

    return AnalyticsResponse(
        total_products=total_products,
        total_stores=total_stores,
        avg_daily_sales=round(avg_sales, 1),
        total_forecasts=total_forecasts,
        latest_forecast=latest_forecast,
        inventory_health=inventory_health,
    )


@router.get("/api/heatmap")
def get_heatmap():
    return forecast_service.get_heatmap_data()


@router.post("/api/report")
def generate_report(request: ReportRequest, db: Session = Depends(get_db)):
    prediction = forecast_service.predict(
        item_id=request.item_id,
        store_id=request.store_id,
        forecast_days=request.forecast_days,
        current_inventory=100,
        price=None,
    )

    if prediction is None:
        raise HTTPException(status_code=404, detail="No data found")

    insights = insights_service.generate_insights(prediction)

    if request.format == "csv":
        filepath = report_service.generate_csv(prediction)
    else:
        filepath = report_service.generate_pdf(prediction, insights)

    return {"file_path": filepath, "format": request.format}
