from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class PredictRequest(BaseModel):
    item_id: str
    store_id: str
    forecast_days: int = 7
    current_inventory: int = 100
    promotion: bool = False
    holiday: bool = False
    price: Optional[float] = None


class ForecastResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction_id: Optional[int] = None
    item_id: str
    store_id: str
    demo_name: Optional[str] = None
    forecast_days: int
    predicted_demand: float
    confidence: float
    revenue: float
    suggested_price: float
    current_inventory: int
    safety_stock: int
    days_until_stockout: int
    reorder_quantity: int
    price_elasticity: Optional[float] = None
    expected_profit: Optional[float] = None
    cost: Optional[float] = None
    pricing_model: Optional[str] = None
    model_used: Optional[str] = None
    ai_summary: Optional[str] = None
    forecast_data: dict
    shap_values: Optional[dict] = None


class ProductResponse(BaseModel):
    product_id: int
    item_id: str
    demo_name: Optional[str]
    category: str
    department: str
    store_id: str
    state_id: str


class StoreResponse(BaseModel):
    store_id: str
    state_id: str


class HistoryResponse(BaseModel):
    prediction_id: int
    timestamp: datetime
    item_id: str
    store_id: str
    demo_name: Optional[str]
    forecast_days: int
    predicted_demand: float
    confidence: float
    revenue: float
    suggested_price: float
    current_inventory: Optional[int] = None
    safety_stock: Optional[int] = None
    days_until_stockout: Optional[int] = None
    reorder_quantity: Optional[int] = None
    ai_summary: Optional[str]


class AnalyticsResponse(BaseModel):
    total_products: int
    total_stores: int
    avg_daily_sales: float
    total_forecasts: int
    latest_forecast: Optional[dict]
    inventory_health: dict


class InventoryRecommendation(BaseModel):
    current_stock: int
    expected_sales: float
    suggested_reorder: int
    safety_stock: int
    days_until_stockout: int


class PricingRecommendation(BaseModel):
    recommended_price: float
    current_price: float
    expected_revenue: float
    expected_profit: float
    price_elasticity: float


class ReportRequest(BaseModel):
    item_id: str
    store_id: str
    forecast_days: int
    format: str = "pdf"
