from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey
from sqlalchemy.sql import func
from app.database import Base


class Product(Base):
    __tablename__ = "products"

    product_id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(String(50), nullable=False)
    demo_name = Column(String(100))
    category = Column(String(50))
    department = Column(String(50))
    store_id = Column(String(20))
    state_id = Column(String(10))


class ForecastHistory(Base):
    __tablename__ = "forecast_history"

    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, server_default=func.now())
    product_id = Column(Integer, ForeignKey("products.product_id"))
    item_id = Column(String(50))
    store_id = Column(String(20))
    forecast_days = Column(Integer)
    predicted_demand = Column(Float)
    confidence = Column(Float)
    revenue = Column(Float)
    suggested_price = Column(Float)
    current_inventory = Column(Integer)
    safety_stock = Column(Integer)
    days_until_stockout = Column(Integer)
    reorder_quantity = Column(Integer)
    ai_summary = Column(Text)
    forecast_data = Column(JSON)


class Inventory(Base):
    __tablename__ = "inventory"

    inventory_id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("products.product_id"))
    item_id = Column(String(50))
    store_id = Column(String(20))
    current_stock = Column(Integer, default=100)
    safety_stock = Column(Integer, default=20)
    last_updated = Column(DateTime, server_default=func.now(), onupdate=func.now())


class ProductAlias(Base):
    __tablename__ = "product_aliases"

    alias_id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(String(50), unique=True, nullable=False)
    demo_name = Column(String(100), nullable=False)
