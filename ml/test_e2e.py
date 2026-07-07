import sys, os, json
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, 'D:\\major project v2\\backend\\app')

from services.forecast_service import ForecastService
import pandas as pd, numpy as np

service = ForecastService(model_dir='D:\\major project v2\\models', dataset_dir='D:\\major project v2\\datasets')
print(f"Loaded TFT: {service.tft_model is not None}")
print(f"Loaded PPO: {service.ppo_model is not None}")

result = service.predict(
    item_id='item_1', store_id='store_1',
    forecast_days=14, current_inventory=100,
    price=15.0, promotion=False, holiday=False
)

print(f"Model used: {result['model_used']}")
print(f"Pricing model: {result['pricing_model']}")
print(f"Sales forecast len: {len(result['forecast_data']['values'])} days")
print(f"Avg forecast: {np.mean(result['forecast_data']['values']):.2f}")
print(f"Predicted demand: {result['predicted_demand']}")
print(f"Revenue: {result['revenue']}")
print(f"Suggested price: {result['suggested_price']}")
print(f"Safety stock: {result['safety_stock']}")
print(f"Days until stockout: {result['days_until_stockout']}")
print(f"Reorder qty: {result['reorder_quantity']}")
print(f"Confidence: {result['confidence']}")
print(f"SHAP values: {result['shap_values']}")
print(f"Cost: {result['cost']}")
print(f"Price elasticity: {result['price_elasticity']}")
print(f"Expected profit: {result['expected_profit']}")
print(f"Forecast dates: {result['forecast_data']['dates'][:3]}...")
print("END-TO-END TEST PASSED")
