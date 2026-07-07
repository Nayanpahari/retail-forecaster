import os
import google.generativeai as genai
from app.config import GEMINI_API_KEY


class InsightsService:
    def __init__(self):
        if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel("gemini-pro")
            self.available = True
        else:
            self.available = False

    def generate_insights(self, prediction: dict) -> dict:
        if self.available:
            return self._generate_with_gemini(prediction)
        return self._generate_mock_insights(prediction)

    def _generate_with_gemini(self, prediction: dict) -> dict:
        prompt = f"""You are a retail business analyst AI. Based on the following demand forecast, provide actionable business insights.

Forecast Data:
- Product: {prediction.get('item_id', 'N/A')}
- Store: {prediction.get('store_id', 'N/A')}
- Predicted Demand (next {prediction.get('forecast_days', 7)} days): {prediction.get('predicted_demand', 0):.1f} units
- Confidence Score: {prediction.get('confidence', 0):.1%}
- Expected Revenue: ${prediction.get('revenue', 0):.2f}
- Suggested Price: ${prediction.get('suggested_price', 0):.2f}
- Current Inventory: {prediction.get('current_inventory', 0)} units
- Days Until Stockout: {prediction.get('days_until_stockout', 0)} days
- Suggested Reorder: {prediction.get('reorder_quantity', 0)} units

Please provide:
1. FORECAST SUMMARY: A brief summary of what the forecast indicates
2. BUSINESS RECOMMENDATIONS: 3-5 actionable business recommendations
3. INVENTORY SUGGESTIONS: Specific inventory management advice
4. PRICING SUGGESTIONS: Pricing strategy recommendations
5. RISK ANALYSIS: Key risks and mitigation strategies

Format your response as JSON with keys: forecast_summary, business_recommendations, inventory_suggestions, pricing_suggestions, risk_analysis"""

        try:
            response = self.model.generate_content(prompt)
            text = response.text
            import json

            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(text[json_start:json_end])
        except Exception:
            pass

        return self._generate_mock_insights(prediction)

    def _generate_mock_insights(self, prediction: dict) -> dict:
        demand = prediction.get("predicted_demand", 50)
        confidence = prediction.get("confidence", 0.8)
        stockout_days = prediction.get("days_until_stockout", 30)
        reorder = prediction.get("reorder_quantity", 0)

        if stockout_days < 7:
            urgency = "CRITICAL"
            stockout_msg = f"Stock will run out in {stockout_days} days. Immediate reorder of {reorder} units required."
        elif stockout_days < 14:
            urgency = "HIGH"
            stockout_msg = f"Stock will last {stockout_days} days. Plan to reorder {reorder} units within the next 3 days."
        else:
            urgency = "NORMAL"
            stockout_msg = f"Current stock is adequate for {stockout_days} days. Reorder {reorder} units as a precaution."

        return {
            "forecast_summary": f"The AI model predicts a demand of {demand:.0f} units over the next {prediction.get('forecast_days', 7)} days with {confidence:.0%} confidence. {'Demand is trending upward.' if demand > 60 else 'Demand is stable.'} Expected revenue is ${prediction.get('revenue', 0):.2f}.",
            "business_recommendations": [
                f"{'Increase' if demand > 50 else 'Maintain'} inventory levels based on predicted demand of {demand:.0f} units",
                f"{'Consider promotional activities to boost sales' if demand < 30 else 'Current demand is healthy - maintain pricing strategy'}",
                f"Monitor competitor pricing and adjust the suggested price of ${prediction.get('suggested_price', 0):.2f} accordingly",
                f"{'Weekend sales tend to be higher - plan staffing accordingly' if confidence > 0.8 else 'Low confidence in prediction - gather more data before major decisions'}",
                f"Review supply chain lead times to ensure timely replenishment of {reorder} units",
            ],
            "inventory_suggestions": {
                "current_stock": prediction.get("current_inventory", 100),
                "safety_stock": prediction.get("safety_stock", 20),
                "suggested_reorder": reorder,
                "days_until_stockout": stockout_days,
                "urgency_level": urgency,
                "message": stockout_msg,
            },
            "pricing_suggestions": {
                "recommended_price": prediction.get("suggested_price", 9.99),
                "price_strategy": "Maintain current pricing" if confidence > 0.8 else "Consider dynamic pricing based on real-time demand",
                "expected_margin": f"{((prediction.get('suggested_price', 9.99) - 5.0) / prediction.get('suggested_price', 9.99) * 100):.1f}%",
                "note": "Price elasticity analysis suggests minor adjustments can impact revenue by 5-15%",
            },
            "risk_analysis": {
                "overall_risk": "Low" if stockout_days > 14 else "Medium" if stockout_days > 7 else "High",
                "risks": [
                    {
                        "risk": "Stockout Risk",
                        "level": "High" if stockout_days < 7 else "Low",
                        "mitigation": f"Maintain safety stock of {prediction.get('safety_stock', 20)} units",
                    },
                    {
                        "risk": "Demand Variability",
                        "level": "Medium" if confidence < 0.8 else "Low",
                        "mitigation": "Use rolling forecasts and weekly reviews",
                    },
                    {
                        "risk": "Price Sensitivity",
                        "level": "Medium",
                        "mitigation": "A/B test pricing changes in controlled batches",
                    },
                ],
            },
        }
