from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.units import inch
import os
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ReportService:
    def __init__(self, reports_dir: str = "../reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    def generate_pdf(self, prediction: dict, insights: dict, filename: str = None) -> str:
        if not filename:
            filename = f"forecast_{prediction.get('item_id', 'unknown')}_{prediction.get('store_id', 'unknown')}.pdf"

        filepath = os.path.join(self.reports_dir, filename)
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        title_style = ParagraphStyle(
            "CustomTitle", parent=styles["Title"], fontSize=20, spaceAfter=20
        )
        elements.append(Paragraph("Retail Demand Forecast Report", title_style))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Forecast Summary", styles["Heading2"]))
        summary = insights.get("forecast_summary", "No summary available.")
        elements.append(Paragraph(summary, styles["Normal"]))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Key Metrics", styles["Heading2"]))
        metrics_data = [
            ["Metric", "Value"],
            ["Product", str(prediction.get("item_id", "N/A"))],
            ["Store", str(prediction.get("store_id", "N/A"))],
            ["Forecast Period", f"{prediction.get('forecast_days', 7)} days"],
            ["Predicted Demand", f"{prediction.get('predicted_demand', 0):.1f} units"],
            ["Confidence Score", f"{prediction.get('confidence', 0):.1%}"],
            ["Expected Revenue", f"${prediction.get('revenue', 0):.2f}"],
            ["Suggested Price", f"${prediction.get('suggested_price', 0):.2f}"],
            ["Current Inventory", f"{prediction.get('current_inventory', 0)} units"],
            ["Safety Stock", f"{prediction.get('safety_stock', 0)} units"],
            ["Days Until Stockout", f"{prediction.get('days_until_stockout', 0)} days"],
            ["Reorder Quantity", f"{prediction.get('reorder_quantity', 0)} units"],
        ]
        metrics_table = Table(metrics_data, colWidths=[2.5 * inch, 2.5 * inch])
        metrics_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3B82F6")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F6")]),
                ]
            )
        )
        elements.append(metrics_table)
        elements.append(Spacer(1, 12))

        chart_path = self._create_forecast_chart(prediction)
        if chart_path:
            elements.append(Paragraph("Demand Forecast Chart", styles["Heading2"]))
            elements.append(Image(chart_path, width=5 * inch, height=2.5 * inch))
            elements.append(Spacer(1, 12))

        elements.append(Paragraph("Business Recommendations", styles["Heading2"]))
        recs = insights.get("business_recommendations", [])
        for i, rec in enumerate(recs, 1):
            elements.append(Paragraph(f"{i}. {rec}", styles["Normal"]))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Risk Analysis", styles["Heading2"]))
        risk = insights.get("risk_analysis", {})
        elements.append(
            Paragraph(f"Overall Risk: {risk.get('overall_risk', 'N/A')}", styles["Normal"])
        )
        for r in risk.get("risks", []):
            elements.append(
                Paragraph(f"- {r.get('risk', '')}: {r.get('level', '')} - {r.get('mitigation', '')}", styles["Normal"])
            )

        doc.build(elements)

        if chart_path and os.path.exists(chart_path):
            os.remove(chart_path)

        return filepath

    def _create_forecast_chart(self, prediction: dict) -> str:
        forecast_data = prediction.get("forecast_data", {})
        dates = forecast_data.get("dates", [])
        values = forecast_data.get("values", [])

        if not dates or not values:
            return None

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(len(values)), values, marker="o", color="#3B82F6", linewidth=2)
        ax.fill_between(range(len(values)), values, alpha=0.1, color="#3B82F6")
        ax.set_title("Demand Forecast", fontsize=14, fontweight="bold")
        ax.set_xlabel("Days Ahead")
        ax.set_ylabel("Predicted Sales")
        ax.grid(True, alpha=0.3)

        short_dates = [d.split("-")[2] if len(d) >= 10 else d for d in dates]
        ax.set_xticks(range(len(short_dates)))
        ax.set_xticklabels(short_dates, rotation=45, fontsize=8)

        chart_path = os.path.join(self.reports_dir, "temp_chart.png")
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150)
        plt.close()

        return chart_path

    def generate_csv(self, prediction: dict) -> str:
        import csv

        filename = f"forecast_{prediction.get('item_id', 'unknown')}_{prediction.get('store_id', 'unknown')}.csv"
        filepath = os.path.join(self.reports_dir, filename)

        forecast_data = prediction.get("forecast_data", {})
        dates = forecast_data.get("dates", [])
        values = forecast_data.get("values", [])

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Predicted_Demand", "Item_ID", "Store_ID", "Confidence", "Revenue"])
            for date, val in zip(dates, values):
                writer.writerow([
                    date, val,
                    prediction.get("item_id", ""),
                    prediction.get("store_id", ""),
                    prediction.get("confidence", 0),
                    prediction.get("revenue", 0),
                ])

        return filepath
