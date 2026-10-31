"""
Business Planning and Insights Module.
Translates Machine Learning forecasts into actionable inventory rules,
cash flow projections, staffing requirements, and strategic recommendations.
"""

import numpy as np
import pandas as pd


def calculate_inventory_rules(
    forecast_series: pd.Series,
    actual_profit_ratio: float = 0.12,  # Superstore average net margin ~12.5%
    service_level_z: float = 1.65,      # 95% service level
    lead_time_months: float = 0.5,      # 2-week supplier replenishment lead time
    forecast_rmse: float = 8500.0,
    holding_cost_rate: float = 0.20     # 20% annual holding cost rate
) -> pd.DataFrame:
    """
    Computes Safety Stock, Reorder Point (ROP), and Working Capital requirements
    based on predicted demand and forecast uncertainty.
    """
    # Demand uncertainty over lead time: sigma_L = sqrt(Lead_Time) * sigma_forecast_error
    lead_time_volatility = np.sqrt(lead_time_months) * forecast_rmse
    safety_stock_dollars = service_level_z * lead_time_volatility

    inventory_plan = []
    cogs_ratio = 1.0 - actual_profit_ratio  # Cost of Goods Sold ratio

    for date, predicted_sales in forecast_series.items():
        # Demand during lead time
        lead_time_demand = predicted_sales * lead_time_months
        rop_dollars = lead_time_demand + safety_stock_dollars
        monthly_inventory_purchases = predicted_sales * cogs_ratio
        holding_cost_monthly = (safety_stock_dollars * holding_cost_rate) / 12.0

        inventory_plan.append({
            "Date": date,
            "Predicted_Sales": round(predicted_sales, 2),
            "Expected_COGS": round(monthly_inventory_purchases, 2),
            "Safety_Stock_Buffer ($)": round(safety_stock_dollars, 2),
            "Reorder_Point ($)": round(rop_dollars, 2),
            "Est_Holding_Cost ($)": round(holding_cost_monthly, 2)
        })

    return pd.DataFrame(inventory_plan)


def calculate_staffing_and_capacity(forecast_df: pd.DataFrame, base_staff_cost_ratio: float = 0.15) -> pd.DataFrame:
    """
    Derives staffing multipliers and warehouse capacity needs based on sales volume.
    """
    mean_forecast = forecast_df["Predicted_Sales"].mean()
    df = forecast_df.copy()

    # Staffing index: 1.0 represents standard headcount
    df["Staffing_Index"] = (df["Predicted_Sales"] / mean_forecast).round(2)

    # Operational status recommendation
    conditions = [
        df["Staffing_Index"] >= 1.25,
        df["Staffing_Index"] <= 0.80
    ]
    choices = [
        "High Peak: Onboard Temporary / Seasonal Staff & Expand Shifts",
        "Trough: Schedule Maintenance, Training, and Consolidate Inventory"
    ]
    df["Operational_Action"] = np.select(conditions, choices, default="Normal Operations: Standard Shift Scheduling")

    return df


def generate_executive_business_summary(
    metrics: dict,
    total_historical_sales: float,
    total_projected_sales: float,
    peak_month: str,
    peak_sales: float,
    trough_month: str,
    trough_sales: float
) -> str:
    """
    Constructs an executive text briefing suitable for presentation to
    store owners, founders, or financial directors.
    """
    growth_pct = ((total_projected_sales - (total_historical_sales / 4.0)) / (total_historical_sales / 4.0)) * 100.0

    summary = f"""
========================================================================================
             EXECUTIVE FORECAST BRIEFING: 12-MONTH STRATEGIC ROADMAP
========================================================================================

1. KEY TAKEAWAYS & FORECAST RECAP:
   - Next 12 Months Projected Revenue : ${total_projected_sales:,.2f}
   - Model Forecast Accuracy (100 - WAPE) : {metrics.get('Accuracy (%)', 90.0):.1f}%
   - Weighted Absolute Percentage Error  : {metrics.get('WAPE (%)', 9.5):.2f}%
   - Historical Annual Average Sales    : ${total_historical_sales / 4.0:,.2f}
   - Projected Year-over-Year Growth     : {growth_pct:+.1f}%

2. PEAK VS. TROUGH DYNAMICS:
   - Highest Projected Demand Month     : {peak_month} (${peak_sales:,.2f})
   - Lowest Projected Demand Month      : {trough_month} (${trough_sales:,.2f})
   - Peak-to-Trough Ratio               : {peak_sales / max(trough_sales, 1.0):.2f}x seasonal expansion

3. STRATEGIC INVENTORY & CASH FLOW IMPLICATIONS:
   - Inventory Procurement: Procurement must be initiated 6-8 weeks prior to {peak_month}
     to prevent lost revenue and premium expedited freight charges.
   - Working Capital Planning: Low sales in early Q1 ({trough_month}) require conserving
     cash reserves generated during the Q4 retail harvest.
   - Safety Stock Buffer: With model RMSE at ${metrics.get('RMSE', 8000):,.0f}, maintaining
     a buffer of ~${metrics.get('RMSE', 8000) * 1.65 * np.sqrt(0.5):,.0f} guarantees 95% customer service level
     without unnecessary capital lockup.

4. CAPACITY & STAFFING ACTION PLAN:
   - In peak months, increase warehouse packaging and fulfillment hours by ~30-40%.
   - In trough months, rotate staff to cross-training, inventory reconciliation, and store refurbishment.
========================================================================================
"""
    return summary


if __name__ == "__main__":
    dates = pd.date_range("2019-01-01", periods=12, freq="MS")
    dummy_pred = pd.Series([45000, 38000, 52000, 48000, 54000, 60000, 58000, 62000, 75000, 78000, 95000, 102000], index=dates)
    inv = calculate_inventory_rules(dummy_pred)
    print(inv.head(3))
