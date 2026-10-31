"""
Master Pipeline Script for Sales & Demand Forecasting System.
Executes end-to-end:
1. Data Ingestion & Cleaning
2. Time Aggregations
3. Feature Engineering
4. Multi-Model Validation & Error Analysis
5. Forward 12-Month Out-of-Sample Forecasting
6. Category Breakdown Demand Forecasting
7. Business Inventory, Safety Stock & Working Capital Sizing
8. High-Resolution Visualizations Generation
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_prep import load_and_clean_superstore_data, aggregate_sales_data
from src.feature_engineering import create_time_features
from src.models import (
    run_model_comparison,
    generate_future_forecast,
    HoltWintersForecaster
)
from src.evaluation import calculate_metrics
from src.business_insights import (
    calculate_inventory_rules,
    calculate_staffing_and_capacity,
    generate_executive_business_summary
)
from src.visualizations import (
    plot_sales_trends_and_seasonality,
    plot_seasonal_decomposition,
    plot_model_comparison,
    plot_test_validation_actual_vs_predicted,
    plot_future_forecast_with_ci,
    plot_category_forecast,
    plot_inventory_and_cashflow_plan
)


def forecast_by_category(df_raw: pd.DataFrame, steps: int = 12) -> pd.DataFrame:
    """
    Fits time-series models for each product category (Technology, Furniture, Office Supplies)
    and forecasts future 12-month demand.
    """
    categories = df_raw["Category"].unique()
    all_cat_forecasts = []

    for cat in sorted(categories):
        cat_subset = df_raw[df_raw["Category"] == cat]
        monthly_cat = aggregate_sales_data(cat_subset, freq="MS")
        ts = monthly_cat.set_index("Date")["Sales"]

        # Fit Holt-Winters for category series
        hw = HoltWintersForecaster(seasonal_periods=12, trend="add", seasonal="add").fit(ts)
        preds = hw.predict(steps)
        future_dates = pd.date_range(ts.index[-1] + pd.DateOffset(months=1), periods=steps, freq="MS")

        for d, p in zip(future_dates, preds):
            all_cat_forecasts.append({
                "Date": d,
                "Category": cat,
                "Forecast_Sales": max(0.0, float(p))
            })

    return pd.DataFrame(all_cat_forecasts)


def main():
    print("=" * 75)
    print(" SALES & DEMAND FORECASTING SYSTEM: END-TO-END EXECUTION ")
    print("=" * 75)

    # 1. Clean & Prepare Data
    print("\n[Step 1/8] Cleaning and validating raw Superstore transaction records...")
    df_raw = load_and_clean_superstore_data(
        raw_path="data/superstore_raw.csv",
        cleaned_path="data/superstore_cleaned.csv"
    )

    # 2. Aggregations
    print("\n[Step 2/8] Generating continuous monthly and weekly time-series aggregations...")
    monthly_sales = aggregate_sales_data(df_raw, freq="MS")
    weekly_sales = aggregate_sales_data(df_raw, freq="W-SUN")
    monthly_sales.to_csv("data/sales_aggregated_monthly.csv", index=False)
    weekly_sales.to_csv("data/sales_aggregated_weekly.csv", index=False)
    print(f" -> Monthly timeline: {len(monthly_sales)} months ({monthly_sales['Date'].min():%Y-%m} to {monthly_sales['Date'].max():%Y-%m})")
    print(f" -> Total Historical Sales: ${df_raw['Sales'].sum():,.2f}")

    # 3. Model Comparison on Holdout Test Set (2018)
    print("\n[Step 3/8] Running benchmark & multi-model evaluation on 2018 holdout test period...")
    metrics_df, pred_df, trained_models = run_model_comparison("data/sales_aggregated_monthly.csv")
    metrics_df.to_csv("data/model_evaluation_metrics.csv")
    pred_df.to_csv("data/holdout_validation_predictions.csv", index=False)
    print("\nModel Evaluation Leaderboard:")
    print(metrics_df.to_string())

    # 4. Out-of-Sample Forward Forecast (2019)
    print("\n[Step 4/8] Retraining champion ensemble on full dataset and forecasting 2019...")
    future_df = generate_future_forecast("data/sales_aggregated_monthly.csv", steps=12)
    future_df.to_csv("data/future_sales_forecast_2019.csv", index=False)
    print(f" -> Total 2019 Projected Sales: ${future_df['Final_Predicted_Sales'].sum():,.2f}")

    # 5. Category Breakdown Forecasting
    print("\n[Step 5/8] Generating category-level demand forecasts (Technology, Furniture, Office Supplies)...")
    cat_forecast_df = forecast_by_category(df_raw, steps=12)
    cat_forecast_df.to_csv("data/category_forecast_2019.csv", index=False)

    # 6. Business Planning & Inventory Calculations
    print("\n[Step 6/8] Deriving inventory buffers, safety stock, reorder points, and cash flow needs...")
    forecast_series = future_df.set_index("Date")["Final_Predicted_Sales"]
    best_rmse = metrics_df.loc["Hybrid Ensemble (SARIMAX + GB)", "RMSE"]
    inventory_plan = calculate_inventory_rules(
        forecast_series,
        actual_profit_ratio=df_raw["Profit"].sum() / df_raw["Sales"].sum(),
        service_level_z=1.65,
        lead_time_months=0.5,
        forecast_rmse=best_rmse
    )
    staffing_plan = calculate_staffing_and_capacity(inventory_plan)
    staffing_plan.to_csv("data/inventory_procurement_plan_2019.csv", index=False)

    # 7. Generate Visualizations
    print("\n[Step 7/8] Rendering publication-quality charts for executive presentations...")
    os.makedirs("visualizations", exist_ok=True)
    plot_sales_trends_and_seasonality(monthly_sales, df_raw)
    plot_seasonal_decomposition(monthly_sales)
    plot_model_comparison(metrics_df)
    plot_test_validation_actual_vs_predicted(pred_df)
    plot_future_forecast_with_ci(monthly_sales, future_df)
    plot_category_forecast(cat_forecast_df)
    plot_inventory_and_cashflow_plan(inventory_plan)

    # 8. Executive Briefing
    print("\n[Step 8/8] Generating executive summary...")
    peak_row = future_df.loc[future_df["Final_Predicted_Sales"].idxmax()]
    trough_row = future_df.loc[future_df["Final_Predicted_Sales"].idxmin()]
    summary_text = generate_executive_business_summary(
        metrics=metrics_df.loc["Hybrid Ensemble (SARIMAX + GB)"].to_dict(),
        total_historical_sales=df_raw["Sales"].sum(),
        total_projected_sales=future_df["Final_Predicted_Sales"].sum(),
        peak_month=f"{peak_row['Date']:%B %Y}",
        peak_sales=peak_row["Final_Predicted_Sales"],
        trough_month=f"{trough_row['Date']:%B %Y}",
        trough_sales=trough_row["Final_Predicted_Sales"]
    )
    print(summary_text)

    # Save summary report text
    with open("reports/EXECUTIVE_SUMMARY_BRIEFING.txt", "w") as f:
        f.write(summary_text)

    print("\n Pipeline execution completed successfully! All artifacts generated.")


if __name__ == "__main__":
    main()
