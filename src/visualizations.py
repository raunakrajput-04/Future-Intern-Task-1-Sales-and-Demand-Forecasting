"""
Visualizations Module for Sales & Demand Forecasting.
Generates publication-quality charts for executive presentations and business stakeholders.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# Configure aesthetics for modern, clean corporate presentations
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_sales_trends_and_seasonality(
    df_monthly: pd.DataFrame,
    df_raw: pd.DataFrame,
    save_path: str = "visualizations/01_sales_trend_and_seasonality.png"
):
    """
    Plots historical sales trend, moving averages, and month-by-month seasonality distribution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Total Monthly Sales Trend + 3M Moving Average
    df = df_monthly.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["MA3"] = df["Sales"].rolling(window=3).mean()

    axes[0, 0].plot(df["Date"], df["Sales"] / 1000, marker="o", color="#1f77b4", label="Monthly Sales", linewidth=2)
    axes[0, 0].plot(df["Date"], df["MA3"] / 1000, color="#ff7f0e", linestyle="--", linewidth=2.5, label="3-Month Moving Average")
    axes[0, 0].set_title("Historical Monthly Sales Trend (2015 - 2018)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 0].set_ylabel("Sales ($ in Thousands)", fontsize=11)
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0, 0].legend(loc="upper left", frameon=True)

    # 2. Monthly Seasonality Bar (Average Sales by Month)
    df["Month_Name"] = df["Date"].dt.strftime("%b")
    df["Month_Num"] = df["Date"].dt.month
    monthly_avg = df.groupby(["Month_Num", "Month_Name"])["Sales"].mean().reset_index()
    monthly_avg = monthly_avg.sort_values("Month_Num")

    colors = ["#4a90e2" if m not in [11, 12] else "#e74c3c" for m in monthly_avg["Month_Num"]]
    bars = axes[0, 1].bar(monthly_avg["Month_Name"], monthly_avg["Sales"] / 1000, color=colors, edgecolor="black", alpha=0.85)
    axes[0, 1].set_title("Seasonal Profile: Average Sales by Month (Q4 Surge Highlighted)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 1].set_ylabel("Average Sales ($ in Thousands)", fontsize=11)
    for bar in bars:
        yval = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, yval + 1, f"${yval:.0f}k", ha="center", va="bottom", fontsize=8.5)

    # 3. Category Sales Over Time
    cat_df = df_raw.copy()
    cat_df["Order Date"] = pd.to_datetime(cat_df["Order Date"])
    cat_monthly = cat_df.groupby([cat_df["Order Date"].dt.to_period("M").dt.to_timestamp(), "Category"])["Sales"].sum().unstack()
    cat_monthly.plot(ax=axes[1, 0], colormap="viridis", linewidth=2)
    axes[1, 0].set_title("Sales by Product Category Over Time", fontsize=13, fontweight="bold", pad=10)
    axes[1, 0].set_ylabel("Sales ($)", fontsize=11)
    axes[1, 0].legend(title="Category", frameon=True)

    # 4. Profitability vs Sales by Region
    region_summary = df_raw.groupby("Region").agg({"Sales": "sum", "Profit": "sum"}).reset_index()
    x = np.arange(len(region_summary))
    width = 0.35
    axes[1, 1].bar(x - width/2, region_summary["Sales"] / 1000, width, label="Sales ($k)", color="#3498db")
    axes[1, 1].bar(x + width/2, region_summary["Profit"] / 1000, width, label="Profit ($k)", color="#2ecc71")
    axes[1, 1].set_title("Regional Revenue & Profitability Distribution", fontsize=13, fontweight="bold", pad=10)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(region_summary["Region"], fontsize=10)
    axes[1, 1].set_ylabel("Amount ($ in Thousands)", fontsize=11)
    axes[1, 1].legend(frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_seasonal_decomposition(
    df_monthly: pd.DataFrame,
    save_path: str = "visualizations/02_seasonal_decomposition.png"
):
    """
    Decomposes monthly time series into Trend, Seasonal, and Residual components.
    """
    df = df_monthly.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    ts = df.set_index("Date")["Sales"]

    decomp = seasonal_decompose(ts, model="additive", period=12)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(decomp.observed, color="#2c3e50", linewidth=2)
    axes[0].set_title("Observed Historical Sales", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Sales ($)")

    axes[1].plot(decomp.trend, color="#2980b9", linewidth=2.5)
    axes[1].set_title("Underlying Trend (Steady Long-Term Business Growth)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Trend ($)")

    axes[2].plot(decomp.seasonal, color="#27ae60", linewidth=2)
    axes[2].set_title("Seasonal Pattern (Annual Repeating Cycle: Q1 Dip & Q4 Spike)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Seasonality ($)")

    axes[3].scatter(decomp.resid.index, decomp.resid, color="#e74c3c", alpha=0.7)
    axes[3].axhline(0, color="black", linestyle="--", alpha=0.6)
    axes[3].set_title("Irregular Residuals (Unsystematic Noise / Random Shocks)", fontsize=12, fontweight="bold")
    axes[3].set_ylabel("Residual ($)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    save_path: str = "visualizations/03_model_performance_comparison.png"
):
    """
    Plots horizontal bar charts of model error metrics (WAPE %, MAE $, R2).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    df = metrics_df.sort_values("WAPE (%)", ascending=False).copy()

    # 1. WAPE (%)
    colors = ["#2ecc71" if i == len(df)-1 else ("#3498db" if i >= len(df)-3 else "#95a5a6") for i in range(len(df))]
    axes[0].barh(df.index, df["WAPE (%)"], color=colors)
    axes[0].set_title("Weighted Absolute % Error (WAPE) - Lower is Better", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("WAPE (%)")
    for i, v in enumerate(df["WAPE (%)"]):
        axes[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9.5, fontweight="bold")

    # 2. MAE ($)
    axes[1].barh(df.index, df["MAE"] / 1000, color="#f39c12")
    axes[1].set_title("Mean Absolute Error (MAE) - Lower is Better", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("MAE ($ in Thousands)")
    for i, v in enumerate(df["MAE"] / 1000):
        axes[1].text(v + 0.2, i, f"${v:.1f}k", va="center", fontsize=9.5)

    # 3. R2 Score
    axes[2].barh(df.index, df["R2 Score"], color="#9b59b6")
    axes[2].set_title("R² Goodness of Fit - Higher is Better", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("R² Score")
    for i, v in enumerate(df["R2 Score"]):
        axes[2].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_test_validation_actual_vs_predicted(
    pred_df: pd.DataFrame,
    save_path: str = "visualizations/04_actual_vs_predicted_validation.png"
):
    """
    Visualizes holdout test validation: Actual vs Top Forecasters for 2018.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [2.5, 1]}, sharex=True)

    dates = pd.to_datetime(pred_df["Date"])
    actual = pred_df["Actual Sales"] / 1000

    ax1.plot(dates, actual, marker="o", color="#2c3e50", linewidth=3, label="Actual Ground Truth (2018)", zorder=5)
    ax1.plot(dates, pred_df["Hybrid Ensemble (SARIMAX + GB)"] / 1000, marker="s", color="#27ae60", linewidth=2.5, linestyle="--", label="Hybrid Ensemble (Champion)")
    ax1.plot(dates, pred_df["Holt-Winters"] / 1000, marker="^", color="#2980b9", linewidth=2, linestyle=":", label="Holt-Winters")
    ax1.plot(dates, pred_df["Gradient Boosting"] / 1000, marker="d", color="#e67e22", linewidth=2, linestyle="-.", label="Gradient Boosting")
    ax1.plot(dates, pred_df["Seasonal Naive Baseline"] / 1000, color="#95a5a6", linewidth=1.8, linestyle=":", label="Seasonal Naive Benchmark")

    ax1.set_title("Out-of-Time Model Validation: Actual vs. Predictions (2018 Holdout)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Monthly Sales ($ in Thousands)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)

    # Residuals / Errors
    ens_resid = (pred_df["Actual Sales"] - pred_df["Hybrid Ensemble (SARIMAX + GB)"]) / 1000
    naive_resid = (pred_df["Actual Sales"] - pred_df["Seasonal Naive Baseline"]) / 1000

    ax2.bar(dates - pd.Timedelta(days=5), ens_resid, width=8, label="Hybrid Ensemble Error ($k)", color="#27ae60", alpha=0.8)
    ax2.bar(dates + pd.Timedelta(days=5), naive_resid, width=8, label="Naive Baseline Error ($k)", color="#e74c3c", alpha=0.5)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_title("Forecast Error Comparison (Actual - Forecast)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Error ($k)", fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_future_forecast_with_ci(
    df_monthly: pd.DataFrame,
    future_df: pd.DataFrame,
    save_path: str = "visualizations/05_future_sales_forecast_with_ci.png"
):
    """
    Generates the core executive presentation forecast chart: 4-year history + 12-month forward forecast with confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(16, 7.5))

    hist_dates = pd.to_datetime(df_monthly["Date"])
    hist_sales = df_monthly["Sales"] / 1000

    fut_dates = pd.to_datetime(future_df["Date"])
    fut_sales = future_df["Final_Predicted_Sales"] / 1000
    lower_95 = future_df["Lower_95_CI"] / 1000
    upper_95 = future_df["Upper_95_CI"] / 1000
    lower_80 = future_df["Lower_80_CI"] / 1000
    upper_80 = future_df["Upper_80_CI"] / 1000

    # Historical line
    ax.plot(hist_dates, hist_sales, color="#1f4e78", linewidth=2.5, label="Historical Actual Sales (2015-2018)")
    ax.scatter(hist_dates, hist_sales, color="#1f4e78", s=30)

    # Forecast line
    ax.plot(fut_dates, fut_sales, color="#e67e22", linewidth=3, marker="o", label="12-Month Projected Sales (2019)")

    # Connect last historical point to first forecast point
    conn_dates = [hist_dates.iloc[-1], fut_dates.iloc[0]]
    conn_sales = [hist_sales.iloc[-1], fut_sales.iloc[0]]
    ax.plot(conn_dates, conn_sales, color="#e67e22", linestyle=":", linewidth=2)

    # Uncertainty Fans
    ax.fill_between(fut_dates, lower_80, upper_80, color="#f39c12", alpha=0.25, label="80% Confidence Interval")
    ax.fill_between(fut_dates, lower_95, upper_95, color="#f39c12", alpha=0.12, label="95% Confidence Interval")

    # Vertical demarcation line
    ax.axvline(hist_dates.iloc[-1], color="#7f8c8d", linestyle="--", linewidth=1.5)
    ax.text(hist_dates.iloc[-1] + pd.Timedelta(days=10), ax.get_ylim()[1]*0.9, "Forecast Horizon (2019)  -->", fontsize=11, fontweight="bold", color="#d35400")

    # Annotate peak and trough
    peak_idx = fut_sales.argmax()
    peak_date = fut_dates.iloc[peak_idx]
    peak_val = fut_sales.iloc[peak_idx]
    ax.annotate(f"Peak Demand: Nov 2019\n${peak_val:.1f}k Projected",
                xy=(peak_date, peak_val),
                xytext=(peak_date - pd.Timedelta(days=70), peak_val + 15),
                arrowprops=dict(facecolor="#c0392b", shrink=0.08, width=1.5, headwidth=7),
                fontsize=9.5, fontweight="bold", color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#fadbd8", edgecolor="#e74c3c"))

    trough_idx = fut_sales.argmin()
    trough_date = fut_dates.iloc[trough_idx]
    trough_val = fut_sales.iloc[trough_idx]
    ax.annotate(f"Trough: Feb 2019\n${trough_val:.1f}k Projected",
                xy=(trough_date, trough_val),
                xytext=(trough_date - pd.Timedelta(days=60), trough_val - 22),
                arrowprops=dict(facecolor="#2980b9", shrink=0.08, width=1.5, headwidth=7),
                fontsize=9.5, fontweight="bold", color="#2980b9",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#d4e6f1", edgecolor="#3498db"))

    ax.set_title("Executive Sales Forecast: Next 12 Months with 95% Confidence Intervals", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Sales ($ in Thousands)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper left", frameon=True, fontsize=10.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_category_forecast(
    cat_forecast_df: pd.DataFrame,
    save_path: str = "visualizations/06_category_demand_forecast.png"
):
    """
    Visualizes forward demand breakdown across Product Categories (Technology, Furniture, Office Supplies).
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    categories = cat_forecast_df["Category"].unique()
    palette = {"Furniture": "#e67e22", "Office Supplies": "#3498db", "Technology": "#2ecc71"}

    for cat in categories:
        sub = cat_forecast_df[cat_forecast_df["Category"] == cat].sort_values("Date")
        dates = pd.to_datetime(sub["Date"])
        ax.plot(dates, sub["Forecast_Sales"] / 1000, marker="o", linewidth=2.5, label=f"{cat} Forecast", color=palette.get(cat, None))

    ax.set_title("Category-Level Forward Demand Forecast (2019)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Forecast Sales ($ in Thousands)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(title="Product Category", frameon=True, fontsize=10.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_inventory_and_cashflow_plan(
    inventory_df: pd.DataFrame,
    save_path: str = "visualizations/07_business_inventory_cashflow_plan.png"
):
    """
    Plots business inventory thresholds, safety stock, and projected monthly COGS.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    dates = pd.to_datetime(inventory_df["Date"])

    # 1. Sales vs Inventory Reorder Point
    ax1.plot(dates, inventory_df["Predicted_Sales"] / 1000, marker="o", color="#2980b9", linewidth=2.5, label="Projected Sales ($k)")
    ax1.plot(dates, inventory_df["Reorder_Point ($)"] / 1000, marker="s", color="#e67e22", linewidth=2, linestyle="--", label="Recommended Reorder Point (ROP $k)")
    ax1.fill_between(dates, 0, inventory_df["Safety_Stock_Buffer ($)"] / 1000, color="#2ecc71", alpha=0.2, label="Required Safety Stock Buffer ($k)")
    ax1.set_title("Supply Chain Operations Plan: Forecasted Sales vs. Reorder Thresholds", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Capital Amount ($k)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)

    # 2. Working Capital & Cash Outflow (Expected COGS)
    ax2.bar(dates, inventory_df["Expected_COGS"] / 1000, width=18, color="#34495e", alpha=0.8, label="Projected COGS Cash Outflow ($k)")
    ax2.set_title("Monthly Working Capital Requirements (Procurement Cash Outflow)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("Procurement Outflow ($k)", fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")
