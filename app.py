"""
Interactive Sales & Demand Forecasting Dashboard for Business Stakeholders.
Designed for Store Owners, Startup Founders, and Operations Managers.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Retail Sales & Demand Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load precomputed datasets
@st.cache_data
def load_all_data():
    monthly = pd.read_csv("data/sales_aggregated_monthly.csv")
    monthly["Date"] = pd.to_datetime(monthly["Date"])

    metrics = pd.read_csv("data/model_evaluation_metrics.csv", index_col=0)
    validation = pd.read_csv("data/holdout_validation_predictions.csv")
    validation["Date"] = pd.to_datetime(validation["Date"])

    future = pd.read_csv("data/future_sales_forecast_2019.csv")
    future["Date"] = pd.to_datetime(future["Date"])

    cat_forecast = pd.read_csv("data/category_forecast_2019.csv")
    cat_forecast["Date"] = pd.to_datetime(cat_forecast["Date"])

    inventory = pd.read_csv("data/inventory_procurement_plan_2019.csv")
    inventory["Date"] = pd.to_datetime(inventory["Date"])

    raw = pd.read_csv("data/superstore_cleaned.csv")
    raw["Order Date"] = pd.to_datetime(raw["Order Date"])

    return monthly, metrics, validation, future, cat_forecast, inventory, raw

monthly_df, metrics_df, validation_df, future_df, cat_forecast_df, inventory_df, raw_df = load_all_data()

# ----------------- SIDEBAR -----------------
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=70)
st.sidebar.title("Business Planning Controls")

st.sidebar.markdown("### 🎯 Forecast Horizon")
horizon = st.sidebar.slider("Forecast Period (Months)", min_value=3, max_value=12, value=12, step=1)

st.sidebar.markdown("### 🧪 What-If Scenario Planning")
demand_shock = st.sidebar.slider("Marketing Uplift / Economic Shock (%)", min_value=-30, max_value=40, value=0, step=5)
lead_time_weeks = st.sidebar.slider("Supplier Lead Time (Weeks)", min_value=1, max_value=8, value=2, step=1)
service_level = st.sidebar.selectbox("Customer Service Level Target", options=["90% (Z=1.28)", "95% (Z=1.65)", "99% (Z=2.33)"], index=1)

z_map = {"90% (Z=1.28)": 1.28, "95% (Z=1.65)": 1.65, "99% (Z=2.33)": 2.33}
z_score = z_map[service_level]

st.sidebar.markdown("---")
st.sidebar.info("""
**Audience Guide**:
- **Store Owner**: Review safety stock and reorder points to avoid stockouts.
- **Startup Founder**: Check revenue growth, cash burn, and seasonality.
- **Operations Manager**: Balance staffing capacity with predicted Q4 surge.
""")

# Slice future forecast by selected horizon
active_future = future_df.iloc[:horizon].copy()
multiplier = 1.0 + (demand_shock / 100.0)
active_future["Scenario_Sales"] = active_future["Final_Predicted_Sales"] * multiplier
active_future["Scenario_Lower"] = np.maximum(0.0, active_future["Lower_95_CI"] * multiplier)
active_future["Scenario_Upper"] = active_future["Upper_95_CI"] * multiplier

# Sizing KPIs
tot_proj_sales = active_future["Scenario_Sales"].sum()
tot_hist_annual = monthly_df.tail(12)["Sales"].sum()
growth_pct = ((tot_proj_sales - (tot_hist_annual * horizon / 12)) / (tot_hist_annual * horizon / 12)) * 100.0
model_accuracy = metrics_df.loc["Hybrid Ensemble (SARIMAX + GB)", "Accuracy (%)"]
best_rmse = metrics_df.loc["Hybrid Ensemble (SARIMAX + GB)", "RMSE"]
lead_time_months = lead_time_weeks / 4.33
recom_safety_stock = z_score * np.sqrt(lead_time_months) * best_rmse
peak_row = active_future.loc[active_future["Scenario_Sales"].idxmax()]

# ----------------- MAIN HEADER -----------------
st.title("📊 Retail Sales & Demand Forecasting Intelligence")
st.markdown("##### Machine Learning-Powered Decision Support for Inventory, Cash Flow & Operations")

# Top KPI row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(f"Projected Revenue ({horizon}M)", f"${tot_proj_sales:,.0f}", f"{growth_pct:+.1f}% vs Last Year")
kpi2.metric("Forecast Reliability (Accuracy)", f"{model_accuracy:.1f}%", f"WAPE: {100 - model_accuracy:.1f}%")
kpi3.metric("Required Safety Stock Buffer", f"${recom_safety_stock:,.0f}", f"For {service_level.split()[0]} SLA")
kpi4.metric("Peak Demand Month", f"{peak_row['Date']:%b %Y}", f"${peak_row['Scenario_Sales']:,.0f}")

st.markdown("---")

# ----------------- TABS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Revenue Forecast & Horizon",
    "📦 Inventory & Staffing Strategy",
    "🔍 Trend & Seasonal Drivers",
    "🏆 Model Leaderboard & Validation",
    "📋 Data & Scenario Exports"
])

# TAB 1: Main Forecast
with tab1:
    st.subheader(f"12-Month Out-of-Sample Demand Projection ({horizon}-Month Selected Window)")
    
    fig_main = go.Figure()

    # Historical
    fig_main.add_trace(go.Scatter(
        x=monthly_df["Date"],
        y=monthly_df["Sales"],
        name="Historical Actual Sales (2015-2018)",
        line=dict(color="#2980b9", width=2.5),
        mode="lines+markers"
    ))

    # Upper 95% Confidence Band
    fig_main.add_trace(go.Scatter(
        x=active_future["Date"],
        y=active_future["Scenario_Upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="Upper 95% CI"
    ))

    # Lower 95% Confidence Band with Fill
    fig_main.add_trace(go.Scatter(
        x=active_future["Date"],
        y=active_future["Scenario_Lower"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(243, 156, 18, 0.2)",
        name="95% Confidence Interval"
    ))

    # Projected Mean
    fig_main.add_trace(go.Scatter(
        x=active_future["Date"],
        y=active_future["Scenario_Sales"],
        name=f"Projected Demand (Scenario: {demand_shock:+d}%)",
        line=dict(color="#e67e22", width=3.5, dash="solid"),
        mode="lines+markers"
    ))

    fig_main.update_layout(
        title="<b>Enterprise Sales Trajectory with Predictive Uncertainty Fans</b>",
        xaxis_title="Date",
        yaxis_title="Monthly Revenue ($)",
        hovermode="x unified",
        template="plotly_white",
        height=520,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_main, use_container_width=True)

    # Category Breakdown
    st.subheader("Category-Level Forward Projections")
    fig_cat = px.line(
        cat_forecast_df[cat_forecast_df["Date"].isin(active_future["Date"])],
        x="Date",
        y="Forecast_Sales",
        color="Category",
        markers=True,
        title="<b>Product Category Demand Trajectory (Technology, Furniture, Office Supplies)</b>",
        color_discrete_map={"Furniture": "#e67e22", "Office Supplies": "#3498db", "Technology": "#2ecc71"}
    )
    fig_cat.update_layout(template="plotly_white", height=400, yaxis_title="Projected Sales ($)")
    st.plotly_chart(fig_cat, use_container_width=True)

# TAB 2: Inventory & Operations Planning
with tab2:
    st.subheader("Supply Chain & Operational Capacity Blueprint")
    st.markdown("""
    Use these metrics to prevent the two biggest retail killers: **Lost Revenue from Stockouts** and **Capital Drain from Overstocking**.
    """)

    col_inv1, col_inv2 = st.columns([3, 2])
    with col_inv1:
        # Dynamic calculation of inventory table based on what-if sliders
        dyn_inv = []
        cogs_ratio = 1.0 - (raw_df["Profit"].sum() / raw_df["Sales"].sum())
        for _, row in active_future.iterrows():
            pred_s = row["Scenario_Sales"]
            ltd = pred_s * lead_time_months
            rop = ltd + recom_safety_stock
            cogs = pred_s * cogs_ratio
            dyn_inv.append({
                "Date": row["Date"].strftime("%b %Y"),
                "Projected Sales": f"${pred_s:,.0f}",
                "Procurement COGS": f"${cogs:,.0f}",
                "Safety Buffer": f"${recom_safety_stock:,.0f}",
                "Reorder Point (ROP)": f"${rop:,.0f}",
                "Est. Monthly Holding Cost": f"${(recom_safety_stock * 0.20 / 12):,.0f}"
            })
        st.dataframe(pd.DataFrame(dyn_inv), use_container_width=True)

    with col_inv2:
        st.markdown("#### 🛡️ Inventory Sizing Formulae")
        st.latex(r"SS = Z \times \sqrt{L} \times \sigma_{\text{error}}")
        st.latex(r"ROP = (\text{Demand during Lead Time}) + SS")
        st.info(f"""
        - **Lead Time**: {lead_time_weeks} weeks ({lead_time_months:.2f} months)
        - **Model RMSE**: ${best_rmse:,.0f}
        - **Z-Multiplier**: {z_score} ({service_level})
        - **Calculated Safety Stock**: **${recom_safety_stock:,.0f}**
        """)

    st.markdown("---")
    st.subheader("Warehouse & Store Staffing Demand Index")
    mean_forecast = active_future["Scenario_Sales"].mean()
    active_future["Staffing_Index"] = (active_future["Scenario_Sales"] / mean_forecast).round(2)
    fig_staff = px.bar(
        active_future,
        x=active_future["Date"].dt.strftime("%b %Y"),
        y="Staffing_Index",
        color="Staffing_Index",
        color_continuous_scale="Blues",
        title="<b>Monthly Staffing Index (1.0 = Baseline Shift Capacity)</b>"
    )
    fig_staff.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Standard Capacity")
    fig_staff.add_hline(y=1.25, line_dash="dot", line_color="red", annotation_text="Overtime / Temp Staff Trigger")
    fig_staff.update_layout(template="plotly_white", height=380, yaxis_title="Staffing Index Multiplier")
    st.plotly_chart(fig_staff, use_container_width=True)

# TAB 3: Trend & Seasonality
with tab3:
    st.subheader("Underlying Drivers: Trend, Seasonality & Cyclical Swings")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.image("visualizations/01_sales_trend_and_seasonality.png", caption="Historical Trends & Monthly Distributions", use_container_width=True)
    with col_t2:
        st.image("visualizations/02_seasonal_decomposition.png", caption="Classical Time-Series Additive Decomposition", use_container_width=True)

# TAB 4: Model Leaderboard & Validation
with tab4:
    st.subheader("Model Performance Benchmark (2018 Holdout Validation)")
    st.markdown("All models were trained strictly on 2015-2017 historical data and tested on the unseen 12 months of 2018.")
    
    st.dataframe(metrics_df.style.format({
        "MAE": "${:,.1f}",
        "RMSE": "${:,.1f}",
        "WAPE (%)": "{:.2f}%",
        "MAPE (%)": "{:.2f}%",
        "R2 Score": "{:.3f}",
        "Mean Bias Error": "${:,.1f}",
        "Accuracy (%)": "{:.1f}%"
    }).highlight_min(subset=["WAPE (%)", "MAE", "RMSE"], color="#d4efdf")
      .highlight_max(subset=["R2 Score", "Accuracy (%)"], color="#d4efdf"),
      use_container_width=True)

    col_val1, col_val2 = st.columns(2)
    with col_val1:
        st.image("visualizations/03_model_performance_comparison.png", caption="Error Metric Comparison", use_container_width=True)
    with col_val2:
        st.image("visualizations/04_actual_vs_predicted_validation.png", caption="Holdout Test Predictions vs Ground Truth Actuals", use_container_width=True)

# TAB 5: Data Explorer & Exports
with tab5:
    st.subheader("Download Production Forecasts & Business Datasets")
    st.dataframe(active_future, use_container_width=True)

    csv_data = active_future.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download 2019 Forecast Plan (CSV)",
        data=csv_data,
        file_name=f"sales_forecast_horizon_{horizon}M.csv",
        mime="text/csv"
    )

    st.markdown("### Executive Summary Text Briefing")
    with open("reports/EXECUTIVE_SUMMARY_BRIEFING.txt", "r") as f:
        st.text(f.read())
