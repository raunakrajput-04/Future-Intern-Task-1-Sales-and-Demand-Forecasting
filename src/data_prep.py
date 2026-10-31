"""
Data Preparation & Cleaning Module for Sales & Demand Forecasting.
Handles loading, cleaning, validation, and multi-granularity time aggregations.
"""

import os
import pandas as pd
import numpy as np


def load_and_clean_superstore_data(
    raw_path: str = "data/superstore_raw.csv",
    cleaned_path: str = "data/superstore_cleaned.csv"
) -> pd.DataFrame:
    """
    Loads raw Superstore data, removes metadata/corrupted rows,
    parses dates, handles missing values, and saves cleaned dataset.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")

    df = pd.read_csv(raw_path, encoding="utf-8")

    # 1. Drop rows with null Sales or Order Date (e.g. appended metadata/people table)
    df = df.dropna(subset=["Order ID", "Sales", "Order Date"]).copy()

    # 2. Parse date columns
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])

    # 3. Handle data types
    numeric_cols = ["Sales", "Quantity", "Discount", "Profit"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Fill missing Postal Code if any with 'Unknown' or mode
    if "Postal Code" in df.columns:
        df["Postal Code"] = df["Postal Code"].fillna("00000").astype(str).str.replace(".0", "", regex=False)

    # 5. Feature: Shipping Duration (days)
    df["Ship_Duration_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

    # 6. Sort chronologically
    df = df.sort_values("Order Date").reset_index(drop=True)

    # 7. Save cleaned data
    if cleaned_path:
        os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
        df.to_csv(cleaned_path, index=False)
        print(f"[Data Prep] Cleaned data saved: {cleaned_path} ({len(df):,} valid orders/items)")

    return df


def aggregate_sales_data(
    df: pd.DataFrame,
    freq: str = "ME",  # 'ME' for Month End, 'W-SUN' for Weekly
    group_col: str = None
) -> pd.DataFrame:
    """
    Aggregates transactional sales data into a continuous time-series.
    Parameters:
        df: Cleaned dataframe with 'Order Date' and sales metrics.
        freq: Pandas frequency alias ('ME' or 'MS' for monthly, 'W-SUN' for weekly).
        group_col: Optional column to group by (e.g. 'Category' or 'Region').
    Returns:
        DataFrame indexed by date or with Date column, guaranteed continuous time-steps.
    """
    df = df.copy()
    df["Date"] = df["Order Date"].dt.to_period(
        "M" if "M" in freq else ("W" if "W" in freq else "D")
    ).dt.to_timestamp()

    agg_dict = {
        "Sales": "sum",
        "Quantity": "sum",
        "Profit": "sum",
        "Discount": "mean",
        "Order ID": "nunique"
    }

    if group_col:
        agg = df.groupby(["Date", group_col]).agg(agg_dict).reset_index()
        agg.rename(columns={"Order ID": "Order_Count", "Discount": "Avg_Discount"}, inplace=True)
        # Ensure pivot/continuous series for each group
        groups = agg[group_col].unique()
        all_dates = pd.date_range(start=agg["Date"].min(), end=agg["Date"].max(), freq=freq)
        idx = pd.MultiIndex.from_product([all_dates, groups], names=["Date", group_col])
        agg = agg.set_index(["Date", group_col]).reindex(idx, fill_value=0.0).reset_index()
    else:
        agg = df.groupby("Date").agg(agg_dict).reset_index()
        agg.rename(columns={"Order ID": "Order_Count", "Discount": "Avg_Discount"}, inplace=True)
        # Ensure continuity
        all_dates = pd.date_range(start=agg["Date"].min(), end=agg["Date"].max(), freq=freq)
        agg = agg.set_index("Date").reindex(all_dates, fill_value=0.0).rename_axis("Date").reset_index()

    return agg


if __name__ == "__main__":
    df = load_and_clean_superstore_data()
    monthly = aggregate_sales_data(df, freq="MS")
    weekly = aggregate_sales_data(df, freq="W-SUN")
    monthly.to_csv("data/sales_aggregated_monthly.csv", index=False)
    weekly.to_csv("data/sales_aggregated_weekly.csv", index=False)
    print(f"[Data Prep] Aggregated Monthly shape: {monthly.shape}")
    print(f"[Data Prep] Aggregated Weekly shape: {weekly.shape}")
    print(monthly.head())
