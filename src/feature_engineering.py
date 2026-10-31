"""
Feature Engineering Module for Time-Series Sales & Demand Forecasting.
Extracts calendar features, cyclical seasonal indicators, autoregressive lags,
rolling statistics, and macroeconomic/trend signals.
"""

import numpy as np
import pandas as pd


def create_time_features(
    df: pd.DataFrame,
    date_col: str = "Date",
    target_col: str = "Sales",
    freq: str = "M",
    lags: list = None,
    rolling_windows: list = None,
    drop_na: bool = True
) -> pd.DataFrame:
    """
    Generates rich time-based features for regression-based forecasting.
    Parameters:
        df: Input DataFrame containing date_col and target_col.
        date_col: Name of datetime column.
        target_col: Target variable (e.g. 'Sales' or 'Quantity').
        freq: Frequency ('M' for monthly, 'W' for weekly).
        lags: List of lag periods to create (default [1, 2, 3, 12] for monthly).
        rolling_windows: List of window sizes for rolling statistics.
        drop_na: Whether to drop initial rows with NaN from lags/rolling.
    Returns:
        DataFrame with engineered feature matrix.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    df = df.sort_values(date_col).reset_index(drop=True)

    # 1. Calendar Features
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["day_of_year"] = df[date_col].dt.dayofyear
    df["time_step"] = np.arange(len(df))  # Linear trend proxy

    if freq == "W":
        df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
        df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52.0)
        df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52.0)

    # 2. Cyclical Seasonality Encodings (sin/cos of month and quarter)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["sin_quarter"] = np.sin(2 * np.pi * df["quarter"] / 4.0)
    df["cos_quarter"] = np.cos(2 * np.pi * df["quarter"] / 4.0)

    # 3. Retail Holiday / Quarter-End Boost Flags
    # November & December are peak holiday shopping months (Black Friday, Cyber Monday, Christmas)
    df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)
    # End of Q1 (March) and Q4 (December) are typical corporate budget clearing periods
    df["is_quarter_end"] = df["month"].isin([3, 6, 9, 12]).astype(int)

    # 4. Autoregressive Lag Features
    if lags is None:
        lags = [1, 2, 3, 12] if freq == "M" else [1, 2, 4, 12]

    for lag in lags:
        if lag < len(df):
            df[f"lag_{lag}"] = df[target_col].shift(lag)

    # 5. Rolling Window Statistics (computed on lagged values to prevent future data leakage)
    if rolling_windows is None:
        rolling_windows = [3, 6] if freq == "M" else [4, 8, 12]

    # Use lag_1 as base for rolling calculations so current observation isn't leaked into features
    shifted_target = df[target_col].shift(1)
    for window in rolling_windows:
        if window < len(df):
            df[f"rolling_mean_{window}"] = shifted_target.rolling(window=window).mean()
            df[f"rolling_std_{window}"] = shifted_target.rolling(window=window).std()
            df[f"rolling_min_{window}"] = shifted_target.rolling(window=window).min()
            df[f"rolling_max_{window}"] = shifted_target.rolling(window=window).max()

    # 6. Momentum & Growth Rate
    if "lag_1" in df.columns and "lag_2" in df.columns:
        df["mom_momentum"] = (df["lag_1"] - df["lag_2"]) / (df["lag_2"] + 1e-5)

    if drop_na:
        df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    monthly_df = pd.read_csv("data/sales_aggregated_monthly.csv")
    feat_df = create_time_features(monthly_df, date_col="Date", target_col="Sales", freq="M")
    print(f"[Feature Engineering] Shape after feature creation: {feat_df.shape}")
    print("Engineered Columns:", feat_df.columns.tolist())
    print(feat_df.head(2))
