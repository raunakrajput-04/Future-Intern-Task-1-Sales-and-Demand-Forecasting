"""
Forecasting Models Module.
Implements Baseline Naive, SARIMAX, Holt-Winters, and Machine Learning Regressors
with multi-step recursive forecasting and uncertainty intervals.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.feature_engineering import create_time_features
from src.evaluation import calculate_metrics, format_metrics_table

warnings.filterwarnings("ignore")


class SeasonalNaiveForecaster:
    """Baseline benchmark: repeats the sales from 12 months ago."""
    def __init__(self, season_length: int = 12):
        self.season_length = season_length
        self.history = None

    def fit(self, y: pd.Series):
        self.history = list(y.values)
        return self

    def predict(self, steps: int = 12) -> np.ndarray:
        preds = []
        for i in range(steps):
            # Take value from season_length steps ago in the evolving sequence
            val = self.history[-self.season_length + (i % self.season_length)]
            preds.append(val)
        return np.array(preds)


class SarimaxForecaster:
    """Classical Time-Series: Seasonal ARIMA model."""
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_res = None
        self.train_series = None

    def fit(self, y: pd.Series):
        self.train_series = y
        model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_res = model.fit(disp=False)
        return self

    def predict(self, steps: int = 12):
        forecast = self.model_res.get_forecast(steps=steps)
        pred_mean = forecast.predicted_mean.values
        conf_int = forecast.conf_int()
        lower = conf_int.iloc[:, 0].values
        upper = conf_int.iloc[:, 1].values
        return pred_mean, lower, upper


class HoltWintersForecaster:
    """Holt-Winters Exponential Smoothing model."""
    def __init__(self, seasonal_periods: int = 12, trend="add", seasonal="add"):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.model_res = None

    def fit(self, y: pd.Series):
        model = ExponentialSmoothing(
            y,
            seasonal_periods=self.seasonal_periods,
            trend=self.trend,
            seasonal=self.seasonal
        )
        self.model_res = model.fit()
        return self

    def predict(self, steps: int = 12) -> np.ndarray:
        return self.model_res.forecast(steps).values


class MLRecursiveForecaster:
    """
    Machine Learning Regressor with Recursive Multi-Step Feature Updating.
    Enables Ridge, Random Forest, and Gradient Boosting to generate forward forecasts.
    """
    def __init__(self, estimator, feature_cols: list):
        self.estimator = estimator
        self.feature_cols = feature_cols
        self.history_df = None
        self.residual_std = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.estimator.fit(X[self.feature_cols], y)
        in_sample_preds = self.estimator.predict(X[self.feature_cols])
        residuals = y.values - in_sample_preds
        self.residual_std = float(np.std(residuals))
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X_test[self.feature_cols])

    def forecast_future(self, last_known_df: pd.DataFrame, steps: int = 12, freq: str = "MS") -> pd.DataFrame:
        """
        Generates forward out-of-sample predictions iteratively updating lags.
        """
        working_df = last_known_df.copy()
        start_date = pd.to_datetime(working_df["Date"].iloc[-1])
        future_dates = pd.date_range(start_date + pd.DateOffset(months=1), periods=steps, freq=freq)

        future_records = []
        for next_date in future_dates:
            # Create a row with next_date
            temp_row = pd.DataFrame([{"Date": next_date, "Sales": np.nan}])
            working_df = pd.concat([working_df, temp_row], ignore_index=True)

            # Re-compute features on full historical + future series
            feat_df = create_time_features(working_df, drop_na=False)
            current_row_features = feat_df.iloc[[-1]][self.feature_cols].bfill(axis=1).fillna(0.0)

            # Predict next step
            pred_val = float(self.estimator.predict(current_row_features)[0])
            pred_val = max(0.0, pred_val)  # Sales cannot be negative

            # Update working_df with predicted sales so next step can lag it
            working_df.loc[working_df.index[-1], "Sales"] = pred_val

            future_records.append({
                "Date": next_date,
                "Predicted_Sales": pred_val,
                "Lower_95": max(0.0, pred_val - 1.96 * self.residual_std),
                "Upper_95": pred_val + 1.96 * self.residual_std,
                "Lower_80": max(0.0, pred_val - 1.28 * self.residual_std),
                "Upper_80": pred_val + 1.28 * self.residual_std
            })

        return pd.DataFrame(future_records)


def run_model_comparison(data_path: str = "data/sales_aggregated_monthly.csv") -> tuple:
    """
    Trains all models on 2015-2017 (36 months) and validates on 2018 (12 months).
    Returns metrics table, predictions dataframe, and trained model instances.
    """
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    ts = df.set_index("Date")["Sales"]

    # 36 months train (2015-2017), 12 months test (2018)
    split_idx = 36
    train_ts = ts.iloc[:split_idx]
    test_ts = ts.iloc[split_idx:]
    test_dates = test_ts.index

    # 1. Seasonal Naive
    naive = SeasonalNaiveForecaster(season_length=12).fit(train_ts)
    naive_pred = naive.predict(12)

    # 2. SARIMAX
    sarima = SarimaxForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)).fit(train_ts)
    sarima_pred, _, _ = sarima.predict(12)

    # 3. Holt-Winters
    hw = HoltWintersForecaster().fit(train_ts)
    hw_pred = hw.predict(12)

    # 4. Feature Matrix for ML models
    feat_df = create_time_features(df, drop_na=False)
    target_col = "Sales"
    ignore_cols = ["Date", "Sales", "Quantity", "Profit", "Avg_Discount", "Order_Count"]
    feature_cols = [c for c in feat_df.columns if c not in ignore_cols]

    # Clean NaNs in features
    X = feat_df[feature_cols].bfill().ffill()
    y = feat_df[target_col]

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 5. Ridge Regression
    ridge_reg = Ridge(alpha=10.0, random_state=42)
    ridge_forecaster = MLRecursiveForecaster(ridge_reg, feature_cols).fit(X_train, y_train)
    ridge_pred = ridge_forecaster.predict(X_test)

    # 6. Random Forest
    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_split=3, random_state=42)
    rf_forecaster = MLRecursiveForecaster(rf_reg, feature_cols).fit(X_train, y_train)
    rf_pred = rf_forecaster.predict(X_test)

    # 7. Gradient Boosting
    gb_reg = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.06, subsample=0.85, random_state=42)
    gb_forecaster = MLRecursiveForecaster(gb_reg, feature_cols).fit(X_train, y_train)
    gb_pred = gb_forecaster.predict(X_test)

    # Ensemble: Average of Top Statistical (SARIMAX) and ML (Gradient Boosting)
    ensemble_pred = (sarima_pred + gb_pred) / 2.0

    predictions_dict = {
        "Actual Sales": test_ts.values,
        "Seasonal Naive Baseline": naive_pred,
        "SARIMAX": sarima_pred,
        "Holt-Winters": hw_pred,
        "Ridge Regression": ridge_pred,
        "Random Forest": rf_pred,
        "Gradient Boosting": gb_pred,
        "Hybrid Ensemble (SARIMAX + GB)": ensemble_pred
    }

    results = {}
    for name, pred in predictions_dict.items():
        if name != "Actual Sales":
            results[name] = calculate_metrics(test_ts.values, pred)

    metrics_df = format_metrics_table(results)
    pred_df = pd.DataFrame(predictions_dict, index=test_dates).reset_index().rename(columns={"index": "Date"})

    return metrics_df, pred_df, {
        "feature_cols": feature_cols,
        "sarima": sarima,
        "gb_reg": gb_reg,
        "rf_reg": rf_reg,
        "ridge_reg": ridge_reg
    }


def generate_future_forecast(data_path: str = "data/sales_aggregated_monthly.csv", steps: int = 12) -> pd.DataFrame:
    """
    Fits models on the complete 4-year history (2015-2018) and forecasts future 12 months (2019).
    """
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    ts = df.set_index("Date")["Sales"]

    # SARIMAX on full series
    sarima_full = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_future = sarima_full.get_forecast(steps=steps)
    sarima_pred_mean = sarima_future.predicted_mean.values
    conf_int = sarima_future.conf_int()

    # Gradient Boosting on full series
    feat_df = create_time_features(df, drop_na=False)
    ignore_cols = ["Date", "Sales", "Quantity", "Profit", "Avg_Discount", "Order_Count"]
    feature_cols = [c for c in feat_df.columns if c not in ignore_cols]
    X_full = feat_df[feature_cols].bfill().ffill()
    y_full = feat_df["Sales"]

    gb = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.85, random_state=42)
    gb_forecaster = MLRecursiveForecaster(gb, feature_cols).fit(X_full, y_full)
    gb_future_df = gb_forecaster.forecast_future(df[["Date", "Sales"]], steps=steps)

    # Hybrid Forecast
    hybrid_pred = (sarima_pred_mean + gb_future_df["Predicted_Sales"].values) / 2.0
    future_dates = gb_future_df["Date"]

    combined_future = pd.DataFrame({
        "Date": future_dates,
        "Forecast_SARIMAX": sarima_pred_mean,
        "Forecast_GradientBoosting": gb_future_df["Predicted_Sales"].values,
        "Final_Predicted_Sales": hybrid_pred,
        "Lower_95_CI": np.maximum(0.0, conf_int.iloc[:, 0].values),
        "Upper_95_CI": conf_int.iloc[:, 1].values,
        "Lower_80_CI": np.maximum(0.0, hybrid_pred - 1.28 * np.std(sarima_full.resid)),
        "Upper_80_CI": hybrid_pred + 1.28 * np.std(sarima_full.resid)
    })

    return combined_future


if __name__ == "__main__":
    metrics_df, pred_df, _ = run_model_comparison()
    print("\n--- MODEL PERFORMANCE COMPARISON (TEST SET: 2018) ---")
    print(metrics_df.to_string())

    future_df = generate_future_forecast()
    print("\n--- 2019 OUT-OF-SAMPLE FUTURE FORECAST ---")
    print(future_df[["Date", "Final_Predicted_Sales", "Lower_95_CI", "Upper_95_CI"]].head(6))
