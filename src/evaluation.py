"""
Model Evaluation and Error Diagnostics Module for Business Forecasting.
Implements business-friendly error metrics: MAE, RMSE, MAPE, WAPE, R², and Bias.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes comprehensive forecasting accuracy and business error metrics.
    Parameters:
        y_true: Actual ground truth values.
        y_pred: Predicted values from model.
    Returns:
        Dictionary of formatted metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # 1. Standard Regression Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # 2. Percentage Metrics
    # Avoid zero-division in MAPE with small epsilon
    non_zero = y_true != 0
    if np.any(non_zero):
        mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0
    else:
        mape = np.nan

    # 3. Weighted Absolute Percentage Error (WAPE) - Industry Retail Gold Standard
    sum_actual = np.sum(np.abs(y_true))
    wape = (np.sum(np.abs(y_true - y_pred)) / (sum_actual + 1e-8)) * 100.0

    # 4. Mean Bias Error (MBE) - Indicates systematic over/under forecasting
    mbe = np.mean(y_pred - y_true)

    # 5. Tracking Signal / Percentage Bias
    pct_bias = (mbe / (np.mean(y_true) + 1e-8)) * 100.0

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE (%)": float(mape),
        "WAPE (%)": float(wape),
        "R2 Score": float(r2),
        "Mean Bias Error": float(mbe),
        "Percentage Bias (%)": float(pct_bias),
        "Accuracy (%)": max(0.0, float(100.0 - wape))
    }


def format_metrics_table(results_dict: dict) -> pd.DataFrame:
    """
    Formats multi-model results into a clean executive comparison table.
    """
    df = pd.DataFrame(results_dict).T
    df = df[["MAE", "RMSE", "WAPE (%)", "MAPE (%)", "R2 Score", "Mean Bias Error", "Accuracy (%)"]]
    return df.sort_values(by="WAPE (%)")


if __name__ == "__main__":
    y_t = np.array([100, 150, 200, 250, 300])
    y_p = np.array([105, 140, 210, 245, 310])
    print(calculate_metrics(y_t, y_p))
