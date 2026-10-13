"""
Feature Engineering Module for AWS Weather Station Anomaly Detection (MODEL V2).
Derives time-aware rate-of-change, step changes, causal rolling statistics,
deviations from mean, rolling z-scores, and short-term volatility.
Strictly causal (backward-looking) to prevent data leakage.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np


def engineer_features(
    df: pd.DataFrame, window_size: int = 12
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Engineers time-series features for Model V2.
    
    Args:
        df: Input DataFrame containing 'time' (or 'timestamp'), 'temperature', 'humidity', 'pressure', 'lux'.
        window_size: Long rolling window size (default: 12 observations ≈ 1 minute at 5-sec sampling).
        
    Returns:
        DataFrame with engineered features and a list of feature column names.
    """
    df = df.copy()
    if "lux" not in df.columns:
        df["lux"] = 0.0
    
    # 1. Parse time column to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        raise KeyError("Input DataFrame must contain 'time' or 'timestamp' column.")
        
    # Sort chronologically to guarantee correct lag calculations
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 2. Time-aware delta in seconds (dt_seconds)
    dt = df["timestamp"].diff().dt.total_seconds()
    df["dt_seconds"] = dt.fillna(5.0).apply(lambda x: x if x > 0 else 5.0)
    
    # 3. Temporal Features (Step Difference)
    df["temperature_change"] = df["temperature"].diff().fillna(0.0)
    df["humidity_change"] = df["humidity"].diff().fillna(0.0)
    df["pressure_change"] = df["pressure"].diff().fillna(0.0)
    
    # 4. Time-Aware Rate-of-Change Features (Units per second)
    df["temperature_rate"] = df["temperature_change"] / df["dt_seconds"]
    df["humidity_rate"] = df["humidity_change"] / df["dt_seconds"]
    df["pressure_rate"] = df["pressure_change"] / df["dt_seconds"]
    
    # 5. Long Rolling Window Features (Backward-looking, causal, min_periods=1)
    df["temperature_rolling_mean"] = (
        df["temperature"].rolling(window=window_size, min_periods=1).mean()
    )
    df["humidity_rolling_mean"] = (
        df["humidity"].rolling(window=window_size, min_periods=1).mean()
    )
    df["pressure_rolling_mean"] = (
        df["pressure"].rolling(window=window_size, min_periods=1).mean()
    )
    
    df["temperature_rolling_std"] = (
        df["temperature"].rolling(window=window_size, min_periods=1).std().fillna(0.0)
    )
    df["humidity_rolling_std"] = (
        df["humidity"].rolling(window=window_size, min_periods=1).std().fillna(0.0)
    )
    df["pressure_rolling_std"] = (
        df["pressure"].rolling(window=window_size, min_periods=1).std().fillna(0.0)
    )
    
    # 6. Causal Deviations from Rolling Mean & Rolling Z-Scores
    df["temperature_dev_from_mean"] = df["temperature"] - df["temperature_rolling_mean"]
    df["humidity_dev_from_mean"] = df["humidity"] - df["humidity_rolling_mean"]
    df["pressure_dev_from_mean"] = df["pressure"] - df["pressure_rolling_mean"]
    
    df["temperature_rolling_zscore"] = (
        df["temperature_dev_from_mean"] / (df["temperature_rolling_std"] + 1e-5)
    ).fillna(0.0)
    df["humidity_rolling_zscore"] = (
        df["humidity_dev_from_mean"] / (df["humidity_rolling_std"] + 1e-5)
    ).fillna(0.0)
    df["pressure_rolling_zscore"] = (
        df["pressure_dev_from_mean"] / (df["pressure_rolling_std"] + 1e-5)
    ).fillna(0.0)
    
    # 7. Short-Term Volatility Features (5-step window ≈ 25s)
    df["temperature_short_std"] = (
        df["temperature"].rolling(window=5, min_periods=1).std().fillna(0.0)
    )
    df["humidity_short_std"] = (
        df["humidity"].rolling(window=5, min_periods=1).std().fillna(0.0)
    )
    df["pressure_short_std"] = (
        df["pressure"].rolling(window=5, min_periods=1).std().fillna(0.0)
    )
    
    # Feature columns for Model V2 input
    feature_cols = [
        "temperature",
        "humidity",
        "pressure",
        "lux",
        "dt_seconds",
        "temperature_change",
        "humidity_change",
        "pressure_change",
        "temperature_rate",
        "humidity_rate",
        "pressure_rate",
        "temperature_rolling_mean",
        "humidity_rolling_mean",
        "pressure_rolling_mean",
        "temperature_rolling_std",
        "humidity_rolling_std",
        "pressure_rolling_std",
        "temperature_dev_from_mean",
        "humidity_dev_from_mean",
        "pressure_dev_from_mean",
        "temperature_rolling_zscore",
        "humidity_rolling_zscore",
        "pressure_rolling_zscore",
        "temperature_short_std",
        "humidity_short_std",
        "pressure_short_std",
    ]
    
    return df, feature_cols
