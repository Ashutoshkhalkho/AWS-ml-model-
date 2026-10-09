"""
Preprocessing module for AWS Weather Station data.
Handles timestamp parsing, sorting, duplicate removal, numeric conversion,
and missing value imputation while preserving extreme weather events.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocesses raw mapped weather data.
    
    Steps:
    1. Timestamp parsing & sorting per station
    2. Numeric conversion for weather metrics
    3. Duplicate detection and handling
    4. Missing value analysis & imputation (linear interpolation for minor gaps)
    5. Returns processed DataFrame and a summary metadata dict.
    """
    df = df.copy()
    report = {}
    
    # 1. Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    invalid_ts_count = df["timestamp"].isna().sum()
    if invalid_ts_count > 0:
        # Drop rows where timestamp cannot be parsed
        df = df.dropna(subset=["timestamp"])
    
    # Ensure station_id is string
    df["station_id"] = df["station_id"].astype(str)
    
    # 2. Convert weather parameters to float numeric
    num_cols = ["temperature", "pressure", "humidity"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    # 3. Sort by station_id and timestamp
    df = df.sort_values(by=["station_id", "timestamp"]).reset_index(drop=True)
    
    # 4. Check duplicates per station and timestamp
    dup_mask = df.duplicated(subset=["station_id", "timestamp"], keep="first")
    duplicate_count = int(dup_mask.sum())
    if duplicate_count > 0:
        df = df[~dup_mask].reset_index(drop=True)
        
    # 5. Missing value report
    missing_before = df[num_cols].isna().sum().to_dict()
    
    # Create indicator mask for original missing values (useful for rule checks)
    df["has_missing_sensor"] = df[num_cols].isna().any(axis=1)
    
    # Station-wise interpolation for missing numerical values
    # Limit max consecutive NaN interpolation to 3 readings (45 mins) to prevent artificial trends
    processed_dfs = []
    for station_id, group in df.groupby("station_id", group_keys=False):
        group_clean = group.copy()
        for col in num_cols:
            # Interpolate short missing gaps per station
            group_clean[col] = group_clean[col].interpolate(method="linear", limit=3, limit_direction="both")
            # If still NaN (long gap), forward fill then backward fill
            group_clean[col] = group_clean[col].ffill().bfill()
        processed_dfs.append(group_clean)
        
    df_clean = pd.concat(processed_dfs, ignore_index=True)
    
    report = {
        "total_rows": len(df_clean),
        "invalid_timestamps_removed": int(invalid_ts_count),
        "duplicates_removed": duplicate_count,
        "missing_values_before_imputation": missing_before,
        "stations_processed": list(df_clean["station_id"].unique()),
    }
    
    return df_clean, report
