"""
Data loader and flexible column mapping layer for Automatic Weather Station (AWS) data.
"""

import pathlib
from typing import Dict, Tuple, Optional
import pandas as pd

# Canonical column names required by the pipeline
CANONICAL_COLUMNS = {
    "timestamp": "timestamp",
    "temperature": "temperature",
    "pressure": "pressure",
    "humidity": "humidity",
    "station_id": "station_id",
}

# Column aliases for flexible column mapping
COLUMN_ALIASES = {
    "timestamp": ["timestamp", "datetime", "date_time", "date_time_utc", "time", "date", "timestamp_utc"],
    "temperature": ["temperature", "temp", "air_temperature", "air_temp", "temp_c", "dry_bulb_temp", "t_air"],
    "pressure": ["pressure", "atmospheric_pressure", "station_pressure", "baro_pressure", "press", "pres", "mslp"],
    "humidity": ["humidity", "relative_humidity", "rh", "rel_hum", "relative_humidity_pct", "rh_pct"],
    "station_id": ["station_id", "station", "aws_id", "stn_id", "site_id", "station_name", "id"],
}


def map_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Maps user CSV columns to canonical names using defined aliases.
    
    Returns:
        Mapped DataFrame and dictionary of {canonical_name: original_name}
    """
    df_mapped = df.copy()
    existing_cols_lower = {col.strip().lower(): col for col in df.columns}
    
    mapping = {}
    missing_required = []
    
    # Required canonical columns
    required_keys = ["timestamp", "temperature", "pressure", "humidity"]
    
    for key, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            if alias.lower() in existing_cols_lower:
                found = existing_cols_lower[alias.lower()]
                break
        
        if found:
            mapping[found] = CANONICAL_COLUMNS[key]
        elif key in required_keys:
            missing_required.append(key)
            
    if missing_required:
        raise ValueError(
            f"Data Loading Error: Missing required weather variable(s): {missing_required}. "
            f"Found columns in file: {list(df.columns)}. "
            f"Supported aliases: {COLUMN_ALIASES}"
        )
        
    df_mapped.rename(columns=mapping, inplace=True)
    
    # Handle optional station_id if not present in dataset
    if CANONICAL_COLUMNS["station_id"] not in df_mapped.columns:
        df_mapped[CANONICAL_COLUMNS["station_id"]] = "AWS_DEFAULT"
        
    return df_mapped, {v: k for k, v in mapping.items()}


def load_weather_data(file_path: str | pathlib.Path) -> pd.DataFrame:
    """
    Loads weather data from CSV and standardizes column names.
    """
    path = pathlib.Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")
        
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")
        
    mapped_df, _ = map_columns(df)
    return mapped_df
