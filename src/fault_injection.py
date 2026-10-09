"""
Fault Injection Module for AWS Weather Station Anomaly Detection (MODEL V2).
Generates controlled, configurable ground-truth synthetic sensor faults on a test set copy.
Leaves the original raw dataset completely untouched.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


@dataclass
class FaultInjectionConfig:
    """Configurable parameters for synthetic fault injection (Model V2)."""
    seed: int = 42
    
    # Fault probabilities / fraction of test dataset
    temp_spike_fraction: float = 0.002       # ~0.2% of rows
    press_spike_fraction: float = 0.002      # ~0.2% of rows
    hum_spike_fraction: float = 0.002        # ~0.2% of rows
    multivariate_fraction: float = 0.002     # ~0.2% of rows
    missing_data_fraction: float = 0.002     # ~0.2% of rows
    
    # Window-based faults (number of windows & length)
    frozen_sensor_num_windows: int = 15      # 15 windows
    frozen_sensor_window_len: int = 180      # 180 consecutive observations (~15 mins at 5s sampling)
    
    sensor_drift_num_windows: int = 15       # 15 windows
    sensor_drift_window_len: int = 120       # 120 consecutive observations (~10 mins at 5s sampling)
    
    # Fault magnitudes (Pressure in Pascals)
    temp_spike_min: float = 12.0             # °C step jump
    temp_spike_max: float = 22.0
    
    press_spike_min: float = 4000.0          # Pa step jump (~40 hPa)
    press_spike_max: float = 9000.0
    
    hum_spike_min: float = 25.0              # % step jump
    hum_spike_max: float = 45.0
    
    sensor_drift_max_temp: float = 10.0      # °C total drift over window
    sensor_drift_max_press: float = 3500.0   # Pa total drift over window


def inject_faults(
    df: pd.DataFrame, config: FaultInjectionConfig = FaultInjectionConfig()
) -> pd.DataFrame:
    """
    Injects ground-truth sensor faults into a copy of the input DataFrame.
    
    Returns:
        DataFrame copy with injected faults and ground truth columns:
        - is_injected_anomaly (0 or 1)
        - injected_anomaly_type (string label)
    """
    np.random.seed(config.seed)
    df_faulty = df.copy().reset_index(drop=True)
    n = len(df_faulty)
    
    # Initialize ground truth columns
    df_faulty["is_injected_anomaly"] = 0
    df_faulty["injected_anomaly_type"] = "NORMAL"
    
    available_indices = np.arange(200, n - 200)
    
    # 1. TEMPERATURE_SPIKE
    n_temp_spikes = int(n * config.temp_spike_fraction)
    spike_indices = np.random.choice(available_indices, size=n_temp_spikes, replace=False)
    for idx in spike_indices:
        if df_faulty.loc[idx, "is_injected_anomaly"] == 0:
            sign = np.random.choice([-1, 1])
            mag = np.random.uniform(config.temp_spike_min, config.temp_spike_max)
            df_faulty.loc[idx, "temperature"] += sign * mag
            df_faulty.loc[idx, "is_injected_anomaly"] = 1
            df_faulty.loc[idx, "injected_anomaly_type"] = "TEMPERATURE_SPIKE"
            
    # 2. PRESSURE_SPIKE (Pressure in Pa)
    n_press_spikes = int(n * config.press_spike_fraction)
    spike_indices_p = np.random.choice(available_indices, size=n_press_spikes, replace=False)
    for idx in spike_indices_p:
        if df_faulty.loc[idx, "is_injected_anomaly"] == 0:
            sign = np.random.choice([-1, 1])
            mag = np.random.uniform(config.press_spike_min, config.press_spike_max)
            df_faulty.loc[idx, "pressure"] += sign * mag
            df_faulty.loc[idx, "is_injected_anomaly"] = 1
            df_faulty.loc[idx, "injected_anomaly_type"] = "PRESSURE_SPIKE"

    # 3. HUMIDITY_SPIKE
    n_hum_spikes = int(n * config.hum_spike_fraction)
    spike_indices_h = np.random.choice(available_indices, size=n_hum_spikes, replace=False)
    for idx in spike_indices_h:
        if df_faulty.loc[idx, "is_injected_anomaly"] == 0:
            sign = np.random.choice([-1, 1])
            mag = np.random.uniform(config.hum_spike_min, config.hum_spike_max)
            new_val = df_faulty.loc[idx, "humidity"] + sign * mag
            df_faulty.loc[idx, "humidity"] = np.clip(new_val, 0.0, 100.0)
            df_faulty.loc[idx, "is_injected_anomaly"] = 1
            df_faulty.loc[idx, "injected_anomaly_type"] = "HUMIDITY_SPIKE"

    # 4. FROZEN_SENSOR (180 consecutive observations = ~15 mins)
    low_bound = min(50, max(1, n // 10))
    for _ in range(config.frozen_sensor_num_windows):
        max_start = n - config.frozen_sensor_window_len - low_bound
        start_idx = np.random.randint(low_bound, max_start) if max_start > low_bound else low_bound
        end_idx = min(n - 1, start_idx + config.frozen_sensor_window_len)
        target_sensor = np.random.choice(["temperature", "humidity", "pressure"])
        frozen_val = df_faulty.loc[start_idx, target_sensor]
        
        df_faulty.loc[start_idx:end_idx, target_sensor] = frozen_val
        df_faulty.loc[start_idx:end_idx, "is_injected_anomaly"] = 1
        df_faulty.loc[start_idx:end_idx, "injected_anomaly_type"] = "FROZEN_SENSOR"

    # 5. SENSOR_DRIFT (Linear progressive shift over window)
    for _ in range(config.sensor_drift_num_windows):
        max_start = n - config.sensor_drift_window_len - low_bound
        start_idx = np.random.randint(low_bound, max_start) if max_start > low_bound else low_bound
        end_idx = min(n - 1, start_idx + config.sensor_drift_window_len)
        w_len = end_idx - start_idx + 1
        
        target_sensor = np.random.choice(["temperature", "pressure"])
        if target_sensor == "temperature":
            drift_curve = np.linspace(0, config.sensor_drift_max_temp, w_len)
        else:
            drift_curve = np.linspace(0, config.sensor_drift_max_press, w_len)
            
        sign = np.random.choice([-1, 1])
        df_faulty.loc[start_idx:end_idx, target_sensor] += sign * drift_curve
        df_faulty.loc[start_idx:end_idx, "is_injected_anomaly"] = 1
        df_faulty.loc[start_idx:end_idx, "injected_anomaly_type"] = "SENSOR_DRIFT"

    # 6. MISSING_DATA
    n_missing = int(n * config.missing_data_fraction)
    missing_indices = np.random.choice(available_indices, size=n_missing, replace=False)
    for idx in missing_indices:
        if df_faulty.loc[idx, "is_injected_anomaly"] == 0:
            target_col = np.random.choice(["temperature", "humidity", "pressure"])
            df_faulty.loc[idx, target_col] = np.nan
            df_faulty.loc[idx, "is_injected_anomaly"] = 1
            df_faulty.loc[idx, "injected_anomaly_type"] = "MISSING_DATA"

    # 7. MULTIVARIATE_ANOMALY
    n_multi = int(n * config.multivariate_fraction)
    multi_indices = np.random.choice(available_indices, size=n_multi, replace=False)
    for idx in multi_indices:
        if df_faulty.loc[idx, "is_injected_anomaly"] == 0:
            df_faulty.loc[idx, "temperature"] += 14.0
            df_faulty.loc[idx, "humidity"] = max(5.0, df_faulty.loc[idx, "humidity"] - 25.0)
            df_faulty.loc[idx, "pressure"] -= 5000.0  # Pa
            df_faulty.loc[idx, "is_injected_anomaly"] = 1
            df_faulty.loc[idx, "injected_anomaly_type"] = "MULTIVARIATE_ANOMALY"

    return df_faulty
