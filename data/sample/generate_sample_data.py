"""
Synthetic Weather Station Dataset Generator for SIH26073 Testing.
Creates realistic weather time-series data with controlled sensor faults
and extreme weather events.
"""

import pathlib
import numpy as np
import pandas as pd

def generate_weather_dataset(seed: int = 42, num_days: int = 7) -> pd.DataFrame:
    np.random.seed(seed)
    
    # 15-minute intervals for 2 stations
    date_range = pd.date_range(start="2026-03-01 00:00:00", periods=num_days * 24 * 4, freq="15min")
    
    data_frames = []
    
    for station_id in ["AWS_STATION_01", "AWS_STATION_02"]:
        n = len(date_range)
        t = np.linspace(0, num_days * 2 * np.pi, n)
        
        # Diurnal diurnal temperature pattern + noise
        base_temp = 25.0 - 6.0 * np.cos(t) + np.random.normal(0, 0.5, n)
        # Pressure inverse to temperature + noise
        base_pressure = 1013.25 + 3.0 * np.cos(t + 0.5) + np.random.normal(0, 0.3, n)
        # Humidity inverse to temp + noise
        base_humidity = 65.0 + 20.0 * np.cos(t) + np.random.normal(0, 1.5, n)
        base_humidity = np.clip(base_humidity, 10.0, 98.0)
        
        df = pd.DataFrame({
            "timestamp": date_range,
            "station_id": station_id,
            "air_temperature": base_temp.round(2),
            "station_pressure": base_pressure.round(2),
            "relative_humidity": base_humidity.round(2),
            "synthetic_fault_label": "NORMAL"
        })
        
        # Inject controlled faults for evaluation (Station 1)
        if station_id == "AWS_STATION_01":
            # 1. Sudden Temperature Spike at index 100
            df.loc[100, "air_temperature"] += 18.5
            df.loc[100, "synthetic_fault_label"] = "SENSOR_SPIKE"
            
            # 2. Frozen Humidity Sensor at indices 200 to 230
            df.loc[200:230, "relative_humidity"] = 45.50
            df.loc[200:230, "synthetic_fault_label"] = "FROZEN_SENSOR"
            
            # 3. Sudden Pressure Spike / Out of Range at index 350
            df.loc[350, "station_pressure"] = 1250.0  # Physically impossible atmospheric pressure
            df.loc[350, "synthetic_fault_label"] = "OUT_OF_RANGE"
            
            # 4. Sensor Drift in Pressure from index 450 to 480
            drift = np.linspace(0, 40, 31)
            df.loc[450:480, "station_pressure"] = df.loc[450:480, "station_pressure"] - drift
            df.loc[450:480, "synthetic_fault_label"] = "SENSOR_DRIFT"
            
            # 5. Missing Data at index 550
            df.loc[550, "air_temperature"] = np.nan
            df.loc[550, "synthetic_fault_label"] = "MISSING_DATA"
            
            # 6. Extreme Weather Event (e.g., Storm Front) at index 600-610:
            # Drop in pressure, drop in temp, jump in humidity (COHERENT MULTIVARIATE EVENT - NOT A SINGLE SENSOR FAULT)
            df.loc[600:610, "station_pressure"] -= 12.0
            df.loc[600:610, "air_temperature"] -= 7.0
            df.loc[600:610, "relative_humidity"] = np.clip(df.loc[600:610, "relative_humidity"] + 25.0, 0, 100)
            df.loc[600:610, "synthetic_fault_label"] = "EXTREME_WEATHER_EVENT"
            
        data_frames.append(df)
        
    full_df = pd.concat(data_frames, ignore_index=True)
    return full_df

if __name__ == "__main__":
    out_dir = pathlib.Path(__file__).parent.parent / "sample"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dir = pathlib.Path(__file__).parent.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    df = generate_weather_dataset()
    
    sample_file = out_dir / "aws_weather_sample.csv"
    raw_file = raw_dir / "aws_weather_raw.csv"
    
    df.to_csv(sample_file, index=False)
    df.to_csv(raw_file, index=False)
    print(f"Generated sample dataset at {sample_file} ({len(df)} rows)")
