"""
Explainability Module for AWS Weather Station Anomaly Detection (MODEL V2).
Generates human-readable, data-grounded explanations using RuleThresholdConfig.
Pressure values are displayed in hPa (with Pa internally).
"""

from typing import Tuple, List, Dict, Any
import pandas as pd
from src.anomaly_detection import RuleThresholdConfig


def generate_explanation(
    row: pd.Series, config: RuleThresholdConfig = RuleThresholdConfig()
) -> Tuple[str, str]:
    """
    Generates affected_parameter and human-readable explanation string for a given row.
    
    Returns:
        (affected_parameter, explanation_text)
    """
    atype = row.get("detected_anomaly_type", row.get("anomaly_type", "NORMAL"))
    
    if atype == "NORMAL":
        return "NONE", "Observation is within normal operational parameters."
        
    if atype == "MISSING_DATA":
        missing_vars = []
        for var in ["temperature", "humidity", "pressure", "lux"]:
            if pd.isna(row.get(var)):
                missing_vars.append(var)
        affected = ", ".join(missing_vars) if missing_vars else "sensor_reading"
        return affected, f"Missing sensor reading(s) detected for: {affected}."
        
    if atype == "OUT_OF_RANGE":
        reasons = []
        affected = []
        if row.get("temperature", 0) < config.temp_min or row.get("temperature", 0) > config.temp_max:
            affected.append("temperature")
            reasons.append(f"Temperature {row['temperature']:.2f}°C is outside valid range [{config.temp_min}°C, {config.temp_max}°C]")
        if row.get("pressure", 0) < config.press_min or row.get("pressure", 0) > config.press_max:
            affected.append("pressure")
            press_hpa = row['pressure'] / 100.0
            reasons.append(f"Pressure {press_hpa:.2f} hPa ({row['pressure']:.0f} Pa) is outside valid range [{config.press_min/100.0:.1f} hPa, {config.press_max/100.0:.1f} hPa]")
        if row.get("humidity", 0) < config.hum_min or row.get("humidity", 0) > config.hum_max:
            affected.append("humidity")
            reasons.append(f"Humidity {row['humidity']:.2f}% is outside valid range [{config.hum_min}%, {config.hum_max}%]")
            
        return ", ".join(affected) if affected else "MULTIVARIATE", "; ".join(reasons) + "."
        
    if atype in ["TEMPERATURE_SPIKE", "PRESSURE_SPIKE", "HUMIDITY_SPIKE", "SENSOR_SPIKE", "RATE_OF_CHANGE"]:
        reasons = []
        affected = []
        t_change = row.get("temperature_change", row.get("temp_diff", 0.0))
        p_change = row.get("pressure_change", row.get("pressure_diff", 0.0))
        h_change = row.get("humidity_change", row.get("humidity_diff", 0.0))
        
        if abs(t_change) > config.max_temp_step:
            affected.append("temperature")
            reasons.append(f"Temperature changed by {t_change:+.2f}°C in single observation (threshold: {config.max_temp_step}°C)")
        if abs(p_change) > config.max_press_step:
            affected.append("pressure")
            p_change_hpa = p_change / 100.0
            reasons.append(f"Pressure changed by {p_change_hpa:+.2f} hPa ({p_change:+.0f} Pa) in single observation (threshold: {config.max_press_step/100.0:.1f} hPa)")
        if abs(h_change) > config.max_hum_step:
            affected.append("humidity")
            reasons.append(f"Humidity changed by {h_change:+.2f}% in single observation (threshold: {config.max_hum_step}%)")
            
        return ", ".join(affected) if affected else "MULTIVARIATE", "; ".join(reasons) + "."
        
    if atype == "FROZEN_SENSOR":
        aff = row.get("affected_sensor", "MULTIVARIATE")
        return aff, f"Sensor readings for {aff} remained frozen/stuck for >={config.frozen_consecutive_limit} consecutive readings (~15 minutes)."
        
    if atype == "DATA_GAP":
        dt_sec = row.get("dt_seconds", 0.0)
        return "timestamp", f"Abnormal communication data gap detected: {dt_sec:.1f} seconds between consecutive observations (threshold: {config.max_gap_seconds}s)."
        
    if atype in ["SENSOR_DRIFT", "MULTIVARIATE_ANOMALY"]:
        score = row.get("anomaly_score", 0.0)
        aff = row.get("affected_sensor", "MULTIVARIATE")
        press_hpa = row.get("pressure", 0.0) / 100.0
        expl = (
            f"Isolation Forest model detected statistical anomaly (score: {score:.4f}) "
            f"for {aff} differing from normal station baseline (Temp: {row.get('temperature', 0):.2f}°C, "
            f"Pressure: {press_hpa:.2f} hPa, Humidity: {row.get('humidity', 0):.2f}%)."
        )
        return aff, expl
        
    return "UNKNOWN", "Unspecified sensor anomaly detected."


def explain_anomalies(
    df: pd.DataFrame, config: RuleThresholdConfig = RuleThresholdConfig()
) -> pd.DataFrame:
    """
    Adds 'affected_parameter' and 'explanation' columns to DataFrame.
    """
    df = df.copy()
    affected_params = []
    explanations = []
    
    for _, row in df.iterrows():
        aff, expl = generate_explanation(row, config)
        affected_params.append(aff)
        explanations.append(expl)
        
    df["affected_parameter"] = affected_params
    df["explanation"] = explanations
    return df
