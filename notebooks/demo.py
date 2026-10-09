"""
Interactive Demo Script for AWS Anomaly Detection.
Runs pipeline and plots interactive temperature, pressure, and humidity anomaly charts.
"""

import sys
import pathlib

# Add project root to sys.path
root_dir = pathlib.Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_weather_data
from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.anomaly_detection import SensorRuleConfig, apply_rule_based_checks, AWSAnomalyDetector, classify_anomalies
from src.explainability import explain_anomalies

def run_demo():
    print("Loading data...")
    df_raw = load_weather_data("data/sample/aws_weather_sample.csv")
    df_prep, _ = preprocess_data(df_raw)
    df_feat, feature_cols = engineer_features(df_prep, window_size=4)
    
    rule_config = SensorRuleConfig()
    df_rules = apply_rule_based_checks(df_feat, config=rule_config)
    
    detector = AWSAnomalyDetector(contamination=0.05, random_state=42)
    detector.fit(df_rules, feature_cols=feature_cols)
    
    is_model_anomaly, scores, raw_preds = detector.predict(df_rules)
    df_rules["is_model_anomaly"] = is_model_anomaly
    df_rules["anomaly_score"] = scores
    
    df_classified = classify_anomalies(df_rules, config=rule_config)
    df_explained = explain_anomalies(df_classified, config=rule_config)
    
    # Filter Station 1
    st1 = df_explained[df_explained["station_id"] == "AWS_STATION_01"].copy()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Temperature Plot
    axes[0].plot(st1["timestamp"], st1["temperature"], label="Temperature (°C)", color="tab:blue")
    anom_t = st1[st1["is_anomaly"]]
    axes[0].scatter(anom_t["timestamp"], anom_t["temperature"], color="tab:red", label="Flagged Anomaly", zorder=5)
    axes[0].set_ylabel("Temp (°C)")
    axes[0].set_title("AWS Station 01: Temperature Anomalies")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Pressure Plot
    axes[1].plot(st1["timestamp"], st1["pressure"], label="Pressure (hPa)", color="tab:green")
    anom_p = st1[st1["is_anomaly"]]
    axes[1].scatter(anom_p["timestamp"], anom_p["pressure"], color="tab:red", label="Flagged Anomaly", zorder=5)
    axes[1].set_ylabel("Pressure (hPa)")
    axes[1].set_title("AWS Station 01: Pressure Anomalies")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Anomaly Score Plot
    axes[2].plot(st1["timestamp"], st1["anomaly_score"], label="Anomaly Score", color="tab:purple")
    axes[2].set_ylabel("Score [0, 1]")
    axes[2].set_xlabel("Timestamp")
    axes[2].set_title("AWS Station 01: Isolation Forest Anomaly Score Distribution")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = root_dir / "data" / "processed" / "demo_anomaly_plot.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"Demo plot generated and saved to: {plot_path.resolve()}")

if __name__ == "__main__":
    run_demo()
