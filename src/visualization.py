"""
Visualization Module for AWS Weather Station Anomaly Detection (MODEL V2).
Generates publication-quality figures saved to reports/model_v2/.
"""

import pathlib
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_v2_plots(df_test: pd.DataFrame, metrics: Dict[str, Any], output_dir: str | pathlib.Path = "reports/model_v2"):
    """
    Generates 7 evaluation plots for Model V2.
    """
    out_path = pathlib.Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Subsample for time series plotting if dataset is large (e.g. plot first 5000 rows for readability)
    df_sub = df_test.head(5000).copy() if len(df_test) > 5000 else df_test.copy()
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    
    # 1. Normal vs Anomalous Timeline
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_sub["timestamp"], df_sub["temperature"], label="Observed Temperature (°C)", color="#2b5c8f", alpha=0.7, lw=1)
    anom_rows = df_sub[df_sub["final_is_anomaly"]]
    ax.scatter(anom_rows["timestamp"], anom_rows["temperature"], color="#d9534f", label="Detected Anomaly", s=25, zorder=5)
    ax.set_title("AWS Weather Station (Model V2): Normal vs Anomalous Observations Over Time", fontsize=12, fontweight="bold")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path / "normal_vs_anomalous_timeline.png", dpi=200)
    plt.close(fig)

    # 2. Temperature Anomalies
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_sub["timestamp"], df_sub["temperature"], color="#337ab7", alpha=0.8, label="Temperature (°C)")
    temp_anom = df_sub[df_sub["final_is_anomaly"] & (df_sub["detected_anomaly_type"].isin(["TEMPERATURE_SPIKE", "FROZEN_SENSOR", "SENSOR_DRIFT"]))]
    ax.scatter(temp_anom["timestamp"], temp_anom["temperature"], color="#d9534f", label="Temperature Anomaly", s=30, zorder=5)
    ax.set_title("Temperature Sensor Anomalies (Spikes, Drift, Frozen)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path / "temperature_anomalies.png", dpi=200)
    plt.close(fig)

    # 3. Pressure Anomalies (Pressure displayed in hPa)
    fig, ax = plt.subplots(figsize=(14, 5))
    press_hpa = df_sub["pressure"] / 100.0
    ax.plot(df_sub["timestamp"], press_hpa, color="#5cb85c", alpha=0.8, label="Pressure (hPa)")
    press_anom_idx = df_sub[df_sub["final_is_anomaly"] & (df_sub["detected_anomaly_type"].isin(["PRESSURE_SPIKE", "SENSOR_DRIFT"]))].index
    ax.scatter(df_sub.loc[press_anom_idx, "timestamp"], press_hpa.loc[press_anom_idx], color="#d9534f", label="Pressure Anomaly", s=30, zorder=5)
    ax.set_title("Pressure Sensor Anomalies (Displayed in hPa)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Pressure (hPa)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path / "pressure_anomalies.png", dpi=200)
    plt.close(fig)

    # 4. Humidity Anomalies
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_sub["timestamp"], df_sub["humidity"], color="#f0ad4e", alpha=0.8, label="Humidity (%)")
    hum_anom = df_sub[df_sub["final_is_anomaly"] & (df_sub["detected_anomaly_type"].isin(["HUMIDITY_SPIKE", "FROZEN_SENSOR"]))]
    ax.scatter(hum_anom["timestamp"], hum_anom["humidity"], color="#d9534f", label="Humidity Anomaly", s=30, zorder=5)
    ax.set_title("Relative Humidity Sensor Anomalies", fontsize=12, fontweight="bold")
    ax.set_ylabel("Humidity (%)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path / "humidity_anomalies.png", dpi=200)
    plt.close(fig)

    # 5. Anomaly Score Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    norm_scores = df_test[~df_test["final_is_anomaly"]]["anomaly_score"]
    anom_scores = df_test[df_test["final_is_anomaly"]]["anomaly_score"]
    
    ax.hist(norm_scores, bins=50, alpha=0.6, color="#2b5c8f", label="Normal Score Distribution", density=True)
    ax.hist(anom_scores, bins=50, alpha=0.6, color="#d9534f", label="Anomaly Score Distribution", density=True)
    ax.set_title("Model V2 Isolation Forest Anomaly Score Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Normalized Anomaly Score [0, 1]")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path / "anomaly_score_distribution.png", dpi=200)
    plt.close(fig)

    # 6. Confusion Matrices (IF, Rules, Combined)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    system_keys = [("isolation_forest_alone", "Isolation Forest V2"), ("rule_based_alone", "Rule Checks V2"), ("combined_system", "Combined System V2")]
    
    for idx, (sys_key, title) in enumerate(system_keys):
        cm = metrics["systems"][sys_key]["confusion_matrix"]
        cm_matrix = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
        sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False,
                    xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
        axes[idx].set_title(title, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Ground Truth Label")
        
    plt.tight_layout()
    fig.savefig(out_path / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    # 7. Detection Performance by Fault Type
    fig, ax = plt.subplots(figsize=(12, 5))
    breakdown = metrics.get("fault_type_breakdown", {})
    
    fault_names = list(breakdown.keys())
    rec_combined = [breakdown[k]["recall_combined"] * 100 for k in fault_names]
    rec_if = [breakdown[k]["recall_if_alone"] * 100 for k in fault_names]
    rec_rules = [breakdown[k]["recall_rules_alone"] * 100 for k in fault_names]
    
    x = np.arange(len(fault_names))
    width = 0.25
    
    ax.bar(x - width, rec_if, width, label="Isolation Forest Alone", color="#5bc0de")
    ax.bar(x, rec_rules, width, label="Rules Alone", color="#f0ad4e")
    ax.bar(x + width, rec_combined, width, label="Combined System V2", color="#5cb85c")
    
    ax.set_ylabel("Detection Recall Rate (%)")
    ax.set_title("Model V2 Anomaly Detection Performance by Injected Fault Type", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(fault_names, rotation=25, ha="right")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right")
    
    for i in x:
        ax.text(i + width, rec_combined[i] + 2, f"{rec_combined[i]:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        
    plt.tight_layout()
    fig.savefig(out_path / "detection_performance_by_type.png", dpi=200)
    plt.close(fig)
    print(f"Generated 7 V2 plots saved to: {out_path.resolve()}")
