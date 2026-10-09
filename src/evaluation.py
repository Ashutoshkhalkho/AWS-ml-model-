"""
Evaluation Module for AWS Weather Station Anomaly Detection (MODEL V2).
Computes Accuracy, Precision, Recall, F1-score, Confusion Matrix,
per-rule false positive breakdown, and fault-type detection rates.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> Dict[str, Any]:
    """Computes standard binary classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }
    }


def evaluate_model_v2(df_test: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates Isolation Forest V2, Rule-Based Engine V2, and Combined System V2
    against ground truth injected faults. Includes per-rule FP diagnostic breakdown.
    """
    if "is_injected_anomaly" not in df_test.columns:
        raise KeyError("Evaluation requires 'is_injected_anomaly' ground truth column.")
        
    y_true = df_test["is_injected_anomaly"].astype(bool)
    
    # 1. System Level Evaluations
    metrics_if = compute_metrics(y_true, df_test["model_is_anomaly"])
    metrics_rules = compute_metrics(y_true, df_test["rule_is_anomaly"])
    metrics_combined = compute_metrics(y_true, df_test["final_is_anomaly"])
    
    # 2. Per-Rule False Positive Breakdown (Normal observations flagged by specific rules)
    is_normal = ~y_true
    rule_fp_breakdown = {}
    rule_flags = ["flag_missing", "flag_invalid_range", "flag_sudden_step", "flag_frozen", "flag_data_gap"]
    
    for rflag in rule_flags:
        if rflag in df_test.columns:
            fp_count = int((is_normal & df_test[rflag]).sum())
            rule_fp_breakdown[rflag] = fp_count
            
    # 3. Performance Breakdown by Injected Fault Type
    fault_types = [
        "TEMPERATURE_SPIKE",
        "PRESSURE_SPIKE",
        "HUMIDITY_SPIKE",
        "FROZEN_SENSOR",
        "SENSOR_DRIFT",
        "MISSING_DATA",
        "MULTIVARIATE_ANOMALY",
    ]
    
    fault_breakdown = {}
    
    for ftype in fault_types:
        total_injected = int((df_test["injected_anomaly_type"] == ftype).sum())
        if total_injected == 0:
            continue
            
        detected_by_combined = int(((df_test["injected_anomaly_type"] == ftype) & df_test["final_is_anomaly"]).sum())
        detected_by_if = int(((df_test["injected_anomaly_type"] == ftype) & df_test["model_is_anomaly"]).sum())
        detected_by_rules = int(((df_test["injected_anomaly_type"] == ftype) & df_test["rule_is_anomaly"]).sum())
        
        recall_combined = (detected_by_combined / total_injected) if total_injected > 0 else 0.0
        recall_if = (detected_by_if / total_injected) if total_injected > 0 else 0.0
        recall_rules = (detected_by_rules / total_injected) if total_injected > 0 else 0.0
        
        fault_breakdown[ftype] = {
            "total_injected": total_injected,
            "detected_combined": detected_by_combined,
            "detected_if_alone": detected_by_if,
            "detected_rules_alone": detected_by_rules,
            "recall_combined": round(float(recall_combined), 4),
            "recall_if_alone": round(float(recall_if), 4),
            "recall_rules_alone": round(float(recall_rules), 4),
        }
        
    overall_summary = {
        "total_test_observations": len(df_test),
        "total_normal_observations": int(is_normal.sum()),
        "total_injected_anomalies": int(y_true.sum()),
        "systems": {
            "isolation_forest_alone": metrics_if,
            "rule_based_alone": metrics_rules,
            "combined_system": metrics_combined,
        },
        "per_rule_false_positives": rule_fp_breakdown,
        "fault_type_breakdown": fault_breakdown,
    }
    
    return overall_summary
