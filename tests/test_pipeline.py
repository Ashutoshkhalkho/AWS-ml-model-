"""
Unit Tests for Model V2 Pipeline (SIH26073).
Tests fault injection, feature engineering, Isolation Forest V2, rule checks, and evaluation.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import pathlib

from src.fault_injection import FaultInjectionConfig, inject_faults
from src.feature_engineering import engineer_features
from src.anomaly_detection import (
    RuleThresholdConfig,
    apply_rule_checks,
    IsolationForestDetectorV2,
    synthesize_results_v2,
)
from src.evaluation import evaluate_model_v2


def test_fault_injection_v2():
    df = pd.DataFrame({
        "time": pd.date_range("2026-03-01", periods=500, freq="5s").astype(str),
        "temperature": [22.0] * 500,
        "humidity": [40.0] * 500,
        "pressure": [98500.0] * 500,
        "lux": [10.0] * 500,
    })
    
    config = FaultInjectionConfig(seed=42)
    df_faulty = inject_faults(df, config)
    
    assert "is_injected_anomaly" in df_faulty.columns
    assert "injected_anomaly_type" in df_faulty.columns
    assert (df_faulty["is_injected_anomaly"] == 1).sum() > 0
    assert "temperature" in df_faulty.columns


def test_feature_engineering_v2():
    df = pd.DataFrame({
        "time": pd.date_range("2026-03-01", periods=20, freq="5s").astype(str),
        "temperature": np.linspace(20, 25, 20),
        "humidity": np.linspace(40, 50, 20),
        "pressure": np.linspace(98000, 98500, 20),
        "lux": [15.0] * 20,
    })
    
    df_feat, feature_cols = engineer_features(df, window_size=5)
    
    assert "temperature_change" in df_feat.columns
    assert "temperature_rate" in df_feat.columns
    assert "temperature_rolling_zscore" in df_feat.columns
    assert "temperature_short_std" in df_feat.columns
    assert len(feature_cols) == 26


def test_detector_v2_fit_predict_save_load():
    df = pd.DataFrame({
        "time": pd.date_range("2026-03-01", periods=100, freq="5s").astype(str),
        "temperature": np.random.normal(22, 1, 100),
        "humidity": np.random.normal(40, 2, 100),
        "pressure": np.random.normal(98500, 100, 100),
        "lux": np.random.normal(10, 2, 100),
    })
    
    df_feat, feature_cols = engineer_features(df, window_size=5)
    
    detector = IsolationForestDetectorV2(n_estimators=50, contamination=0.01, random_state=42)
    detector.fit(df_feat, feature_cols)
    
    model_is_anom, scores, raw_preds = detector.predict(df_feat)
    assert len(model_is_anom) == 100
    assert len(scores) == 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        m_file = pathlib.Path(tmpdir) / "test_model.joblib"
        f_file = pathlib.Path(tmpdir) / "test_features.json"
        detector.save(m_file, f_file)
        
        loaded = IsolationForestDetectorV2.load(m_file)
        model_is_anom2, scores2, _ = loaded.predict(df_feat)
        np.testing.assert_array_almost_equal(scores.values, scores2.values)


def test_evaluation_v2():
    df_test = pd.DataFrame({
        "is_injected_anomaly": [0, 0, 1, 1],
        "injected_anomaly_type": ["NORMAL", "NORMAL", "TEMPERATURE_SPIKE", "FROZEN_SENSOR"],
        "model_is_anomaly": [False, False, True, False],
        "rule_is_anomaly": [False, False, False, True],
        "final_is_anomaly": [False, False, True, True],
        "temperature": [22.0, 22.1, 40.0, 22.0],
        "humidity": [40.0, 40.1, 40.0, 40.0],
        "pressure": [98500.0, 98500.0, 98500.0, 98500.0],
        "lux": [10.0, 10.0, 10.0, 10.0],
    })
    
    metrics = evaluate_model_v2(df_test)
    assert metrics["systems"]["combined_system"]["precision"] == 1.0
    assert metrics["systems"]["combined_system"]["recall"] == 1.0
    assert "TEMPERATURE_SPIKE" in metrics["fault_type_breakdown"]
