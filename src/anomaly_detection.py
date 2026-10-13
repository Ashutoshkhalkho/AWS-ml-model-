"""
Anomaly Detection Module for AWS Weather Station Data (MODEL V2).
Combines Isolation Forest ML with Deterministic Sensor Quality Rules.
Supports separate per-rule false-positive tracking, refined frozen sensor detection,
and confident statistical decision synthesis.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import pathlib
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


@dataclass
class RuleThresholdConfig:
    """Configurable physical limits and change thresholds for AWS weather station data."""
    # Physical plausibility limits (Pressure in Pascals)
    temp_min: float = -40.0         # °C
    temp_max: float = 60.0          # °C
    hum_min: float = 0.0            # %
    hum_max: float = 100.0          # %
    press_min: float = 70000.0      # Pa (~700 hPa)
    press_max: float = 110000.0     # Pa (~1100 hPa)
    lux_min: float = 0.0
    
    # Step change thresholds (for 5-sec sampling, Pressure in Pa)
    max_temp_step: float = 5.0      # °C step
    max_hum_step: float = 15.0      # % step
    max_press_step: float = 2000.0  # Pa step (~20 hPa)
    
    # Single configuration variable for frozen sensor limit (180 steps ≈ 15 mins at 5s sampling)
    frozen_consecutive_limit: int = 180
    
    # Abnormal time gap (seconds)
    max_gap_seconds: float = 60.0


def apply_rule_checks(
    df: pd.DataFrame, config: RuleThresholdConfig = RuleThresholdConfig()
) -> pd.DataFrame:
    """
    Applies deterministic physical sensor rules and data quality checks for Model V2.
    Detects frozen behavior separately for temperature, humidity, and pressure.
    
    Returns:
        DataFrame with detailed per-rule flag columns and boolean 'rule_is_anomaly'.
    """
    df = df.copy()
    
    # 1. Missing Values Flag
    df["flag_missing"] = df[["temperature", "humidity", "pressure", "lux"]].isna().any(axis=1)
    
    # 2. Out of Physical Range Flags
    df["flag_temp_range"] = (df["temperature"] < config.temp_min) | (df["temperature"] > config.temp_max)
    df["flag_hum_range"] = (df["humidity"] < config.hum_min) | (df["humidity"] > config.hum_max)
    df["flag_press_range"] = (df["pressure"] < config.press_min) | (df["pressure"] > config.press_max)
    df["flag_lux_range"] = df["lux"] < config.lux_min
    df["flag_invalid_range"] = (
        df["flag_temp_range"] | df["flag_hum_range"] | df["flag_press_range"] | df["flag_lux_range"]
    )
    
    # 3. Sudden Step Change Flags
    df["flag_temp_step"] = df["temperature_change"].abs() > config.max_temp_step
    df["flag_hum_step"] = df["humidity_change"].abs() > config.max_hum_step
    df["flag_press_step"] = df["pressure_change"].abs() > config.max_press_step
    df["flag_sudden_step"] = df["flag_temp_step"] | df["flag_hum_step"] | df["flag_press_step"]
    
    # 4. Separate Frozen Sensor Detection (using single config variable frozen_consecutive_limit = 180)
    limit = config.frozen_consecutive_limit
    
    # Pressure: 180 consecutive identical readings in Pascals
    df["flag_press_frozen"] = (
        (df["pressure"].diff() == 0)
        .groupby((df["pressure"].diff() != 0).cumsum())
        .transform("size") >= limit
    )
    
    # Humidity: 180 consecutive identical readings with pressure_change == 0
    df["flag_hum_frozen"] = (
        ((df["humidity"].diff() == 0)
        .groupby((df["humidity"].diff() != 0).cumsum())
        .transform("size") >= limit) & (df["pressure_change"] == 0)
    )
    
    # Temperature: 180 consecutive identical readings with pressure_change == 0
    df["flag_temp_frozen"] = (
        ((df["temperature"].diff() == 0)
        .groupby((df["temperature"].diff() != 0).cumsum())
        .transform("size") >= limit) & (df["pressure_change"] == 0)
    )
    
    df["flag_frozen"] = df["flag_temp_frozen"] | df["flag_hum_frozen"] | df["flag_press_frozen"]
    
    # 5. Abnormal Time Gap Flag
    if "timestamp" in df.columns:
        dt_sec = df["timestamp"].diff().dt.total_seconds()
        df["flag_data_gap"] = dt_sec > config.max_gap_seconds
    else:
        df["flag_data_gap"] = False
        
    # Combined Rule Anomaly Flag
    df["rule_is_anomaly"] = (
        df["flag_missing"] | df["flag_invalid_range"] | df["flag_sudden_step"] | df["flag_frozen"] | df["flag_data_gap"]
    )
    
    return df


class IsolationForestDetectorV2:
    """
    Isolation Forest Anomaly Detector Wrapper for Model V2.
    Includes feature scaling, model persistence, and score normalization.
    """
    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float | str = 0.01,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.feature_cols: List[str] = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, feature_cols: List[str]):
        """Fits StandardScaler and IsolationForest on input training features."""
        self.feature_cols = feature_cols
        X = df[feature_cols].copy().fillna(0.0)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Runs model inference on input features.
        
        Returns:
            - model_is_anomaly (boolean Series, True if anomaly)
            - anomaly_score (float Series, normalized into [0, 1], higher = more anomalous)
            - raw_predictions (int Series, 1 = normal, -1 = anomaly)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before calling predict().")
            
        X = df[self.feature_cols].copy().fillna(0.0)
        X_scaled = self.scaler.transform(X)
        
        raw_preds = self.model.predict(X_scaled)  # 1 = normal, -1 = anomaly
        model_is_anomaly = pd.Series(raw_preds == -1, index=df.index)
        
        # Isolation Forest decision_function returns negative values for anomalies
        raw_scores = -self.model.decision_function(X_scaled)
        
        # Min-max normalize scores to [0.0, 1.0]
        s_min, s_max = raw_scores.min(), raw_scores.max()
        if s_max > s_min:
            norm_scores = (raw_scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.zeros_like(raw_scores)
            
        anomaly_scores = pd.Series(norm_scores.round(4), index=df.index)
        raw_predictions = pd.Series(raw_preds, index=df.index)
        
        return model_is_anomaly, anomaly_scores, raw_predictions

    def save(self, model_file_path: str | pathlib.Path, feature_file_path: str | pathlib.Path):
        """Saves trained model binary and feature metadata."""
        m_path = pathlib.Path(model_file_path)
        f_path = pathlib.Path(feature_file_path)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        f_path.parent.mkdir(parents=True, exist_ok=True)
        
        artifact = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "random_state": self.random_state,
        }
        joblib.dump(artifact, m_path)
        
        with open(f_path, "w") as f:
            json.dump({"features": self.feature_cols, "n_features": len(self.feature_cols)}, f, indent=2)

    @classmethod
    def load(cls, model_file_path: str | pathlib.Path) -> "IsolationForestDetectorV2":
        """Loads trained detector instance from saved joblib artifact."""
        m_path = pathlib.Path(model_file_path)
        if not m_path.exists():
            raise FileNotFoundError(f"Model file not found at: {m_path}")
            
        artifact = joblib.load(m_path)
        detector = cls(
            n_estimators=artifact.get("n_estimators", 100),
            contamination=artifact.get("contamination", 0.01),
            random_state=artifact.get("random_state", 42),
        )
        detector.model = artifact["model"]
        detector.scaler = artifact["scaler"]
        detector.feature_cols = artifact["feature_cols"]
        detector.is_fitted = True
        return detector


def synthesize_results_v2(df: pd.DataFrame, score_threshold: float = 0.55) -> pd.DataFrame:
    """
    Intelligent decision synthesis for Model V2:
    - Hard physical/data quality rule violations are automatic anomalies.
    - Model/statistical anomalies require sufficient anomaly score evidence (anomaly_score >= score_threshold)
      OR high z-score deviation.
    """
    df = df.copy()
    
    # Statistical anomaly confirmation logic
    if "temperature_rolling_zscore" in df.columns:
        max_z = df[["temperature_rolling_zscore", "humidity_rolling_zscore", "pressure_rolling_zscore"]].abs().max(axis=1)
    else:
        max_z = pd.Series(0.0, index=df.index)
        
    confirmed_model_anomaly = (
        df["model_is_anomaly"] & ((df["anomaly_score"] >= score_threshold) | (max_z > 2.5))
    )
    
    # Final combined decision
    df["final_is_anomaly"] = df["rule_is_anomaly"] | confirmed_model_anomaly
    
    # Classify anomaly type & identify affected sensor
    detected_types = []
    affected_sensors = []
    
    for _, row in df.iterrows():
        if not row["final_is_anomaly"]:
            detected_types.append("NORMAL")
            affected_sensors.append("NONE")
        elif row.get("flag_missing", False):
            detected_types.append("MISSING_DATA")
            affected_sensors.append("sensor_reading")
        elif row.get("flag_frozen", False):
            detected_types.append("FROZEN_SENSOR")
            aff = []
            if row.get("flag_temp_frozen", False): aff.append("temperature")
            if row.get("flag_hum_frozen", False): aff.append("humidity")
            if row.get("flag_press_frozen", False): aff.append("pressure")
            affected_sensors.append(", ".join(aff) if aff else "MULTIVARIATE")
        elif row.get("flag_temp_step", False):
            detected_types.append("TEMPERATURE_SPIKE")
            affected_sensors.append("temperature")
        elif row.get("flag_press_step", False):
            detected_types.append("PRESSURE_SPIKE")
            affected_sensors.append("pressure")
        elif row.get("flag_hum_step", False):
            detected_types.append("HUMIDITY_SPIKE")
            affected_sensors.append("humidity")
        elif row.get("flag_invalid_range", False):
            detected_types.append("OUT_OF_RANGE")
            aff = []
            if row.get("flag_temp_range", False): aff.append("temperature")
            if row.get("flag_hum_range", False): aff.append("humidity")
            if row.get("flag_press_range", False): aff.append("pressure")
            affected_sensors.append(", ".join(aff) if aff else "MULTIVARIATE")
        elif row.get("flag_data_gap", False):
            detected_types.append("DATA_GAP")
            affected_sensors.append("timestamp")
        elif confirmed_model_anomaly.loc[row.name]:
            t_rate = abs(row.get("temperature_rate", 0.0))
            p_rate = abs(row.get("pressure_rate", 0.0))
            t_z = abs(row.get("temperature_rolling_zscore", 0.0))
            p_z = abs(row.get("pressure_rolling_zscore", 0.0))
            
            if (t_rate > 0.05 or p_rate > 5.0) or (t_z > 2.0 or p_z > 2.0):
                detected_types.append("SENSOR_DRIFT")
                aff = []
                if t_z > p_z and t_z > 1.5: aff.append("temperature")
                elif p_z > 1.5: aff.append("pressure")
                affected_sensors.append(", ".join(aff) if aff else "MULTIVARIATE")
            else:
                detected_types.append("MULTIVARIATE_ANOMALY")
                affected_sensors.append("MULTIVARIATE")
        else:
            detected_types.append("UNKNOWN_ANOMALY")
            affected_sensors.append("MULTIVARIATE")
            
    df["detected_anomaly_type"] = detected_types
    df["affected_sensor"] = affected_sensors
    # Legacy compatibility fields
    df["is_anomaly"] = df["final_is_anomaly"]
    df["anomaly_type"] = df["detected_anomaly_type"]
    df["affected_parameter"] = df["affected_sensor"]
    return df


# Backwards compatibility aliases for Model V1 code and legacy scripts
SensorRuleConfig = RuleThresholdConfig
apply_rule_based_checks = apply_rule_checks
AWSAnomalyDetector = IsolationForestDetectorV2


def classify_anomalies(df: pd.DataFrame, config: RuleThresholdConfig = RuleThresholdConfig()) -> pd.DataFrame:
    """Compatibility wrapper that routes to synthesize_results_v2."""
    if "model_is_anomaly" not in df.columns:
        df["model_is_anomaly"] = df.get("is_model_anomaly", False)
    if "anomaly_score" not in df.columns:
        df["anomaly_score"] = 0.0
    res = synthesize_results_v2(df)
    res["is_anomaly"] = res["final_is_anomaly"]
    res["anomaly_type"] = res["detected_anomaly_type"]
    res["affected_parameter"] = res["affected_sensor"]
    return res

