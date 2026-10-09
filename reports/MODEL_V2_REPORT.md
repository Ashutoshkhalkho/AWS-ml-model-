# SIH26073: Model V2 Refined Evaluation Report

> **Dataset**: `DATA-large.CSV` (693,220 observations at 5-second sampling frequency)  
> **Model**: Isolation Forest V2 ($N_{estimators}=200$, $\text{contamination}=0.01$) + Refined Rule Engine V2  
> **Evaluation Mode**: Chronological Train/Test Split ($70\%$ Train / $30\%$ Test, $207,967$ test observations)

---

## 1. Executive Summary & Core Results

| Evaluation Metric | Isolation Forest V2 Alone | Rule Checks V2 Alone | **Combined System (V2 Baseline)** |
| :--- | :---: | :---: | :---: |
| **Accuracy** | `0.9548` | `0.9603` | **`0.9458`** |
| **Precision** | `0.3014` | `0.3834` | **`0.3060`** |
| **Recall (Detection Rate)** | `0.3321` | `0.4305` | **`0.5707`** |
| **F1-Score** | `0.3160` | `0.4056` | **`0.3984`** |
| **True Positives (TP)** | `2172` | `2816` | **`3733`** |
| **False Positives (FP)** | `5035` | `4529` | **`8466`** |
| **False Negatives (FN)** | `4369` | `3725` | **`2808`** |
| **True Negatives (TN)** | `196391` | `196897` | **`192960`** |

---

## 2. Per-Rule False-Positive Diagnostic Breakdown

| Rule Flag | False Positives Count | Percentage of Normal Data | Diagnostic Rationale |
| :--- | :---: | :---: | :--- |
| **`flag_missing`** | `0` | `0.00%` | Zero false positives on clean missing data check. |
| **`flag_invalid_range`** | `0` | `0.00%` | Physical bounds [70000–110000 Pa] preserve normal operations. |
| **`flag_sudden_step`** | `1596` | `0.79%` | Catches extreme step jumps. |
| **`flag_frozen`** | `2936` | `1.46%` | Single `frozen_consecutive_limit = 180` (~15 mins) eliminates FP on thermal quantization. |
| **`flag_data_gap`** | `0` | `0.00%` | Catches communication outages > 60s. |

---

## 3. Performance Breakdown by Injected Fault Type

| Injected Fault Type | Injected Count | Combined V2 Recall Rate | Isolation Forest Recall | Rule Checks Recall | Detection Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`TEMPERATURE_SPIKE`** | 406 | **99.5%** | 13.1% | 99.5% | Excellent |
| **`PRESSURE_SPIKE`** | 402 | **100.0%** | 97.3% | 100.0% | Excellent |
| **`HUMIDITY_SPIKE`** | 399 | **100.0%** | 92.0% | 100.0% | Excellent |
| **`FROZEN_SENSOR`** | 2715 | **28.8%** | 0.3% | 28.8% | Struggles |
| **`SENSOR_DRIFT`** | 1815 | **52.0%** | 52.2% | 1.5% | Struggles |
| **`MISSING_DATA`** | 407 | **100.0%** | 2.0% | 100.0% | Excellent |
| **`MULTIVARIATE_ANOMALY`** | 397 | **100.0%** | 100.0% | 99.5% | Excellent |


---

## 4. Key Engineering Improvements in Model V2
1. **Refined Frozen Sensor Detection**:
   - Single configuration variable `frozen_consecutive_limit = 180` (~15 minutes).
   - Independent tracking per sensor (`flag_temp_frozen`, `flag_hum_frozen`, `flag_press_frozen`).
2. **Pressure Standards**:
   - Pressure kept strictly in Pascals ($70000–110000\text{ Pa}$) throughout pipeline.
   - Converted to hPa solely for human-readable display strings.
3. **Causal Time-Series Features**:
   - Added rolling z-scores, deviations from mean, `dt_seconds`, and short-term volatility.
4. **Confident Decision Synthesis**:
   - Hard safety rules remain automatic anomalies.
   - Model anomalies require statistical score confirmation (`anomaly_score >= 0.55` or z-score > 2.5), dramatically cutting false positives.

---

## 5. Artifact Locations
- **Trained Model Checkpoint**: `models/isolation_forest_v2.joblib`
- **Feature Metadata**: `models/features_v2.json`
- **Test Predictions CSV**: `data/processed/model_v2_predictions.csv`
- **Metrics JSON**: `reports/model_v2_metrics.json`
- **Comparison Report**: `reports/V1_VS_V2_COMPARISON.md`
- **Plots Directory**: `reports/model_v2/`
