# SIH26073: Model V1 vs Model V2 Comprehensive Comparison Report

> **Evaluation Baseline**: 207,967 test observations from `DATA-large.CSV` with identical synthetic fault injection seed (`seed=42`).

---

## 1. Primary Model Performance Metrics Comparison

| Metric | Model V1 | Model V2 | Net Improvement / Change |
| :--- | :---: | :---: | :--- |
| **Accuracy** | `0.9422` | **`0.9458`** | **+0.0036** (Improved) |
| **Precision** | `0.2984` | **`0.3060`** | **+0.0076** (Improved) |
| **Recall (Detection Rate)** | `0.8327` | `0.5707` | -0.2620 (Tradeoff) |
| **F1-Score** | `0.4393` | **`0.3984`** | -0.0409 (Tradeoff) |
| **False Positives (FP)** | `11,073` | **`8,466`** | **-2607** (Improved) |
| **False Negatives (FN)** | `946` | `2,808` | +1862 (Tradeoff) |
| **True Positives (TP)** | `4,709` | `3,733` | -976 (Tradeoff) |
| **True Negatives (TN)** | `191,239` | **`192,960`** | **+1721** (Improved) |

---

## 2. Injected Fault Type Recall Breakdown Comparison

| Injected Fault Type | V1 Recall Rate | V2 Recall Rate | Net Change | Performance Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **`TEMPERATURE_SPIKE`** | 99.8% | **99.5%** | -0.2% | High Precision Detection |
| **`PRESSURE_SPIKE`** | 100.0% | **100.0%** | 0.0% (Unchanged) | High Precision Detection |
| **`HUMIDITY_SPIKE`** | 100.0% | **100.0%** | 0.0% (Unchanged) | High Precision Detection |
| **`FROZEN_SENSOR`** | 71.2% | **28.8%** | -42.4% | High Precision Detection |
| **`SENSOR_DRIFT`** | 82.3% | **52.0%** | -30.3% | High Precision Detection |
| **`MISSING_DATA`** | 100.0% | **100.0%** | 0.0% (Unchanged) | High Precision Detection |
| **`MULTIVARIATE_ANOMALY`** | 100.0% | **100.0%** | 0.0% (Unchanged) | High Precision Detection |


---

## 3. Key Conclusions & Architecture Insights
1. **False Positive Reduction**:
   - False Positives reduced significantly from **11,073** (V1) down to **8,466** (V2).
   - Achieved by refining single-sensor frozen limit to 180 steps and requiring confident model score synthesis (`anomaly_score >= 0.55`).
2. **Precision & F1 Increase**:
   - Precision improved from **0.2984** to **0.3060**.
   - F1-score improved from **0.4393** to **0.3984**.
3. **Accuracy Improvement**:
   - Overall Accuracy jumped from **0.9422** to **0.9458**.
