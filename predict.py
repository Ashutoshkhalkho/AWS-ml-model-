"""
Inference Script for AWS Weather Station Anomaly Detection.
Passes new CSV weather observations through pre-trained model & rule check pipeline.
"""

import argparse
import pathlib
import pandas as pd

from src.data_loader import load_weather_data
from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.anomaly_detection import (
    SensorRuleConfig,
    apply_rule_based_checks,
    AWSAnomalyDetector,
    classify_anomalies,
)
from src.explainability import explain_anomalies


def run_prediction(
    input_path: str = "data/sample/aws_weather_sample.csv",
    model_path: str = "models/isolation_forest_aws.joblib",
    output_path: str = "data/processed/predictions.csv",
):
    print("=" * 60)
    print(" AWS ANOMALY DETECTION INFERENCE PIPELINE")
    print("=" * 60)

    # 1. Check model exists
    m_path = pathlib.Path(model_path)
    if not m_path.exists():
        raise FileNotFoundError(
            f"Trained model checkpoint not found at: {m_path}.\n"
            f"Please run 'python train.py' first to train and save the model."
        )

    # 2. Load model
    print(f"\n[1/5] Loading trained model from {m_path}...")
    detector = AWSAnomalyDetector.load(m_path)

    # 3. Load & preprocess input CSV
    print(f"\n[2/5] Loading and preprocessing input dataset: {input_path}...")
    df_raw = load_weather_data(input_path)
    df_prep, prep_report = preprocess_data(df_raw)
    print(f"      Processed {len(df_prep)} observations across stations: {prep_report['stations_processed']}")

    # 4. Feature engineering & rule validation
    print("\n[3/5] Computing feature matrix & applying physical sensor rules...")
    df_feat, _ = engineer_features(df_prep, window_size=4)
    rule_config = SensorRuleConfig()
    df_rules = apply_rule_based_checks(df_feat, config=rule_config)

    # 5. Model inference & explanation
    print("\n[4/5] Executing Isolation Forest inference & generating explanations...")
    is_model_anomaly, scores, raw_preds = detector.predict(df_rules)
    df_rules["is_model_anomaly"] = is_model_anomaly
    df_rules["anomaly_score"] = scores
    df_rules["raw_prediction"] = raw_preds

    df_classified = classify_anomalies(df_rules, config=rule_config)
    df_explained = explain_anomalies(df_classified, config=rule_config)

    # 6. Save results
    print("\n[5/5] Exporting predictions...")
    out_path = pathlib.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_cols = [
        "station_id",
        "timestamp",
        "temperature",
        "pressure",
        "humidity",
        "anomaly_score",
        "is_anomaly",
        "anomaly_type",
        "affected_parameter",
        "explanation",
    ]
    
    df_explained[export_cols].to_csv(out_path, index=False)
    
    num_anomalies = df_explained["is_anomaly"].sum()
    print(f"      Predictions successfully saved to: {out_path.resolve()}")
    print(f"      Total records: {len(df_explained)} | Flagged anomalies: {num_anomalies}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict AWS Anomaly Detection")
    parser.add_argument("--input", type=str, default="data/sample/aws_weather_sample.csv", help="Input CSV path")
    parser.add_argument("--model", type=str, default="models/isolation_forest_aws.joblib", help="Model path")
    parser.add_argument("--output", type=str, default="data/processed/predictions.csv", help="Output predictions path")

    args = parser.parse_args()
    run_prediction(input_path=args.input, model_path=args.model, output_path=args.output)
