from pathlib import Path

import mlflow
import pandas as pd
 
from src.data.preprocess import preprocess_data
from src.features.feature_builder import feature_builder

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURE_COLUMNS = [
    line.strip()
    for line in (BASE_DIR / "feature_columns.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
MODEL_DIR = BASE_DIR / "mlartifacts/2/models/m-080ba0968aad43a9a36458921d28661e/artifacts"
THRESHOLD = 0.25

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

try:
    model = mlflow.lightgbm.load_model(str(MODEL_DIR))
except Exception as exc:
    model = None
    print(f"Failed to load the model from {MODEL_DIR}: {exc}")


def _prepare_features(payload: dict) -> pd.DataFrame:
    raw_frame = pd.DataFrame([payload])
    raw_frame = preprocess_data(raw_frame)
    raw_frame = feature_builder(raw_frame)
    raw_frame = raw_frame.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return raw_frame


def predict(payload: dict) -> dict:
    if model is None:
        raise RuntimeError("Model is not available for inference.")

    feature_frame = _prepare_features(payload)
    probability = float(model.predict_proba(feature_frame)[0, 1])
    prediction = int(probability >= THRESHOLD)

    return {
        "prediction": int(prediction),
        "probability": round(probability, 4),
        "threshold": THRESHOLD,
        "risk_level": "likely to churn" if prediction == 1 else "likely to stay",
        "message": (
            "Customer is predicted to churn based on the provided attributes."
            if prediction == 1
            else "Customer is predicted to remain with the company."
        ),
    }