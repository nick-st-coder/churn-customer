import traceback
from pathlib import Path

import lightgbm
import mlflow
import mlflow.lightgbm
import pandas as pd

from src.data.preprocess import preprocess_data
from src.features.feature_builder import feature_builder

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
FEATURE_COLUMNS = [
    line.strip()
    for line in (BASE_DIR / "feature_columns.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]

THRESHOLD = 0.36
EXPECTED_MLFLOW_VERSION = "3.14.0"
EXPECTED_LIGHTGBM_VERSION = "4.6.0"


def _validate_runtime() -> None:
    mlflow_version = getattr(mlflow, "__version__", "unknown")
    lightgbm_version = getattr(lightgbm, "__version__", "unknown")

    if mlflow_version != EXPECTED_MLFLOW_VERSION or lightgbm_version != EXPECTED_LIGHTGBM_VERSION:
        raise RuntimeError(
            "Model artifact was saved with mlflow==3.14.0 and lightgbm==4.6.0. "
            f"The current environment is mlflow=={mlflow_version} and lightgbm=={lightgbm_version}. "
            "Recreate the virtual environment from the pinned dependency set in pyproject.toml or model/requirements.txt."
        )


try:
    _validate_runtime()
    model = mlflow.lightgbm.load_model(str(MODEL_DIR))
    print("Model loaded successfully")
except Exception:
    print("Failed to load the model")
    traceback.print_exc()
    model = None


def _prepare_features(payload: dict) -> pd.DataFrame:
    raw_frame = pd.DataFrame([payload])
    raw_frame = preprocess_data(raw_frame)
    raw_frame = feature_builder(raw_frame)
    raw_frame = raw_frame.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return raw_frame.astype(float)

def predict(payload: dict) -> dict:
    if model is None:
        raise RuntimeError("Model is not available for inference.")

    feature_frame = _prepare_features(payload)

    try:
        probability = float(model.predict_proba(feature_frame)[0, 1])
    except OSError as exc:
        raise RuntimeError(
            "LightGBM prediction crashed while scoring the model. This usually means the runtime ABI is incompatible "
            "with the serialized model artifact. Reinstall the project in the pinned environment from pyproject.toml "
            "or model/requirements.txt."
        ) from exc

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