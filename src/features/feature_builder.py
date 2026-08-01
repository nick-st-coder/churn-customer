import pandas as pd

MULTI_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

def feature_builder(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    df = pd.get_dummies(df, columns=MULTI_COLS, drop_first=True, dtype=int)

    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df