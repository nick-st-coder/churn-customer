import pandas as pd

BINARY_TARGET_MAP = {"No": 0, "Yes": 1}
BINARY_INPUT_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col, mapping in BINARY_INPUT_MAP.items():
        if col in df.columns:
            df[col] = (
                df[col].astype("string").str.strip()
                .map(mapping).fillna(0).astype(int)
            )

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df