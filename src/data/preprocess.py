import pandas as pd

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    for id_col in ["customerID", "CustomerID", "customer_id"]:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df