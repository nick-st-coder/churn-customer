from app.prediction.predict import _prepare_features, FEATURE_COLUMNS

def test_prepare_features_returns_numeric_schema_matching_model():
    sample = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 350.75,
    }

    prepared = _prepare_features(sample)

    assert set(FEATURE_COLUMNS).issubset(prepared.columns)
    assert prepared.select_dtypes(include=["object"]).empty
    assert prepared.reindex(columns=FEATURE_COLUMNS).shape[1] == len(FEATURE_COLUMNS)