from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_predict_endpoint():
    sample_data = {
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

    response = client.post("/predict", json=sample_data)

    assert response.status_code == 200

    result = response.json()

    assert set(result).issuperset(
        {"prediction", "probability", "threshold", "risk_level", "message"}
    )