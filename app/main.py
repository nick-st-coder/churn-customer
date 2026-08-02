from fastapi import FastAPI
import gradio as gr
from pydantic import BaseModel
from app.prediction.predict import predict

app = FastAPI(title="Churn Customer Prediction")

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = 0
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def get_prediction(data: Customer):
    try:
        result = predict(data.model_dump())
        return {"prediction": result}
    except Exception as exc:
        return {"error": str(exc)}


def _gradio_predict(
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges,
):
    payload = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }
    return predict(payload)


with gr.Blocks(title="Churn Customer Prediction") as demo:
    gr.Markdown("## Churn customer prediction")
    with gr.Row():
        gender = gr.Dropdown(["Male", "Female"], value="Male", label="Gender")
        senior_citizen = gr.Dropdown([0, 1], value=0, label="SeniorCitizen")
        partner = gr.Dropdown(["Yes", "No"], value="Yes", label="Partner")
        dependents = gr.Dropdown(["Yes", "No"], value="No", label="Dependents")
    with gr.Row():
        tenure = gr.Number(value=5, precision=0, label="Tenure")
        phone_service = gr.Dropdown(["Yes", "No"], value="Yes", label="PhoneService")
        multiple_lines = gr.Dropdown(["No", "Yes", "No phone service"], value="No", label="MultipleLines")
        internet_service = gr.Dropdown(["DSL", "Fiber optic", "No"], value="DSL", label="InternetService")
    with gr.Row():
        online_security = gr.Dropdown(["Yes", "No", "No internet service"], value="No", label="OnlineSecurity")
        online_backup = gr.Dropdown(["Yes", "No", "No internet service"], value="No", label="OnlineBackup")
        device_protection = gr.Dropdown(["Yes", "No", "No internet service"], value="No", label="DeviceProtection")
        tech_support = gr.Dropdown(["Yes", "No", "No internet service"], value="No", label="TechSupport")
    with gr.Row():
        streaming_tv = gr.Dropdown(["Yes", "No", "No internet service"], value="No", label="StreamingTV")
        streaming_movies = gr.Dropdown(["Yes", "No", "No internet service"], value="No", label="StreamingMovies")
        contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], value="Month-to-month", label="Contract")
        paperless_billing = gr.Dropdown(["Yes", "No"], value="Yes", label="PaperlessBilling")
    with gr.Row():
        payment_method = gr.Dropdown([
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ], value="Electronic check", label="PaymentMethod")
        monthly_charges = gr.Number(value=70.35, precision=2, label="MonthlyCharges")
        total_charges = gr.Number(value=350.75, precision=2, label="TotalCharges")
    predict_button = gr.Button("Predict churn")
    output = gr.JSON(label="Prediction")
    predict_button.click(
        fn=_gradio_predict,
        inputs=[
            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            monthly_charges,
            total_charges,
        ],
        outputs=output,
    )

app = gr.mount_gradio_app(app, demo, path="/gradio")