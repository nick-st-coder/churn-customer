from fastapi import FastAPI
import gradio as gr
from pydentic import BaseModel

app = FastAPI(
    title='Churn Customer Prediction'
)

@app.get("/")
def root():
    return {"status": "ok"}

class Customer(BaseModel):
    gender: str                
    Partner: str              
    Dependents: str           
    
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
    
    tenure: int               
    MonthlyCharges: float      
    TotalCharges: float 

# @app.post("/predict")
# def get_prediction(data: Customer):
#     try:
#         result = predict(data.dict())    