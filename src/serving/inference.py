import mlflow
import os
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

try:
    model = mlflow.pyfunc.load_model(
        "models:/churn-lgbm/1"
    )
except Exception as e:
    print("Failed to load a model")    

def predict():
    return True    