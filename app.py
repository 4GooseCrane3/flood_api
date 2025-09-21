from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load the scikit-learn model
model = joblib.load("app/hightide_model.joblib")

@app.get("/")
def home():
    return {"message": "Model API is running"}

@app.post("/predict")
def predict(features: list):
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    return {"prediction": prediction.tolist()}
