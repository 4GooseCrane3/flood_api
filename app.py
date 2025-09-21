from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load your joblib model (file is in the same folder)
model = joblib.load("hightide_model.joblib")

@app.get("/")
def read_root():
    return {"message": "Hightide model API is running"}

@app.get("/predict")
def predict(value: float):
    # Example: model expects 1D array
    prediction = model.predict(np.array([[value]]))
    return {"input": value, "prediction": prediction.tolist()}
