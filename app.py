from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load your joblib model
model = joblib.load("hightide_model.joblib")

@app.get("/")
def home():
    return {"message": "Hightide Model API is running"}

@app.post("/predict")
def predict(features: list):
    try:
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
