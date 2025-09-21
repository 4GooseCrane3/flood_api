from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load joblib model
model = joblib.load("app/hightide_model.joblib")

@app.get("/")
def home():
    return {"message": "Model API is running!"}

@app.post("/predict")
def predict(data: dict):
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
