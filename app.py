from fastapi import FastAPI, Request
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("flood_model.joblib")

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    features = np.array(data.get("features", []), dtype=np.float32).reshape(1, -1)
    pred = model.predict(features)
    return {"alert_type": "flood", "severity": float(pred[0])}

