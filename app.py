from fastapi import FastAPI, Request
import joblib
import numpy as np

app = FastAPI()

# Load the model
model = joblib.load("flood_model.joblib")

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        features = np.array(data.get("features", []), dtype=np.float32).reshape(1, -1)
        pred = model.predict(features)
        return {"alert_type": "flood", "severity": float(pred[0])}
    except Exception as e:
        return {"error": str(e)}
