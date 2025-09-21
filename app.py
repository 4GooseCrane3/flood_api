# app.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="Flood API")

MODEL_PATH = "flood_model.joblib"

# Load model at startup
try:
    model = joblib.load(MODEL_PATH)
    load_error = None
except Exception as e:
    model = None
    load_error = str(e)

# Useful model metadata
N_FEATURES = getattr(model, "n_features_in_", None)
FEATURE_NAMES = getattr(model, "feature_names_in_", None)  # may be None

@app.get("/")
async def root():
    return {"message": "Flood API is running"}

@app.post("/predict")
async def predict(request: Request):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded", "details": load_error})

    # parse body
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON", "details": str(e)})

    if "features" not in body:
        return JSONResponse(status_code=400, content={"error": "Missing 'features' in JSON body"})

    features = body["features"]

    # If features is dict and model has feature names, construct ordered array
    if isinstance(features, dict):
        if FEATURE_NAMES is None:
            return JSONResponse(status_code=400, content={"error": "Model expects unnamed array; please send features as list"})
        missing = [n for n in FEATURE_NAMES if n not in features]
        if missing:
            return JSONResponse(status_code=400, content={"error": "Missing feature names", "missing": missing})
        try:
            arr = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32).reshape(1, -1)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": "Invalid feature values", "details": str(e)})

    # If features is list/tuple/ndarray
    elif isinstance(features, (list, tuple, np.ndarray)):
        arr = np.array(features, dtype=np.float32)
        # If one-dimensional list, validate length and reshape
        if arr.ndim == 1:
            if N_FEATURES is not None and arr.size != N_FEATURES:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid input size. Model expects {N_FEATURES} features, got {arr.size}."}
                )
            arr = arr.reshape(1, -1)
        # If 2D already, accept as is (but still check width)
        elif arr.ndim == 2:
            if N_FEATURES is not None and arr.shape[1] != N_FEATURES:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid input width. Model expects {N_FEATURES} features per sample, got {arr.shape[1]}."}
                )
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported features array shape"})

    else:
        return JSONResponse(status_code=400, content={"error": "features must be a list or a dict"})

    # Perform prediction
    try:
        pred = model.predict(arr)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Prediction failed", "details": str(e)})

    return {"alert_type": "flood", "severity": float(pred[0])}

# Allow running with `python app.py` locally (optional)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
