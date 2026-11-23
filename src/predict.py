import joblib, json
from pathlib import Path
import pandas as pd
from src.preprocessing import add_features, FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "trained_model.pkl"

def load_model():
    return joblib.load(MODEL)

def predict_one(data: dict):
    df = pd.DataFrame([data])
    df = add_features(df)
    X = df[FEATURES]
    model = load_model()
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(model.predict(X)[0])
    if proba < 0.3:
        risk = "Low"
    elif proba < 0.7:
        risk = "Moderate"
    else:
        risk = "High"
    return {"prediction": pred, "probability": proba, "risk": risk}

def predict_batch(df):
    d = add_features(df)
    X = d[FEATURES]
    model = load_model()
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    out = df.copy()
    out["Failure_Probability"] = proba
    out["Predicted_Failure"] = pred
    out["Risk_Category"] = ["Low" if p < 0.3 else "Moderate" if p < 0.7 else "High" for p in proba]
    return out
