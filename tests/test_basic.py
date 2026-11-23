from src.data_loader import load_data
from src.preprocessing import split_data, add_features
from src.predict import predict_one, predict_batch
import pandas as pd

def test_load():
    df = load_data()
    assert len(df) == 10000

def test_split():
    df = load_data()
    Xtr, Xte, ytr, yte = split_data(df)
    assert len(Xtr) > 0

def test_predict_one():
    data = {"Type":"L","Air temperature [K]":298.0,"Process temperature [K]":308.0,
            "Rotational speed [rpm]":1500.0,"Torque [Nm]":40.0,"Tool wear [min]":100.0}
    r = predict_one(data)
    assert "probability" in r
    assert r["risk"] in ["Low","Moderate","High"]

def test_batch():
    df = load_data().head(5)
    out = predict_batch(df)
    assert "Risk_Category" in out.columns
