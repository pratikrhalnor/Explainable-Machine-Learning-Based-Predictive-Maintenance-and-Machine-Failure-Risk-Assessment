import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

NUMERIC = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
CATEGORICAL = ["Type"]
TARGET = "Machine failure"

def add_features(df):
    d = df.copy()
    d["Temp_diff"] = d["Process temperature [K]"] - d["Air temperature [K]"]
    d["Power"] = d["Rotational speed [rpm]"] * d["Torque [Nm]"]
    d["Wear_rate"] = d["Tool wear [min]"] / (d["Rotational speed [rpm]"] + 1)
    return d

FEATURES = NUMERIC + CATEGORICAL + ["Temp_diff", "Power", "Wear_rate"]

def get_preprocessor():
    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC + ["Temp_diff", "Power", "Wear_rate"]),
        ("cat", cat_pipe, CATEGORICAL),
    ])

def split_data(df, test_size=0.2, random_state=42):
    d = add_features(df)
    X = d[FEATURES]
    y = d[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
