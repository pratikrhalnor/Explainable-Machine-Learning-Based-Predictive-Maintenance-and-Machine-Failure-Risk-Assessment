import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "ai4i2020.csv"

def load_data(path=None):
    p = Path(path) if path else DATA_PATH
    df = pd.read_csv(p)
    return df
