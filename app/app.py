import streamlit as st
import pandas as pd, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predict import predict_one, predict_batch, load_model
from src.data_loader import load_data

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
page = st.sidebar.selectbox("Page", ["Home","Overview","Data Analysis","Failure Prediction","Risk Assessment","Model Performance","About"])

if page == "Home":
    st.title("Predictive Maintenance ML")
    df = load_data()
    st.write("Records:", len(df))
    st.write("Failures:", int(df["Machine failure"].sum()))
    st.bar_chart(df["Machine failure"].value_counts())
    st.subheader("Batch Prediction")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        bdf = pd.read_csv(up)
        out = predict_batch(bdf)
        st.dataframe(out)
        st.download_button("Download", out.to_csv(index=False), "predictions.csv")

elif page == "Failure Prediction":
    st.title("Failure Prediction")
    mtype = st.selectbox("Type", ["L","M","H"])
    air = st.number_input("Air temp K", value=300.0)
    proc = st.number_input("Process temp K", value=310.0)
    speed = st.number_input("Rotational speed rpm", value=1500.0)
    torque = st.number_input("Torque Nm", value=40.0)
    wear = st.number_input("Tool wear min", value=100.0)
    if st.button("Predict"):
        data = {"Type": mtype, "Air temperature [K]": air, "Process temperature [K]": proc,
                "Rotational speed [rpm]": speed, "Torque [Nm]": torque, "Tool wear [min]": wear}
        res = predict_one(data)
        st.write(res)

elif page == "Risk Assessment":
    st.title("Risk Assessment")
    mtype = st.selectbox("Type", ["L","M","H"])
    air = st.number_input("Air temp K", value=300.0, key="a2")
    proc = st.number_input("Process temp K", value=310.0, key="p2")
    speed = st.number_input("Rotational speed rpm", value=1500.0, key="s2")
    torque = st.number_input("Torque Nm", value=40.0, key="t2")
    wear = st.number_input("Tool wear min", value=100.0, key="w2")
    if st.button("Assess"):
        data = {"Type": mtype, "Air temperature [K]": air, "Process temperature [K]": proc,
                "Rotational speed [rpm]": speed, "Torque [Nm]": torque, "Tool wear [min]": wear}
        res = predict_one(data)
        st.metric("Failure Probability", f"{res['probability']:.3f}")
        st.write("Risk:", res["risk"])

elif page == "Model Performance":
    st.title("Model Performance")
    p = Path(__file__).resolve().parents[1] / "reports" / "evaluation" / "model_comparison.json"
    if p.exists():
        d = json.loads(p.read_text())
        st.write("Best:", d["best_model"])
        st.json(d["results"])

elif page == "Data Analysis":
    st.title("Data Analysis")
    df = load_data()
    st.dataframe(df.head())
    st.bar_chart(df["Type"].value_counts())

else:
    st.title(page)
    st.write("Project documentation.")

# batch upload in Home page appended
