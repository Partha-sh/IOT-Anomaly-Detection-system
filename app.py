import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

import tensorflow as tf

st.set_page_config(
    page_title="Industrial IoT Anomaly Detection",
    page_icon="🏭",
    layout="centered"
)

# Load assets relative to this file so the app works from any launch directory.
BASE_DIR = Path(__file__).resolve().parent

model = tf.keras.models.load_model(
    BASE_DIR / "autoencoder_model.keras"
)
scaler = joblib.load(BASE_DIR / "scaler.pkl")
threshold = float(np.asarray(joblib.load(BASE_DIR / "threshold.pkl")).ravel()[0])

st.title("Industrial IoT Anomaly Detection System")

st.write(
    "Enter machine sensor readings to detect anomalies."
)

# Inputs
air_temp = st.number_input(
    "Air Temperature [K]",
    value=300.0
)

process_temp = st.number_input(
    "Process Temperature [K]",
    value=310.0
)

rpm = st.number_input(
    "Rotational Speed [rpm]",
    value=1500
)

torque = st.number_input(
    "Torque [Nm]",
    value=40.0
)

tool_wear = st.number_input(
    "Tool Wear [min]",
    value=100
)

if st.button("Analyze"):

    sample = pd.DataFrame({
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rpm],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    sample_scaled = scaler.transform(sample)

    reconstruction = model.predict(
        sample_scaled,
        verbose=0
    )

    error = np.mean(
        np.square(
            sample_scaled - reconstruction
        )
    )

    st.write(f"Reconstruction Error: {error:.4f}")

    if error > threshold:
        st.error("⚠️ Anomaly Detected")
    else:
        st.success("✅ Normal Machine")
