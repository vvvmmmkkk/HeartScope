import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline

# Load trained model
model_path = "model.pkl"
with open(model_path, 'rb') as f_in:
    model = pickle.load(f_in)

# Label Mapping
label_to_category = {
    1: 'Normal beats',
    2: 'Supraventricular ectopic beats',
    3: 'Ventricular ectopic beats',
    4: 'Fusion beats',
    5: 'Unknown beats'
}

# Medical Recommendations
recommendations = {
    "Normal beats": "✅ Your ECG appears normal. Maintain a healthy lifestyle!",
    "Supraventricular ectopic beats": "⚠️ Irregular atrial activity detected. Consult a cardiologist if symptoms persist.",
    "Ventricular ectopic beats": "❗ Possible ventricular arrhythmia detected. Immediate medical attention is advised.",
    "Fusion beats": "⚠️ Mixed normal and abnormal beats. Further evaluation recommended.",
    "Unknown beats": "❓ Unclassified pattern. Consider a detailed ECG examination."
}

# Feature Names
feature_names = ["pre-RR", "post-RR", "pPeak", "tPeak", "rPeak", "sPeak", "qPeak",
                 "qrs_interval", "pq_interval", "qt_interval", "st_interval", "qrs_morph0",
                 "qrs_morph1", "qrs_morph2", "qrs_morph3", "qrs_morph4"]

expected_feature_names = [f"{i}_{feature}" for i in range(2) for feature in feature_names]

# Streamlit App Configuration
st.set_page_config(page_title="HeartScope", page_icon="🫀", layout="wide")

# App Header
st.markdown("<h1 style='text-align: center; color: #D63384; font-size: 90px; font-weight: bold;'>HeartScope 🫀</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 30px;'>💡 ECG Based Arrhythmia Prediction System</p>", unsafe_allow_html=True)

# 📊 ECG PLOTTING FUNCTION
def plot_ecg_waveform(ecg_signal, title="ECG Waveform", color="cyan"):
    time = np.linspace(0, len(ecg_signal) / 250, len(ecg_signal))
    cs = CubicSpline(time, ecg_signal)
    time_smooth = np.linspace(time.min(), time.max(), 1000)
    ecg_smooth = cs(time_smooth)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_facecolor("#000")
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='red', alpha=0.6)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, color='red', alpha=0.5)
    ax.minorticks_on()

    ax.plot(time_smooth, ecg_smooth, color=color, lw=2)
    ax.scatter(time, ecg_signal, color='yellow', marker='o', s=10, label="Measured Points")

    ax.set_title(title, fontsize=14, color='white', fontweight='bold')
    ax.set_xlabel("Time (seconds)", color='white')
    ax.set_ylabel("Voltage (mV)", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend()
    st.pyplot(fig)

# 📁 FILE UPLOAD BLOCK
st.markdown("<h2 style='color: #0D6EFD; text-align: center;'>📁 File Upload Prediction</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an ECG data file (.xlsx or .csv)", type=["xlsx", "csv"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)

        if set(df.columns) != set(expected_feature_names):
            df.columns = expected_feature_names

        if df.shape[1] != len(expected_feature_names):
            st.error(f"Incorrect number of features. Expected {len(expected_feature_names)}, got {df.shape[1]}")
        else:
            predictions = model.predict(df)
            df["Prediction"] = predictions
            df["Prediction Label"] = df["Prediction"].map(label_to_category)

            st.success("✅ Predictions Completed!")
            st.write(df[["Prediction", "Prediction Label"]])

            # Graph Visualization for File Input
            st.subheader("📊 ECG Feature Visualization (File Input)")

            sample_idx = st.slider("Select a sample index:", 0, len(df) - 1, 0)
            sample_data = df.iloc[sample_idx, :-2].astype(float)
            pred_label = df.iloc[sample_idx]["Prediction Label"]

            plot_ecg_waveform(sample_data, f"ECG Feature Trends for Sample {sample_idx} ({pred_label})")

            # Medical Recommendations
            st.subheader("🩺 Medical Recommendation")
            st.info(recommendations.get(pred_label, "No recommendation available."))

    except Exception as e:
        st.error(f"Error processing file: {e}")

# 📋 MANUAL INPUT BLOCK
st.markdown("<h2 style='color: #0D6EFD; text-align: center;'>📋 Manual Input Prediction</h2>", unsafe_allow_html=True)

st.sidebar.markdown("### 🛠 Enter ECG Data Manually")

def get_user_input():
    input_data = []
    lead_names = ["Lead II", "Lead V5"]
    default_values = [0.85, 0.9, 0.2, 0.3, 1.1, -0.5, 0.6, 0.1, 0.12, 0.35, 0.08, 0.4, 0.2, -0.1, -0.3, 0.5]

    for i, lead in enumerate(lead_names):
        st.sidebar.subheader(f"📊 {lead} Parameters")
        for j, feature in enumerate(feature_names):
            value = st.sidebar.number_input(f"{lead}: {feature}", value=default_values[j % len(default_values)], step=0.01)
            input_data.append(value)

    return np.array([input_data])

sample_array = get_user_input()

if st.sidebar.button("🔮 Predict", use_container_width=True):
    if sample_array.shape[1] == 32:
        y_pred = model.predict(sample_array)
        decoded_label = label_to_category.get(y_pred[0], "Unknown")

        st.success(f"🔍 Prediction: **{decoded_label}**")

        st.subheader("📊 ECG Waveform (Manual Input)")
        plot_ecg_waveform(sample_array[0], "ECG Waveform (Manual Input)", color="blue")

        st.subheader("🩺 Medical Recommendation")
        st.info(recommendations.get(decoded_label, "No recommendation available."))

# Footer
st.markdown("""<br><br><br>
    <div style="text-align: center; background-color: black; color: white; padding: 20px; border-radius: 10px;">
        <h2 style='color: blue'>🫀About HeartScope</h2>
        <p style='font-size: 18px;'>
            HeartScope is a machine learning-based ECG arrhythmia prediction system designed to analyze
            electrocardiogram (ECG) data and classify heartbeats into different categories. The goal is to provide an
            early warning system for cardiac abnormalities and assist healthcare professionals in detecting arrhythmias.
            By leveraging advanced algorithms, HeartScope can efficiently analyze ECG signals, identify irregularities,
            and offer medical insights to support timely diagnosis. 
            The system is designed to be user-friendly, allowing both clinicians and individuals to upload ECG data for quick and accurate analysis.❤️With a focus on accessibility and precision, HeartScope aims to bridge the gap between technology and healthcare, enhancing early detection and proactive cardiac care.
        </p>
    </div>
    <br>
""", unsafe_allow_html=True)

st.markdown("<br><br><p style='text-align: center; color: gray;'>✅ Created with Streamlit.<br>⚠️ Not a substitute for medical diagnosis.</p>", unsafe_allow_html=True)
