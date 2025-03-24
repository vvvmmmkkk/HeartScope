import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load trained model
model_path = r"C:\Users\vishnu mk\Project\ECG-Arrhythmia-Classifier\deployment\web_deployment\model.pkl"
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

# Expected Feature Names
expected_feature_names = [
    "0_pre-RR", "0_post-RR", "0_pPeak", "0_tPeak", "0_rPeak", "0_sPeak", "0_qPeak",
    "0_qrs_interval", "0_pq_interval", "0_qt_interval", "0_st_interval", "0_qrs_morph0",
    "0_qrs_morph1", "0_qrs_morph2", "0_qrs_morph3", "0_qrs_morph4",
    "1_pre-RR", "1_post-RR", "1_pPeak", "1_tPeak", "1_rPeak", "1_sPeak", "1_qPeak",
    "1_qrs_interval", "1_pq_interval", "1_qt_interval", "1_st_interval", "1_qrs_morph0",
    "1_qrs_morph1", "1_qrs_morph2", "1_qrs_morph3", "1_qrs_morph4"
]

# Streamlit App Configuration
st.set_page_config(page_title="HeartScope ", page_icon="🫀", layout="wide")

# App Header
st.markdown("<h1 style='text-align: center; color: #D63384;'>  HeartScope🫀 </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #5C5C5C;'>AI-powered ECG classification and health recommendations.</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar: File Upload
st.sidebar.subheader("📂 Upload ECG Data (Excel)")
uploaded_file = st.sidebar.file_uploader("Upload an Excel or CSV file", type=["xlsx", "csv"])

# Separate Block for File Input Prediction
with st.container():
    st.markdown("<hr>", unsafe_allow_html=True)  # Horizontal separator
    st.markdown("<h2 style='color: #0D6EFD; text-align: center;'>📁 File Input Prediction Result</h2>", unsafe_allow_html=True)

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)

            # Fix column name mismatches
            if set(df.columns) != set(expected_feature_names):
                df.columns = expected_feature_names  # Rename columns to match expected format

            # Ensure column count matches expected input format
            if df.shape[1] != len(expected_feature_names):
                st.error(f"Incorrect number of features. Expected {len(expected_feature_names)}, got {df.shape[1]}")
            else:
                # Make Predictions
                predictions = model.predict(df)
                df["Prediction"] = predictions
                df["Prediction Label"] = df["Prediction"].map(label_to_category)

                st.success("✅ Predictions Completed!")
                st.write(df[["Prediction", "Prediction Label"]])

                # Provide Download Option
                st.download_button(
                    label="📥 Download Predictions",
                    data=df.to_csv(index=False),
                    file_name="ecg_predictions.csv",
                    mime="text/csv"
                )

                # Graph Visualization
                st.subheader("📊 ECG Feature Visualization")

                sample_idx = st.slider("Select a sample index:", 0, len(df) - 1, 0)
                sample_data = df.iloc[sample_idx, :-2]

                plt.figure(figsize=(10, 5))
                plt.plot(sample_data.index, sample_data.values, marker='o', linestyle='-')
                pred_label = df.iloc[sample_idx]["Prediction Label"]
                plt.title(f"ECG Feature Trends for Sample {sample_idx} ({pred_label})")
                plt.xlabel("ECG Features")
                plt.ylabel("Feature Values")
                plt.xticks(rotation=90)
                plt.grid(True)
                st.pyplot(plt)

                # Display Medical Recommendations
                st.subheader("🩺 Medical Recommendation")
                st.info(recommendations.get(pred_label, "No recommendation available."))

        except Exception as e:
            st.error(f"Error processing file: {e}")

# Sidebar: Manual Input
st.sidebar.markdown("### 🛠 Enter ECG Data Manually")

def get_user_input():
    input_data = []
    lead_names = ["Lead II", "Lead V5"]
    
    for i, lead in enumerate(lead_names):
        st.sidebar.subheader(f"{lead} Parameters")
        input_data.extend([
            st.sidebar.number_input(f"{lead}: Pre-RR Interval (ms)", value=206.0),
            st.sidebar.number_input(f"{lead}: Post-RR Interval (ms)", value=243.0),
            st.sidebar.number_input(f"{lead}: P Peak Amplitude", value=0.0479),
            st.sidebar.number_input(f"{lead}: T Peak Amplitude", value=1.5416),
            st.sidebar.number_input(f"{lead}: R Peak Amplitude", value=1.5098),
            st.sidebar.number_input(f"{lead}: S Peak Amplitude", value=1.5098),
            st.sidebar.number_input(f"{lead}: Q Peak Amplitude", value=0.0111),
            st.sidebar.number_input(f"{lead}: QRS Interval (ms)", value=9),
            st.sidebar.number_input(f"{lead}: PQ Interval (ms)", value=3),
            st.sidebar.number_input(f"{lead}: QT Interval (ms)", value=13),
            st.sidebar.number_input(f"{lead}: ST Interval (ms)", value=1),
            st.sidebar.number_input(f"{lead}: QRS Morph 0", value=0.0111),
            st.sidebar.number_input(f"{lead}: QRS Morph 1", value=0.0131),
            st.sidebar.number_input(f"{lead}: QRS Morph 2", value=0.1677),
            st.sidebar.number_input(f"{lead}: QRS Morph 3", value=0.5834),
            st.sidebar.number_input(f"{lead}: QRS Morph 4", value=1.1195)
        ])
    
    return np.array([input_data])

# Get manual input from the user
sample_array = get_user_input()

# Separate Block for Manual Input Prediction Result
with st.container():
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #0D6EFD; text-align: center;'>✍️ Manual Input Prediction Result</h2>", unsafe_allow_html=True)

    if st.sidebar.button("🔮 Predict", use_container_width=True):
        if sample_array.shape[1] == 32:
            y_pred = model.predict(sample_array)
            decoded_label = label_to_category.get(y_pred[0], "Unknown")

            st.markdown(f"<h2 style='color: #198754 ; text-align: center;'>Predicted: {decoded_label}</h2>", unsafe_allow_html=True)
            st.subheader("🩺 Medical Recommendation")
            st.info(recommendations.get(decoded_label, "No recommendation available."))

            plt.figure(figsize=(10, 5))
            plt.plot(expected_feature_names, sample_array[0], marker='o', linestyle='-', color="blue")
            plt.xlabel("ECG Features")
            plt.xticks(rotation=90)
            plt.grid(True)
            st.pyplot(plt)

# Footer
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Impact&display=swap');

  


    </style>
    """,
    unsafe_allow_html=True
)
