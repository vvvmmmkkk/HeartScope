# --- START OF FULL FILE app.py ---

import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
import io 
import seaborn as sns 

# --- Model Loading ---
model_path = "model.pkl"
try:
    with open(model_path, 'rb') as f_in:
        model = pickle.load(f_in)
except FileNotFoundError:
    st.error(f"Error: Model file '{model_path}' not found. Please ensure it's in the same directory as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model '{model_path}': {e}")
    st.stop()

# Label Mapping 
label_to_category = {
    1: 'Normal beats',
    2: 'Supraventricular ectopic beats',
    3: 'Ventricular ectopic beats',
    4: 'Fusion beats',
    5: 'Unknown beats'
}

# Medical Recommendations based on Predicted Category
recommendations = {
    "Normal beats": "✅ Your ECG appears normal based on the provided features. Maintain a healthy lifestyle!",
    "Supraventricular ectopic beats": "⚠️ Irregular atrial activity detected based on features. Consult a cardiologist if symptoms persist.",
    "Ventricular ectopic beats": "❗ Possible ventricular arrhythmia detected based on features. Immediate medical attention is advised.",
    "Fusion beats": "⚠️ Mixed normal and abnormal beats detected based on features. Further evaluation recommended.",
    "Unknown beats": "❓ Unclassified pattern detected based on features. Consider a detailed ECG examination."
}

# Feature Names - Base names used for one lead's features
feature_names_single_lead = [
    "pre-RR", "post-RR", "pPeak", "tPeak", "rPeak", "sPeak", "qPeak",
    "qrs_interval", "pq_interval", "qt_interval", "st_interval",
    "qrs_morph0", "qrs_morph1", "qrs_morph2", "qrs_morph3", "qrs_morph4"
]

# Expected Feature Names required by the model (assuming 2 leads: 0 and 1)
expected_feature_names = [f"{i}_{feature}" for i in range(2) for feature in feature_names_single_lead]
N_EXPECTED_FEATURES = len(expected_feature_names) # Should be 32

# Potential Raw Signal Column Names (kept for robustness)
POTENTIAL_SIGNAL_COLUMNS = ['ecg_signal', 'signal', 'ecg', 'waveform', 'lead_ii', 'lead_v5']


# --- Streamlit App Configuration ---
st.set_page_config(page_title="HeartScope", page_icon="🫀", layout="wide")

# App Header
st.markdown("<h1 style='text-align: center; color: #D63384; font-size: 90px; font-weight: bold;'>HeartScope 🫀</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 30px;'>💡 ECG Based Arrhythmia Prediction System</p>", unsafe_allow_html=True)
st.markdown("---") # Horizontal line separator

# --- Helper Functions ---

def find_signal_column(df):
    """
    Tries to find a column potentially containing raw ECG signal data in a DataFrame.
    Checks against a predefined list of common names (case-insensitive).
    """
    if df is None:
        return None
    df_cols_lower = [str(col).lower() for col in df.columns] # Ensure column names are strings
    for potential_name in POTENTIAL_SIGNAL_COLUMNS:
        try:
            idx = df_cols_lower.index(potential_name.lower())
            if pd.api.types.is_numeric_dtype(df.iloc[:, idx]) and len(df.iloc[:, idx].dropna()) > 10:
                 return df.columns[idx]
        except ValueError:
            continue
    return None

# --- Plotting Functions ---

def plot_qrs_morphology(qrs_morph_points, r_peak_value, lead_name, title="QRS Morphology Representation"):
    """
    Plots the 5 QRS morphology feature points and the R peak amplitude feature value
    for a single beat, representing the shape of the QRS complex.
    """
    if qrs_morph_points is None or not isinstance(qrs_morph_points, (list, np.ndarray)) or len(qrs_morph_points) != 5:
        st.warning(f"Could not plot '{title}': Invalid or missing QRS morphology points.")
        return

    try:
        qrs_morph_points = np.asarray(qrs_morph_points, dtype=float)
        r_peak_value = float(r_peak_value)
    except (ValueError, TypeError) as e:
        st.warning(f"Could not plot '{title}': Morphology or R-peak values are not numeric ({e}).")
        return

    relative_time = np.arange(len(qrs_morph_points))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor("#0E1117")
    ax.grid(True, which='major', linestyle='-', linewidth=0.4, color='gray', alpha=0.7)

    ax.plot(relative_time, qrs_morph_points, color='lime', marker='o', linestyle='-', lw=1.5, label=f'{lead_name} QRS Morphology Points')
    r_peak_time_approx = (len(qrs_morph_points) - 1) / 2
    ax.plot(r_peak_time_approx, r_peak_value, color='red', marker='^', markersize=10, linestyle='None', label=f'{lead_name} R Peak Amp.')

    ax.set_title(title, fontsize=12, color='white', fontweight='bold')
    ax.set_xlabel("Relative Position within QRS Morphology Sample", color='white', fontsize=10)
    ax.set_ylabel("Amplitude / Voltage (mV)", color='white', fontsize=10)
    ax.tick_params(axis='x', colors='white', labelsize=9)
    ax.tick_params(axis='y', colors='white', labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color('white')
    ax.spines["bottom"].set_color('white')
    ax.set_xlim(-0.5, len(qrs_morph_points) - 0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Note: This plot shows 5 specific voltage points sampled from the QRS complex and the R-peak amplitude, based on the extracted features. It represents the QRS shape but is NOT a full, continuous ECG waveform.")


def plot_feature_trend(df, feature_col, title, y_label, color='cyan'):
    """
    Plots the trend of a specific feature column from the DataFrame against the beat index.
    """
    if df is None or feature_col not in df.columns:
        st.warning(f"Could not plot '{title}': DataFrame is missing or column '{feature_col}' not found.")
        return
    if len(df) < 2:
         st.info(f"Not enough data points (need at least 2) to plot trend for '{title}'.")
         return

    feature_data = pd.to_numeric(df[feature_col], errors='coerce')
    beat_index = df.index

    # Filter out NaNs that might result from coercion or exist in the data
    valid_indices = ~feature_data.isna()
    if not valid_indices.any():
         st.warning(f"No valid numeric data found in column '{feature_col}' to plot '{title}'.")
         return

    feature_data = feature_data[valid_indices]
    beat_index = beat_index[valid_indices]


    fig, ax = plt.subplots(figsize=(12, 4)) # Wider plot for trends
    ax.set_facecolor("#0E1117")
    ax.grid(True, which='major', linestyle='-', linewidth=0.4, color='gray', alpha=0.7)

    # Plot as line and markers
    ax.plot(beat_index, feature_data, color=color, marker='o', linestyle='-', markersize=4, linewidth=1.0, label=feature_col)

    ax.set_title(title, fontsize=14, color='white', fontweight='bold')
    ax.set_xlabel("Beat Index (Row Number)", color='white')
    ax.set_ylabel(y_label, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color('white')
    ax.spines["bottom"].set_color('white')

    # Optional: Add mean line
    mean_val = feature_data.mean()
    ax.axhline(mean_val, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.2f}')

    ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)


# --- FILE UPLOAD BLOCK ---
st.markdown("<h2 style='color: #17A2B8; text-align: center;'>📁 File Upload Prediction & Visualization</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload an ECG data file (.xlsx or .csv). "
    f"**Required:** {N_EXPECTED_FEATURES} feature columns (e.g., '0_pre-RR', '1_qrs_morph4', etc.).",
    type=["xlsx", "csv"],
    help=f"The file must contain columns named exactly: {', '.join(expected_feature_names)}"
)

if uploaded_file:
    df = None
    feature_df = None

    try:
        # --- 1. Read File ---
        file_name = uploaded_file.name
        if file_name.lower().endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif file_name.lower().endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                st.warning("UTF-8 decoding failed, trying latin1 encoding.")
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            st.error("Unsupported file type. Please upload .xlsx or .csv")
            st.stop()

        # --- 2. Prepare for Prediction ---
        df_cols = list(df.columns)
        feature_df_for_pred = df.copy() # Use a specific copy for prediction features

        missing_features = [col for col in expected_feature_names if col not in df_cols]

        if not missing_features:
             pass
        elif len(missing_features) == N_EXPECTED_FEATURES:
             if df.shape[1] >= N_EXPECTED_FEATURES:
                 st.warning(f"⚠️ Input file columns don't match expected names. Assuming the first {N_EXPECTED_FEATURES} columns are the features in the correct order.")
                 feature_df_for_pred = df.iloc[:, :N_EXPECTED_FEATURES].copy()
                 feature_df_for_pred.columns = expected_feature_names
             else:
                 st.error(f"❌ Incorrect number of columns. Expected {N_EXPECTED_FEATURES} feature columns, but file has {df.shape[1]}. Cannot perform prediction.")
                 feature_df_for_pred = None
        else:
             st.error(f"❌ Missing required feature columns for prediction: {', '.join(missing_features)}")
             feature_df_for_pred = None

        # --- 3. Perform Prediction ---
        if feature_df_for_pred is not None:
            try:
                # Ensure correct columns and order
                feature_df_for_pred = feature_df_for_pred[expected_feature_names]
            except KeyError as e:
                 st.error(f"❌ Error selecting expected feature columns: {e}. Check column names.")
                 feature_df_for_pred = None

        if feature_df_for_pred is not None:
            try:
                # Convert feature data to numeric, coercing errors
                feature_df_for_pred_numeric = feature_df_for_pred.apply(pd.to_numeric, errors='coerce')

                if feature_df_for_pred_numeric.isnull().any().any():
                    st.warning("⚠️ Some feature values were non-numeric and ignored (set to NaN). Predictions might be affected.")
                    # Consider filling NaNs if appropriate for the model
                    # feature_df_for_pred_numeric = feature_df_for_pred_numeric.fillna(feature_df_for_pred_numeric.median())

                predictions = model.predict(feature_df_for_pred_numeric)
                pred_proba = model.predict_proba(feature_df_for_pred_numeric)

                # Add results back to the *original* DataFrame 'df'
                df["Prediction"] = predictions
                df["Prediction_Label"] = df["Prediction"].map(label_to_category).fillna("Unknown Mapping")
                df["Confidence"] = [pred_proba[i, pred-1] if 0 < pred <= pred_proba.shape[1] else 0 for i, pred in enumerate(predictions)]

                st.success("✅ Predictions Completed!")
                columns_to_show = ["Prediction_Label", "Confidence"] + expected_feature_names[:2] + expected_feature_names[-2:]
                # Ensure columns actually exist in df before trying to display them
                columns_to_show_existing = [col for col in columns_to_show if col in df.columns]
                st.dataframe(df[columns_to_show_existing])

                # --- 4. Display Aggregate Results & Recommendations ---
                st.subheader("📊 Prediction Summary & Recommendations")
                if not df.empty and "Prediction_Label" in df.columns:
                    prediction_counts = df["Prediction_Label"].value_counts()
                    st.bar_chart(prediction_counts)
                    unique_predictions = df["Prediction_Label"].unique()
                    st.write("**Recommendations based on detected categories:**")
                    for label in unique_predictions:
                        if label in recommendations:
                            st.info(f"**{label}:** {recommendations[label]}")
                        elif label != "Unknown Mapping":
                            st.warning(f"**{label}:** No recommendation available.")
                else:
                    st.info("No predictions to summarize.")

                # --- 5. Visualize Feature Trends (NEW PLOTS) ---
                st.markdown("---")
                st.subheader("📈 Feature Trend Visualization")

                # Choose which lead's features to plot (e.g., Lead 0)
                lead_prefix = "0_" # Or make this selectable via st.radio/st.selectbox

                rr_col = f"{lead_prefix}pre-RR"
                qrs_col = f"{lead_prefix}qrs_interval"

                if rr_col in df.columns:
                     plot_feature_trend(df, rr_col, title=f"Trend of {rr_col} (Heart Rate Variability)", y_label="RR Interval (samples or ms)", color='yellow')
                else:
                     st.warning(f"Column '{rr_col}' not found in the data for trend plotting.")

                if qrs_col in df.columns:
                     plot_feature_trend(df, qrs_col, title=f"Trend of {qrs_col} (QRS Duration)", y_label="QRS Duration (samples or ms)", color='magenta')
                else:
                      st.warning(f"Column '{qrs_col}' not found in the data for trend plotting.")


                # --- 6. Visualize QRS Morphology from Features ---
                st.markdown("---")
                st.subheader("📈 QRS Morphology Visualization (Single Beat)")
                st.markdown("Select a beat (row index) from the uploaded file to visualize its QRS morphology points:")

                if not df.empty:
                    sample_idx = st.slider(
                        "Select Beat Index:",
                        min_value=0,
                        max_value=len(df) - 1,
                        value=0,
                        key="file_upload_slider"
                    )
                    selected_beat_data = df.iloc[sample_idx]
                    pred_label = selected_beat_data.get("Prediction_Label", "N/A")

                    st.write(f"Displaying morphology for Beat Index: **{sample_idx}** (Predicted: **{pred_label}**)")

                    col1, col2 = st.columns(2)

                    # Plot Lead 0 morphology
                    with col1:
                        lead0_morph_cols = [f"0_qrs_morph{i}" for i in range(5)]
                        lead0_rpeak_col = "0_rPeak"
                        if all(col in selected_beat_data.index for col in lead0_morph_cols + [lead0_rpeak_col]):
                            lead0_morph_points = selected_beat_data[lead0_morph_cols].values
                            lead0_rpeak_value = selected_beat_data[lead0_rpeak_col]
                            plot_qrs_morphology(lead0_morph_points, lead0_rpeak_value, "Lead 0", title=f"Lead 0 QRS Morphology (Beat {sample_idx})")
                        else:
                            st.warning("Could not find all required Lead 0 morphology/R-peak columns.")

                    # Plot Lead 1 morphology
                    with col2:
                        lead1_morph_cols = [f"1_qrs_morph{i}" for i in range(5)]
                        lead1_rpeak_col = "1_rPeak"
                        if all(col in selected_beat_data.index for col in lead1_morph_cols + [lead1_rpeak_col]):
                            lead1_morph_points = selected_beat_data[lead1_morph_cols].values
                            lead1_rpeak_value = selected_beat_data[lead1_rpeak_col]
                            plot_qrs_morphology(lead1_morph_points, lead1_rpeak_value, "Lead 1", title=f"Lead 1 QRS Morphology (Beat {sample_idx})")
                        else:
                            st.warning("Could not find all required Lead 1 morphology/R-peak columns.")
                else:
                    st.warning("No data available to visualize morphology.")

            except Exception as pred_err:
                 st.error(f"❌ An error occurred during prediction or visualization: {pred_err}")
                 st.exception(pred_err)

    except Exception as file_err:
        st.error(f"❌ Error processing file '{uploaded_file.name}': {file_err}")
        st.exception(file_err)


# --- MANUAL INPUT BLOCK (Sidebar) ---
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("### 🛠️ Enter ECG Features Manually")
st.sidebar.markdown("_Predict based on single beat features (2 Leads)_")

def get_user_input():
    """Collects feature values entered manually by the user in the sidebar."""
    input_data = []
    lead_names = ["Lead 0 (e.g., II)", "Lead 1 (e.g., V5)"]
    default_values = [
        0.85, 0.90, 0.20, 0.30, 1.10, -0.50, 0.05, 
        0.10, 0.12, 0.35, 0.08,                  
        0.40, 0.20, -0.10, -0.30, 0.50           
    ] * 2 

    if len(default_values) != N_EXPECTED_FEATURES:
         st.sidebar.error(f"Internal Error: Default values count ({len(default_values)}) mismatch.")
         return None

    with st.sidebar.expander("Expand to Enter Feature Values", expanded=False):
        feat_index = 0
        for i, lead in enumerate(lead_names):
            st.sidebar.markdown(f"**{lead} Features:**")
            for j, feature_base_name in enumerate(feature_names_single_lead):
                key = f"manual_{i}_{feature_base_name}"
                default_val = float(default_values[feat_index])
                value = st.number_input(
                    label=f"{feature_base_name}",
                    value=default_val, step=0.01, key=key, format="%.3f"
                )
                input_data.append(value)
                feat_index += 1

    if len(input_data) != N_EXPECTED_FEATURES:
         st.sidebar.error(f"Error collecting input: Expected {N_EXPECTED_FEATURES}, got {len(input_data)}.")
         return None

    return np.array([input_data])

manual_feature_array = get_user_input()

if st.sidebar.button("🔮 Predict Manually", use_container_width=True, type="primary"):
    if manual_feature_array is not None and manual_feature_array.shape == (1, N_EXPECTED_FEATURES):
        try:
            y_pred = model.predict(manual_feature_array)
            pred_proba = model.predict_proba(manual_feature_array)
            predicted_label_num = y_pred[0]
            decoded_label = label_to_category.get(predicted_label_num, "Unknown Mapping")
            confidence = 0.0
            if 0 < predicted_label_num <= pred_proba.shape[1]:
                 confidence = pred_proba[0, predicted_label_num - 1]

            st.markdown("<h2 style='color: #17A2B8; text-align: center;'>📋 Manual Input Prediction Result</h2>", unsafe_allow_html=True)
            st.success(f"🔍 Predicted Label: **{decoded_label}** (Confidence: {confidence:.2f})")

            st.subheader("📈 Representative QRS Morphology (Manual Input)")
            col1, col2 = st.columns(2)
            num_features_per_lead = len(feature_names_single_lead)

            with col1: # Lead 0 Manual Morphology
                 try:
                     lead0_morph_indices = [feature_names_single_lead.index(f"qrs_morph{i}") for i in range(5)]
                     lead0_rpeak_index = feature_names_single_lead.index("rPeak")
                     lead0_morph_points = manual_feature_array[0, lead0_morph_indices]
                     lead0_rpeak_value = manual_feature_array[0, lead0_rpeak_index]
                     plot_qrs_morphology(lead0_morph_points, lead0_rpeak_value, "Lead 0 (Manual)", title="Lead 0 QRS Morphology (Manual)")
                 except (IndexError, ValueError) as e:
                      st.warning(f"Could not plot Lead 0 morphology (Manual): {e}")

            with col2: # Lead 1 Manual Morphology
                 try:
                     lead1_morph_indices = [num_features_per_lead + feature_names_single_lead.index(f"qrs_morph{i}") for i in range(5)]
                     lead1_rpeak_index = num_features_per_lead + feature_names_single_lead.index("rPeak")
                     lead1_morph_points = manual_feature_array[0, lead1_morph_indices]
                     lead1_rpeak_value = manual_feature_array[0, lead1_rpeak_index]
                     plot_qrs_morphology(lead1_morph_points, lead1_rpeak_value, "Lead 1 (Manual)", title="Lead 1 QRS Morphology (Manual)")
                 except (IndexError, ValueError) as e:
                      st.warning(f"Could not plot Lead 1 morphology (Manual): {e}")

            st.subheader("🩺 Medical Recommendation (Based on Features)")
            st.info(recommendations.get(decoded_label, "No recommendation available."))

        except Exception as e:
             st.error(f"❌ An error occurred during manual prediction: {e}")
             st.exception(e)

    elif manual_feature_array is not None:
         st.error(f"Data shape mismatch. Expected (1, {N_EXPECTED_FEATURES}), got {manual_feature_array.shape}")


# --- Footer ---
st.markdown("---")
st.markdown("""
    <div style="text-align: center; background-color: #0E1117; color: #FAFAFA; padding: 20px; border-radius: 10px; border: 1px solid #262730; margin-top: 40px;">
        <h3 style='color: #17A2B8'>🫀 About HeartScope</h3>
        <p style='font-size: 16px; color: #A0A0A0;'>
            HeartScope utilizes a machine learning model trained on ECG features to predict potential arrhythmia categories.
            It analyzes features extracted from ECG signals (like RR intervals, peak amplitudes, morphology points, etc.) to classify heartbeats.
            The goal is to provide a preliminary assessment and highlight potential cardiac abnormalities.
            <br><br>
            Users can upload data containing pre-extracted features for batch prediction or manually enter features for a single beat prediction.
            The application visualizes key QRS morphology points for selected beats and shows trends for important features like RR intervals and QRS duration based on the provided data.
        </p>
    </div>
    <br>
""", unsafe_allow_html=True)

st.markdown("<br><p style='text-align: center; color: gray; font-size: 14px;'>Created with Streamlit & Scikit-learn.<br>⚠️ **Disclaimer:** This tool is for informational purposes only and is not a substitute for professional medical diagnosis or advice. Always consult with a qualified healthcare professional for any health concerns.</p>", unsafe_allow_html=True)
