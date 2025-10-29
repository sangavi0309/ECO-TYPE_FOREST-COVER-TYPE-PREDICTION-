# Eco_forest type streamlit app code 
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- CHANGE THESE PATHS if your files are elsewhere ---
MODEL_PATH = r"C:\Users\Sangavi A\Downloads\best_forest_model_tuned.pkl"
SCALER_PATH = r"C:\Users\Sangavi A\Downloads\scaler.pkl"
# optional: columns file if you created it in notebook
COLUMNS_PKL = r"C:\Users\Sangavi A\Downloads\model_columns.pkl"

# Load model & scaler (with friendly errors)
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Cannot load model at {MODEL_PATH}. Check path. Error: {e}")
    st.stop()

try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error(f"Cannot load scaler at {SCALER_PATH}. Check path. Error: {e}")
    st.stop()

st.title("🌲 EcoType – Forest Cover Prediction")
st.write("Enter values and press Predict. The app will fill any missing features automatically.")

# --- Basic user inputs (main measurable features) ---
elevation = st.number_input("Elevation (m)", 0, 5000, 2000)
aspect = st.number_input("Aspect (0–360)", 0, 360, 180)
slope = st.number_input("Slope (0–90)", 0, 90, 10)
hd_hydro = st.number_input("Horizontal Distance To Hydrology (m)", 0, 1000, 100)
vd_hydro = st.number_input("Vertical Distance To Hydrology (m)", -500, 500, 30)
hd_road = st.number_input("Horizontal Distance To Roadways (m)", 0, 7000, 1000)
hd_fire = st.number_input("Horizontal Distance To Fire Points (m)", 0, 7000, 1000)
hs_9am = st.number_input("Hillshade 9am (0–255)", 0, 255, 200)
hs_noon = st.number_input("Hillshade Noon (0–255)", 0, 255, 220)
hs_3pm = st.number_input("Hillshade 3pm (0–255)", 0, 255, 180)

# derived
hs_diff_noon_3pm = hs_noon - hs_3pm
hs_diff_noon_9am = hs_noon - hs_9am
road_hydro_ratio = hd_road / (hd_hydro + 1)

# wilderness selection (one-hot)
wilderness_choice = st.selectbox("Wilderness Area", ["Area 1", "Area 2", "Area 3", "Area 4"])
wilderness_values = {
    "Wilderness_Area_1": 1 if wilderness_choice == "Area 1" else 0,
    "Wilderness_Area_2": 1 if wilderness_choice == "Area 2" else 0,
    "Wilderness_Area_3": 1 if wilderness_choice == "Area 3" else 0,
    "Wilderness_Area_4": 1 if wilderness_choice == "Area 4" else 0,
}

# Build a dictionary of features we know from user (subset)
known_dict = {
    'Elevation': elevation,
    'Aspect': aspect,
    'Slope': slope,
    'Horizontal_Distance_To_Hydrology': hd_hydro,
    'Vertical_Distance_To_Hydrology': vd_hydro,
    'Horizontal_Distance_To_Roadways': hd_road,
    'Horizontal_Distance_To_Fire_Points': hd_fire,
    'Hillshade_9am': hs_9am,
    'Hillshade_Noon': hs_noon,
    'Hillshade_3pm': hs_3pm,
    'Hillshade_Diff_Noon_3pm': hs_diff_noon_3pm,
    'Hillshade_Diff_Noon_9am': hs_diff_noon_9am,
    'Road_Hydro_Ratio': road_hydro_ratio,
}
# add wilderness
known_dict.update(wilderness_values)

# If the model has attribute feature_names_in_, use it (sklearn >= 1.0)
if hasattr(model, "feature_names_in_"):
    required_cols = list(model.feature_names_in_)
else:
    # Try to load saved columns from notebook if present
    if os.path.exists(COLUMNS_PKL):
        try:
            required_cols = joblib.load(COLUMNS_PKL)
        except Exception as e:
            st.error(f"Failed to load columns list {COLUMNS_PKL}: {e}")
            required_cols = None
    else:
        required_cols = None

if required_cols is None:
    st.warning("Model does not expose feature names and no model_columns.pkl found.\n"
               "Please run one line in your training notebook to save columns list:\n\n"
               "    joblib.dump(X.columns.tolist(), 'model_columns.pkl')\n\n"
               "Then place that file in the same folder and reload the app.")
    st.stop()

# Build input row with zeros for missing features and fill known where possible
input_row = {}
for c in required_cols:
    if c in known_dict:
        input_row[c] = known_dict[c]
    else:
        # if soil type or wilderness etc not provided, default 0
        input_row[c] = 0

# Create DataFrame with the same column order as required_cols
input_df = pd.DataFrame([input_row], columns=required_cols)

st.subheader("Preview of model input (first 20 cols)")
st.write(input_df.iloc[:, :20])

# Scale using loaded scaler
try:
    scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"Scaler.transform failed. Likely column order or dtype mismatch. Error: {e}")
    st.stop()

# Prediction
if st.button("Predict Forest Cover Type"):
    try:
        pred = model.predict(scaled)[0]
        st.success(f"🌳 Predicted Forest Cover Type: {pred}")
    except Exception as e:
        st.error(f"Model prediction failed: {e}")