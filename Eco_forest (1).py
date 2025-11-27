### IMPORT LIBRARIES

import streamlit as st
import pandas as pd
import numpy as np
import pickle

### LOAD SAVED MODEL AND LABEL ENCODER

with open('best_random_forest_model.pkl','rb') as file:
    model=pickle.load(file)

with open('labelencoder.pkl','rb') as file:
    label_encoder=pickle.load(file)

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: yellow;'>☘️FOREST COVER TYPE PREDICTION WITH 20 FEATURES</h1>", unsafe_allow_html=True)

### CREATE USER INPUTS FOR 20 FEATURES 

elevation = st.number_input("Elevation", value=2000)
total_distance = st.number_input("Total Distance", value=500)
horiz_dist_road = st.number_input("Horizontal Distance To Roadways", value=1000)
horiz_dist_fire = st.number_input("Horizontal Distance To Fire Points", value=500)
horiz_dist_hydro = st.number_input("Horizontal Distance To Hydrology", value=100)
wilderness_area_1 = st.selectbox("Wilderness Area 1 (0 or 1)", [0,1])
wilderness_area_4 = st.selectbox("Wilderness Area 4 (0 or 1)", [0,1])
vert_dist_hydro = st.number_input("Vertical Distance To Hydrology", value=0)
soil_type_10 = st.selectbox("Soil Type 10 (0 or 1)", [0,1])
wilderness_area_3 = st.selectbox("Wilderness Area 3 (0 or 1)", [0,1])
hillshade_9am = st.number_input("Hillshade 9am", value=100)
aspect = st.number_input("Aspect", value=90)
hillshade_3pm = st.number_input("Hillshade 3pm", value=120)
mean_hillshade = st.number_input("Mean Hillshade", value=100)
hillshade_noon = st.number_input("Hillshade Noon", value=150)
soil_type_3 = st.selectbox("Soil Type 3 (0 or 1)", [0,1])
slope = st.number_input("Slope", value=10)
soil_type_38 = st.selectbox("Soil Type 38 (0 or 1)", [0,1])
soil_type_39 = st.selectbox("Soil Type 39 (0 or 1)", [0,1])
soil_type_4 = st.selectbox("Soil Type 4 (0 or 1)", [0,1])

### PREPARE INPUT DATAFRAME 

input_data = [[
    elevation, total_distance, horiz_dist_road, horiz_dist_fire,
    horiz_dist_hydro, wilderness_area_1, wilderness_area_4,
    vert_dist_hydro, soil_type_10, wilderness_area_3, hillshade_9am,
    aspect, hillshade_3pm, mean_hillshade, hillshade_noon,
    soil_type_3, slope, soil_type_38, soil_type_39, soil_type_4
]]
input_df = pd.DataFrame(input_data, columns=model.feature_names_in_)

if st.button("Predict Forest Cover Type"):
    predicted_class_encoded = model.predict(input_df)[0]
    predicted_class = label_encoder.inverse_transform([predicted_class_encoded])[0]
    st.success(f"Predicted Forest Cover Type: {predicted_class}")

