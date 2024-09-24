import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import altair.vegalite.v4 as alt


# Load the trained model and scaler
model = joblib.load("best_model1.pkl")


# Define the app title and layout
st.title("Iris Flower Species App")

# Define input fields for features
id = st.number_input("Id", min_value=0, max_value=100, value=60, step=1)
sepal_length_cm = st.number_input("Sepal Length in CM", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width_cm = st.number_input("Sepal Width in CM", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
petal_length_cm = st.number_input("Petal Length in CM", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
petal_width_cm = st.number_input("Petal Width in CM", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

# Create a button for making predictions
if st.button("Predict"):
    # Process input values
    input_data = pd.DataFrame(
        {
            "Id": [id],
            "SepalLengthCm": [sepal_length_cm],
            "SepalWidthCm": [sepal_width_cm],
            "PetalLengthCm": [petal_length_cm],
            "PetalWidthCm": [petal_width_cm]
        }
    )

    # Scale input data using the scaler used during training
    input_data_scaled = scaler.transform(input_data)

    # Make a prediction using the trained model
    prediction = model.predict(input_data_scaled)

    # Display the prediction
    if prediction[0] == 1:
        st.success("Unhealthy Flower.")
    else:
        st.success("Healthy Flower.")
