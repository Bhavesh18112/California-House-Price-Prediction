import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("best_house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="California House Price Prediction")

st.title("🏠 California House Price Prediction")
st.write("Enter the house details below.")

# Input fields
MedInc = st.number_input("Median Income", value=3.5)
HouseAge = st.number_input("House Age", value=20.0)
AveRooms = st.number_input("Average Rooms", value=5.0)
AveBedrms = st.number_input("Average Bedrooms", value=1.0)
Population = st.number_input("Population", value=1000.0)
AveOccup = st.number_input("Average Occupancy", value=3.0)
Latitude = st.number_input("Latitude", value=34.0)
Longitude = st.number_input("Longitude", value=-118.0)

if st.button("Predict Price"):
    data = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms,
                          Population, AveOccup, Latitude, Longitude]],
                        columns=[
                            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                            "Population", "AveOccup", "Latitude", "Longitude"
                        ])

    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)

    st.success(f"🏡 Predicted House Price: ${prediction[0] * 100000:,.2f}")