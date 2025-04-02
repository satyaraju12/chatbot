import streamlit as st
import google.generativeai as genai
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Load API key securely (replace with your actual API key)
genai.configure(api_key="AIzaSyC8tuqLoWlAMrz90XmNkXBL36ErDppMD4I")  # Replace with your actual API key


# Load trained XGBoost model
best_xgb = joblib.load("C:/Users/Aditya/OneDrive/Desktop/ml model/xgboost_crop_yield_model.pkl")

# Soil suitability mapping
soil_suitability = {
    "Wheat": ["Loamy", "Clayey"],
    "Rice": ["Clayey", "Loamy"],
    "Maize": ["Loamy", "Sandy"],
    "Sugarcane": ["Clayey", "Loamy"],
    "Cotton": ["Sandy", "Loamy"],
    "Pulses": ["Sandy", "Loamy"],
}

# Streamlit UI
st.title("🌾 AI-Powered Crop Yield Prediction & Farming Report")

# User input fields
state = st.text_input("Enter the State:")
crop = st.selectbox("Select the Crop:", list(soil_suitability.keys()))
soil_type = st.selectbox("Select the Soil Type:", ["Loamy", "Clayey", "Sandy"])
season = st.selectbox("Select the Season:", ["Kharif", "Rabi", "Summer"])
rainfall_category = st.selectbox("Select the Rainfall Category:", ["Low", "Medium", "High"])
area_of_land = st.slider("Enter the Land Area (hectares):", min_value=0.1, max_value=1000.0, value=1.0)

if st.button("Predict Yield & Generate Report"):
    # Check soil suitability
    if crop in soil_suitability:
        best_soils = soil_suitability[crop]
        if soil_type not in best_soils:
            soil_warning = f"⚠️ {soil_type} is NOT ideal for {crop}. Best soils: {', '.join(best_soils)}."
            soil_advice = "✅ Consider amendments or switching crops."
        else:
            soil_warning = f"✅ {soil_type} is ideal for {crop}!"
            soil_advice = ""
    else:
        soil_warning = "ℹ️ No soil data available."
        soil_advice = ""
    
    # Dummy input features (Replace with real data processing)
    try:
        new_data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])  # Adjust as per model features
        predicted_yield_per_hectare = best_xgb.predict(new_data)[0]
        total_yield = predicted_yield_per_hectare * area_of_land
    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")
        st.stop()
    
    # Construct AI Prompt
    prompt = f"""
    Generate a farming report for:
    - **State:** {state}
    - **Crop:** {crop}
    - **Soil Type:** {soil_type}
    - **Season:** {season}
    - **Rainfall Category:** {rainfall_category}
    - **Land Area:** {area_of_land:.2f} hectares
    - **Predicted Yield per hectare:** {predicted_yield_per_hectare:.2f} tons
    - **Estimated Total Yield:** {total_yield:.2f} tons
    
    ### **Soil Suitability Check**
    {soil_warning}
    {soil_advice}

    ### **Report Requirements:**
    1. **Farming Suggestions:** Region & season-specific strategies for better crop production.
    2. **Soil Health & Fertilizer Recommendations:** Fertilizers for {soil_type} soil and {crop}.
    3. **Pest & Disease Control:** Common threats for {crop} in {state} during {season}.
    4. **Water & Irrigation Management:** Best irrigation methods based on {rainfall_category} rainfall.
    5. **Best Agricultural Practices:** Sowing time, crop rotation, and harvesting guidelines.
    """
    
    # Generate Farming Report
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_model.generate_content(prompt)
    
    # Display results
    st.success("✅ Prediction & Report Generated Successfully!")
    st.write(f"**Predicted Yield per hectare:** {predicted_yield_per_hectare:.2f} tons")
    st.write(f"**Total Estimated Yield:** {total_yield:.2f} tons")
    st.subheader("📝 Generated Farming Report")
    st.write(response.text)
