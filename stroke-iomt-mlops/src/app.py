import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests
import ast

st.set_page_config(page_title="Seizure Prediction", layout="wide")

st.title("Privacy-Preserving Seizure Prediction System")
st.markdown("Upload raw EEG signal points and select a model to predict seizure activity.")

# Input simulation
st.sidebar.header("Data Source")
option = st.sidebar.selectbox("Select Signal Type", ["Input Manually", "Generate Dummy Signal"])

signal_data = None

if option == "Generate Dummy Signal":
    # Generate 1 sample of 23 channels x 256 timesteps
    dummy = np.random.randn(23, 256)
    signal_data = dummy.flatten().tolist()
    st.write("Generated a 1-second dummy EEG signal (23 channels, 256Hz).")
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(dummy[0][:100], label="Channel 0")
    ax.plot(dummy[1][:100], label="Channel 1")
    ax.set_title("First 100 timesteps of Dummy Channels 0 and 1")
    ax.legend()
    st.pyplot(fig)

elif option == "Input Manually":
    signal_input = st.text_area("Paste Signal Data (List of 5888 floats)", height=150)
    if st.button("Parse Data"):
        try:
            signal_data = ast.literal_eval(signal_input)
            if len(signal_data) != 5888:
                st.error("Expected 5888 points!")
                signal_data = None
            else:
                st.success("Successfully loaded.")
        except:
            st.error("Invalid format.")

st.header("Prediction")
model_type = st.selectbox("Select Model", ["Random Forest", "CNN"])

if st.button("Predict"):
    if signal_data is None:
        st.warning("Please provide or generate signal data first.")
    else:
        # We assume the FastAPI is running locally on port 8000
        url = "http://127.0.0.0:8000/predict_rf" if model_type == "Random Forest" else "http://localhost:8000/predict_cnn"
        
        payload = {
            "data": signal_data,
            "channels": 23,
            "timesteps": 256
        }
        
        try:
            with st.spinner("Predicting..."):
                response = requests.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                
                prediction = result["prediction"]
                probability = result["probability"]
                
                if prediction == 1:
                    st.error(f"⚠️ SEIZURE DETECTED (Probability: {probability:.2f})")
                else:
                    st.success(f"✅ Normal Activity (Probability: {probability:.2f})")
        except Exception as e:
            st.error(f"Error communicating with API (Is it running?): {str(e)}")

st.markdown("---")
st.markdown("*Note: For this dashboard to work, run `uvicorn src.api:app --reload` first in a separate terminal.*")
