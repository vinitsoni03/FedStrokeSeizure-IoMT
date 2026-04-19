import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
from datetime import datetime

# Set Page Config
st.set_page_config(
    page_title="FedStrokeSeizure Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .stMetric {
        background-color: #1e2227;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# sidebar header
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2491/2491413.png", width=80)
st.sidebar.title("Clinical Portal")
st.sidebar.markdown("---")

# API Configuration
API_URL = "http://api:8000"  # Inside docker network

# Main Header
st.title("🧠 Seizure & Stroke Prediction System")
st.markdown("*Production-grade clinical monitoring using Federated Learning optimized models.*")

tabs = st.tabs(["📊 Seizure Prediction (EEG)", "💓 Stroke Risk Analysis", "📂 S3 Log History"])

# ==========================================
# SEIZURE PREDICTION TAB
# ==========================================
with tabs[0]:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Data Acquisition")
        data_source = st.radio("EEG Source", ["Generate Test Signal", "Manual Entry"])
        
        signal_data = None
        if data_source == "Generate Test Signal":
            # 23 channels x 256Hz = 5888 points
            dummy = np.random.randn(23, 256)
            signal_data = dummy.flatten().tolist()
            st.info("✅ 23-Channel EEG buffer generated (1 second at 256Hz)")
        else:
            raw_input = st.text_area("Paste Signal Data (JSON Array)", height=200)
            if raw_input:
                try:
                    signal_data = json.loads(raw_input)
                except:
                    st.error("Invalid JSON format.")

        model_choice = st.selectbox("Inference Engine", ["CNN Deep Learning", "Random Forest Standalone"])

    with col2:
        st.subheader("Signal Visualization")
        if signal_data:
            # Reshape for plotting
            signals = np.array(signal_data).reshape(23, 256)
            fig, ax = plt.subplots(figsize=(12, 4), facecolor='#0e1117')
            ax.set_facecolor('#0e1117')
            
            # Plot first 5 channels
            for i in range(min(5, 23)):
                ax.plot(signals[i] + (i * 5), label=f"Ch {i}", linewidth=0.7)
            
            ax.set_title("Real-time EEG Streaming Data", color='white')
            ax.tick_params(colors='white')
            ax.legend(loc='upper right', fontsize='small')
            st.pyplot(fig)
        else:
            st.info("Awaiting signal input...")

    if st.button("RUN SEIZURE INFERENCE"):
        if signal_data:
            endpoint = "/predict_seizure_cnn" if "CNN" in model_choice else "/predict_seizure_rf"
            try:
                payload = {"data": signal_data, "channels": 23, "timesteps": 256}
                with st.spinner("Analyzing high-frequency components..."):
                    resp = requests.post(f"{API_URL}{endpoint}", json=payload)
                    data = resp.json()
                
                # Results Display
                st.markdown("---")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    if data['prediction'] == 1:
                        st.error("🚨 SEIZURE DETECTED")
                    else:
                        st.success("✅ NORMAL ACTIVITY")
                
                with res_col2:
                    st.metric("Confidence Level", f"{data['probability'] * 100:.2f}%")
                    
            except Exception as e:
                st.error(f"API Error: {str(e)}")
        else:
            st.warning("No signal data loaded.")

# ==========================================
# STROKE RISK TAB
# ==========================================
with tabs[1]:
    st.subheader("Patient Clinical Profile (XGBoost Analysis)")
    
    with st.form("stroke_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 0, 120, 45)
            gender = st.selectbox("Gender", ["Female", "Male"], index=0)
            married = st.selectbox("Ever Married", ["No", "Yes"], index=1)
        with c2:
            ht = st.selectbox("Hypertension", ["No", "Yes"], index=0)
            hd = st.selectbox("Heart Disease", ["No", "Yes"], index=0)
            work = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        with c3:
            glucose = st.number_input("Avg Glucose Level", 50.0, 300.0, 105.0)
            bmi = st.number_input("BMI", 10.0, 60.0, 24.0)
            residence = st.selectbox("Residence Type", ["Urban", "Rural"])
            smoke = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

        submit = st.form_submit_button("EVALUATE STROKE RISK")
        
        if submit:
            # Mapping
            mapped_data = {
                "gender": 1.0 if gender == "Male" else 0.0,
                "age": float(age),
                "hypertension": 1.0 if ht == "Yes" else 0.0,
                "heart_disease": 1.0 if hd == "Yes" else 0.0,
                "ever_married": 1.0 if married == "Yes" else 0.0,
                "work_type": {"Private": 2.0, "Self-employed": 3.0, "Govt_job": 0.0, "children": 4.0, "Never_worked": 1.0}[work],
                "Residence_type": 1.0 if residence == "Urban" else 0.0,
                "avg_glucose_level": float(glucose),
                "bmi": float(bmi),
                "smoking_status": {"formerly smoked": 1.0, "never smoked": 2.0, "smokes": 3.0, "Unknown": 0.0}[smoke]
            }
            
            try:
                with st.spinner("Processing XGBoost risk layers..."):
                    resp = requests.post(f"{API_URL}/predict_stroke", json=mapped_data)
                    data = resp.json()
                
                st.markdown("---")
                if data['prediction'] == 1:
                    st.warning(f"⚠️ High Risk Detected (Probability: {data['probability'] * 100:.1f}%)")
                else:
                    st.success(f"✅ Low Risk (Probability: {data['probability'] * 100:.1f}%)")
            except Exception as e:
                st.error(f"API Error: {str(e)}")

# ==========================================
# HISTORY TAB
# ==========================================
with tabs[2]:
    st.subheader("Data Lake Integration (AWS S3)")
    st.info("System is configured to log all predictions to `seizure-prediction-logs` bucket.")
    st.markdown("""
        Every interaction above is automatically captured as a JSON record in your S3 archive.
        - **Latency**: < 100ms (Asynchronous Background Task)
        - **Storage Format**: `predictions/YYYY-MM-DD/uuid.json`
    """)
    if st.button("Manual Log Verification"):
        st.write("Fetching latest logs metadata... [Functionality under development]")

st.sidebar.markdown("---")
st.sidebar.caption(f"Connected to: {API_URL}")
st.sidebar.caption(f"Last sync: {datetime.now().strftime('%H:%M:%S')}")
