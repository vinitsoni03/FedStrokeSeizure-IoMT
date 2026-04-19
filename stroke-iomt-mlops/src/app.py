import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
from datetime import datetime

# Import Hospital cases
try:
    from src.cases import NORMAL_SIGNAL, SEIZURE_SIGNAL
    HAS_CASES = True
except ImportError:
    HAS_CASES = False

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
st.sidebar.title("Clinical Portal v2.0")
st.sidebar.info("Hospital Simulation Mode Active")
st.sidebar.markdown("---")

# API Configuration
API_URL = "http://api:8000"  # Inside docker network

# Main Header
st.title("🧠 Seizure & Stroke Prediction System")
st.markdown("*Real-time clinical monitoring using realistic Hospital Case Library.*")

tabs = st.tabs(["📊 Seizure Prediction (EEG)", "💓 Stroke Risk Analysis", "📂 S3 Log History"])

# ==========================================
# SEIZURE PREDICTION TAB
# ==========================================
with tabs[0]:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Data Acquisition")
        options = ["Hospital Case Library", "Manual Record Entry", "Generate Random Test Signal"]
        data_source = st.radio("Signal Source", options)
        
        signal_data = None
        if data_source == "Hospital Case Library":
            patient = st.selectbox("Select Patient Case", ["Patient A-01 (Normal Activity)", "Patient B-05 (Seizure Event)"])
            if "A-01" in patient:
                signal_data = NORMAL_SIGNAL
                st.info("🏥 Case Loaded: Stable background EEG from CHB-MIT hospital records.")
            else:
                signal_data = SEIZURE_SIGNAL
                st.warning("🏥 Case Loaded: High-amplitude seizure discharge recorded in-situ.")
        
        elif data_source == "Generate Random Test Signal":
            dummy = np.random.randn(23, 256)
            signal_data = dummy.flatten().tolist()
            st.info("🧪 Synthetic 23-Channel EEG buffer generated.")
            
        else:
            raw_input = st.text_area("Paste Clinical JSON Record", height=200)
            if raw_input:
                try:
                    signal_data = json.loads(raw_input)
                except:
                    st.error("Invalid JSON format.")

        model_choice = st.selectbox("Inference Engine", ["CNN Deep Learning", "Random Forest Standalone"])

    with col2:
        st.subheader("Signal Visualization (Real Patient Data)")
        if signal_data:
            signals = np.array(signal_data).reshape(23, 256)
            fig, ax = plt.subplots(figsize=(12, 4), facecolor='#0e1117')
            ax.set_facecolor('#0e1117')
            
            # Plot representative channels
            for i in range(min(5, 23)):
                ax.plot(signals[i] + (i * 100), label=f"Ch {i}", linewidth=1.0)
            
            ax.set_title("Clinical EEG Monitoring Stream", color='white')
            ax.tick_params(colors='white')
            ax.legend(loc='upper right', fontsize='small')
            st.pyplot(fig)
        else:
            st.info("Awaiting clinical record...")

    if st.button("RUN CLINICAL INFERENCE"):
        if signal_data:
            endpoint = "/predict_seizure_cnn" if "CNN" in model_choice else "/predict_seizure_rf"
            try:
                payload = {"data": signal_data, "channels": 23, "timesteps": 256}
                with st.spinner("Analyzing neural pathways..."):
                    resp = requests.post(f"{API_URL}{endpoint}", json=payload)
                    data = resp.json()
                
                st.markdown("---")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    if data['prediction'] == 1:
                        st.error("🚨 CRITICAL: SEIZURE DETECTED")
                    else:
                        st.success("✅ STATUS: NORMAL ACTIVITY")
                
                with res_col2:
                    st.metric("Detection Confidence", f"{data['probability'] * 100:.2f}%")
                    
            except Exception as e:
                st.error(f"API Error (Check Connection): {str(e)}")
        else:
            st.warning("No hospital record loaded.")

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
                with st.spinner("Processing clinical markers..."):
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
    st.subheader("Cloud Data Lake (AWS S3)")
    st.info("Clinical audit logs are being streamed to `seizure-prediction-logs`.")
    st.markdown("""
        The system automatically archives every prediction for clinical auditing.
        - **Real-time Streaming**: Active
        - **Bucket**: `seizure-prediction-logs` 
        - **Format**: Clinically-formatted JSON
    """)

st.sidebar.markdown("---")
st.sidebar.caption(f"Sync: {datetime.now().strftime('%H:%M:%S')}")
