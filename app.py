"""
BioBot Streamlit UI
Beautiful web interface for testing the CNN-LSTM model
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_and_merge, build_sequences, split_data, FEATURES
from cnn_lstm_model import BioBotCNNLSTM

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioBot - Livability Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .score-good {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .score-medium {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
    }
    .score-bad {
        background: linear-gradient(135deg, #ed213a 0%, #93291e 100%);
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained CNN-LSTM model"""
    try:
        model = BioBotCNNLSTM(verbose=0)
        model.load('models/biobot.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data():
    """Load and prepare test data"""
    try:
        df = load_and_merge()
        X, y, scaler = build_sequences(df)
        _, _, (X_test, y_test) = split_data(X, y)
        return df, X_test, y_test, scaler
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

# ── Main App ─────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown('<h1 class="main-header">🌿 BioBot Livability Predictor</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">CNN-LSTM Hybrid Model for Environmental Livability Assessment</p>', 
                unsafe_allow_html=True)
    
    # Load model and data
    with st.spinner("Loading model..."):
        model = load_model()
        df, X_test, y_test, scaler = load_data()
    
    if model is None:
        st.error("Failed to load model. Please check model file.")
        return
    
    # ── Sidebar: Model Info ──────────────────────────────────────────────
    with st.sidebar:
        st.header("📊 Model Information")
        st.info(f"""
        **Architecture**: CNN-LSTM Hybrid
        
        **Features**: {len(FEATURES)}
        {', '.join(FEATURES)}
        
        **Sequence Length**: 24 timesteps (2h)
        
        **Target**: Vivabilite (0-1)
        """)
        
        st.divider()
        st.header("🎯 Quick Actions")
        
        if st.button("🔄 Reload Model", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        
        if st.button("📈 Show Test Samples", use_container_width=True):
            st.session_state.show_samples = True
        
        if st.button("🗑️ Clear Results", use_container_width=True):
            if 'results' in st.session_state:
                del st.session_state.results
            st.rerun()
    
    # ── Main Content: Two Columns ────────────────────────────────────────
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("⚙️ Sensor Inputs")
        st.markdown("Adjust biosense parameters to simulate environmental conditions.")
        
        # Feature Sliders
        inputs = {}
        
        st.subheader("🌡️ Temperature & Humidity")
        inputs['temperature'] = st.slider(
            "Temperature (°C)", 
            0.0, 50.0, 22.0, 0.5,
            help="Air temperature in Celsius"
        )
        inputs['humidity'] = st.slider(
            "Humidity (%)", 
            0.0, 100.0, 60.0, 1.0,
            help="Relative humidity percentage"
        )
        inputs['humidex'] = st.slider(
            "Humidex (feels-like °C)", 
            0.0, 60.0, 25.0, 0.5,
            help="Combined temperature + humidity index"
        )
        
        st.subheader("💨 Wind & Soil")
        inputs['wind'] = st.slider(
            "Wind Speed (km/h)", 
            0.0, 50.0, 10.0, 0.5,
            help="Wind speed for cooling effect"
        )
        inputs['soil_moisture'] = st.slider(
            "Soil Moisture (%)", 
            0.0, 100.0, 40.0, 1.0,
            help="Soil moisture content"
        )
        
        st.subheader("🏭 Air Quality")
        inputs['CO2'] = st.slider(
            "CO₂ (ppm)", 
            300, 2000, 500, 10,
            help="Carbon dioxide concentration"
        )
        inputs['TVOC'] = st.slider(
            "TVOC (ppb)", 
            0, 1000, 100, 10,
            help="Total volatile organic compounds"
        )
        inputs['PM2_5'] = st.slider(
            "PM2.5 (μg/m³)", 
            0.0, 200.0, 15.0, 0.5,
            help="Fine particulate matter"
        )
        inputs['PM10'] = st.slider(
            "PM10 (μg/m³)", 
            0.0, 300.0, 30.0, 0.5,
            help="Coarse particulate matter"
        )
        inputs['sound_level'] = st.slider(
            "Sound Level (dB)", 
            30, 100, 55, 1,
            help="Ambient noise level"
        )
        
        # Predict Button
        if st.button("🚀 Predict Livability", type="primary", use_container_width=True):
            # Create sequence (repeat input for 24 timesteps)
            input_seq = np.array([[inputs[f] for f in FEATURES]])
            input_seq = np.tile(input_seq, (24, 1))
            input_seq = input_seq.reshape(1, 24, len(FEATURES))
            
            # Scale if scaler exists
            if scaler is not None:
                input_flat = input_seq.reshape(-1, len(FEATURES))
                input_scaled = scaler.transform(input_flat)
                input_seq = input_scaled.reshape(1, 24, len(FEATURES))
            
            # Predict
            prediction = model.predict(input_seq)[0][0]
            
            # Store results
            st.session_state.results = {
                'score': prediction,
                'inputs': inputs,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }
    
    with col2:
        st.header("📊 Prediction Results")
        
        if 'results' in st.session_state:
            score = st.session_state.results['score']
            inputs = st.session_state.results['inputs']
            timestamp = st.session_state.results['timestamp']
            
            # Score Display
            st.markdown(f"**Prediction Time**: {timestamp}")
            
            # Score Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Livability Score", 'font': {'size': 24}},
                delta={'reference': 0.5, 'increasing': {'color': "RebeccaPurple"}},
                gauge={
                    'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 0.3], 'color': 'red'},
                        {'range': [0.3, 0.7], 'color': 'yellow'},
                        {'range': [0.7, 1], 'color': 'green'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Score Interpretation
            if score >= 0.7:
                st.success(f"✅ Excellent livability! Score: {score:.3f}")
                st.balloons()
            elif score >= 0.3:
                st.warning(f"⚠️ Moderate livability. Score: {score:.3f}")
            else:
                st.error(f"❌ Poor livability. Score: {score:.3f}")
            
            # Feature Importance Radar Chart
            st.subheader("🎯 Feature Values")
            
            # Normalize inputs for radar chart
            input_values = []
            for f in FEATURES:
                val = inputs[f]
                # Simple normalization for display (min/max from typical ranges)
                if f == 'temperature': norm = val / 50
                elif f == 'humidity': norm = val / 100
                elif f == 'humidex': norm = val / 60
                elif f == 'wind': norm = val / 50
                elif f == 'soil_moisture': norm = val / 100
                elif f == 'CO2': norm = (val - 300) / 1700
                elif f == 'TVOC': norm = val / 1000
                elif f == 'PM2_5': norm = val / 200
                elif f == 'PM10': norm = val / 300
                elif f == 'sound_level': norm = (val - 30) / 70
                else: norm = val / 100
                input_values.append(max(0, min(1, norm)))
            
            fig2 = go.Figure(data=go.Scatterpolar(
                r=input_values,
                theta=FEATURES,
                fill='toself',
                name='Input Profile',
                line_color='royalblue'
            ))
            fig2.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1]),
                ),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Input Summary Table
            st.subheader("📋 Input Summary")
            df_inputs = pd.DataFrame({
                'Feature': FEATURES,
                'Value': [inputs[f] for f in FEATURES],
                'Unit': ['°C', '%', '°C', 'km/h', '%', 'ppm', 'ppb', 'μg/m³', 'μg/m³', 'dB']
            })
            st.dataframe(df_inputs, use_container_width=True, hide_index=True)
            
        else:
            st.info("👆 Adjust the sliders and click 'Predict Livability' to see results.")
            
            # Show model performance metrics if available
            st.subheader("📈 Model Performance")
            
            col_metrics = st.columns(3)
            with col_metrics[0]:
                st.metric("Test MAE", "0.0416", "vivabilite units")
            with col_metrics[1]:
                st.metric("Test RMSE", "0.1638", "vivabilite units")
            with col_metrics[2]:
                st.metric("Accuracy", "~96%", "±0.04 on 0-1 scale")
            
            # Show sample predictions if button clicked
            if st.session_state.get('show_samples') and y_test is not None:
                st.subheader("🔍 Sample Predictions from Test Set")
                
                # Get predictions for first 10 test samples
                sample_preds = model.predict(X_test[:10], verbose=0)
                
                # Create comparison table
                sample_df = pd.DataFrame({
                    'Sample': [f'#{i+1}' for i in range(10)],
                    'Predicted': [f'{p[0]:.3f}' for p in sample_preds],
                    'Actual': [f'{y_test[i][0]:.3f}' for i in range(10)],
                    'Error': [f'{abs(sample_preds[i][0] - y_test[i][0]):.3f}' for i in range(10)]
                })
                
                st.dataframe(sample_df, use_container_width=True, hide_index=True)
                
                # Scatter plot of predictions vs actual
                fig3 = go.Figure(data=go.Scatter(
                    x=y_test[:10, 0],
                    y=[p[0] for p in sample_preds],
                    mode='markers',
                    marker=dict(size=10, color='royalblue'),
                    name='Predictions'
                ))
                fig3.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Perfect Prediction'
                ))
                fig3.update_layout(
                    title='Predicted vs Actual',
                    xaxis_title='Actual Vivabilite',
                    yaxis_title='Predicted Vivabilite',
                    height=400
                )
                st.plotly_chart(fig3, use_container_width=True)
    
    # ── Footer ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🌿 **BioBot CNN-LSTM** | Livable Area Prediction | Real biosense data</p>
        <p>Built with Streamlit • TensorFlow • Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
