import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json
import os

model_path = os.path.join(current_dir, 'exoplanet_classifier_model.pkl')

st.set_page_config(
    page_title="NASA Exoplanet Classifier",
    page_icon="🪐",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    return model

@st.cache_resource  
def load_feature_info(json_path='model_features.json'):
    with open(json_path, 'r') as f:
        data = json.load(f)
    feature_names = data.get('feature_names', [])
    class_names = ['CANDIDATE', 'PLANET', 'NON_PLANET']
    return feature_names, class_names

# Loaded resources:
try:
    model = load_model()
    feature_names, class_names = load_feature_info()
    
    st.success("✅ Model loaded successfully!")
    st.subheader("🪐 NASA Exoplanet Classification Tool")
    st.write("Upload data or enter parameters to classify exoplanet candidates")
    
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Created two input methods
input_method = st.radio("Choose input method:", 
                       ["🔭 Single Observation", "📁 Upload CSV File"])

if input_method == "🔭 Single Observation":
    
    st.markdown("### Enter Observation Parameters")
    
    # Created input fields for each feature
    col1, col2, col3 = st.columns(3)
    
    input_data = {}
    with col1:
        input_data['koi_period'] = st.number_input('Orbital Period (days)', value=10.0, min_value=0.1)
        input_data['koi_impact'] = st.number_input('Impact Parameter (0-1)', value=0.5, min_value=0.0, max_value=1.0)
        input_data['koi_duration'] = st.number_input('Transit Duration (hours)', value=5.0, min_value=0.1)
        input_data['koi_depth'] = st.number_input('Transit Depth (ppm)', value=100.0, min_value=0.1)
        input_data['koi_ror'] = st.number_input('Planet/Star Radius Ratio', value=0.01, min_value=0.0)
        input_data['koi_srho'] = st.number_input('Stellar Density (g/cm³)', value=2.0, min_value=0.1)
        
    with col2:
        input_data['koi_prad'] = st.number_input('Planetary Radius (Earth radii)', value=2.0, min_value=0.1)
        input_data['koi_sma'] = st.number_input('Semi-Major Axis (AU)', value=0.1, min_value=0.01)
        input_data['koi_incl'] = st.number_input('Inclination (degrees)', value=89.0, min_value=0.0, max_value=90.0)
        input_data['koi_teq'] = st.number_input('Equilibrium Temp (K)', value=500.0, min_value=0.0)
        input_data['koi_insol'] = st.number_input('Insolation Flux (Earth=1)', value=1.0, min_value=0.0)
        input_data['koi_dor'] = st.number_input('Distance/Radius Ratio', value=20.0, min_value=1.0)
        
    with col3:
        input_data['koi_model_snr'] = st.number_input('Signal-to-Noise Ratio', value=15.0, min_value=0.0)
        input_data['koi_bin_oedp_sig'] = st.number_input('Odd-Even Depth Significance', value=0.0)
        input_data['koi_steff'] = st.number_input('Stellar Temp (K)', value=5500.0, min_value=0.0)
        input_data['koi_slogg'] = st.number_input('Stellar Surface Gravity (log10(cm/s²))', value=4.5, min_value=0.0)
        input_data['koi_smet'] = st.number_input('Stellar Metallicity (dex)', value=0.0)
        input_data['koi_srad'] = st.number_input('Stellar Radius (Solar radii)', value=1.0, min_value=0.1)
        input_data['koi_smass'] = st.number_input('Stellar Mass (Solar mass)', value=1.0, min_value=0.1)
    
    if st.button("Classify Observation", type="primary"):
        # Convert to DataFrame with correct column order
        observation_df = pd.DataFrame([input_data])[feature_names]
        
        # Make prediction
        probabilities = model.predict_proba(observation_df)[0]
        prediction_idx = np.argmax(probabilities)
        prediction = class_names[prediction_idx]
        
        # Display results
        st.markdown("### 📊 Classification Results")
        
        # Create result columns
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("Predicted Class", prediction)
            
            # Show confidence
            confidence = probabilities[prediction_idx]
            st.metric("Confidence", f"{confidence:.1%}")
        
        with res_col2:
            # Show all probabilities
            st.write("**Class Probabilities:**")
            for i, class_name in enumerate(class_names):
                prob = probabilities[i]
                st.write(f"{class_name}: {prob:.1%}")
        
        # Visual indicator
        if prediction == "PLANET":
            st.success("HIGH CONFIDENCE: This appears to be a real exoplanet!")
        elif prediction == "CANDIDATE":
            st.warning("CANDIDATE: Shows planetary characteristics but needs follow-up")
        else:
            st.error("LIKELY NOT A PLANET: Signal doesn't match expected planetary patterns")

else:  # CSV Upload method
    st.markdown("### Upload CSV Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read uploaded data
            user_data = pd.read_csv(uploaded_file)
            
            # Check if required columns exist
            missing_cols = [col for col in feature_names if col not in user_data.columns]
            
            if missing_cols:
                st.error(f"Failed - Missing columns: {missing_cols}")
            else:
                st.success(f"✅ Data loaded successfully! {len(user_data)} observations")
                
                # Make predictions
                predictions = model.predict(user_data[feature_names])
                probabilities = model.predict_proba(user_data[feature_names])
                
                # Add predictions to data
                results_df = user_data.copy()
                results_df['Predicted_Class'] = [class_names[idx] for idx in predictions]
                
                for i, class_name in enumerate(class_names):
                    results_df[f'{class_name}_Probability'] = probabilities[:, i]
                
                # Show results
                st.markdown("### 📈 Classification Results Preview")
                st.dataframe(results_df.head(10))
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="exoplanet_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Add model info in sidebar
with st.sidebar:
    
    st.markdown("### How to Use")
    st.write("1. Enter parameters manually OR upload CSV")
    st.write("2. Click 'Classify'")
    st.write("3. Review predictions and probabilities")
    
    st.markdown("### Model Details")
    st.write("Trained on NASA Kepler data")
    st.write("XGBoost algorithm")

    st.write("19 physical parameters used")
