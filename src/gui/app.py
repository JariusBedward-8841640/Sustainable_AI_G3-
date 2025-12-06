import streamlit as st
import pandas as pd
import os
import sys

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.prediction.estimator import EnergyEstimator
from src.nlp.simplifier import PromptSimplifier

# --- PAGE CONFIG ---
st.set_page_config(page_title="Sustainable AI", layout="wide")
st.title("🌱 Sustainable AI - Energy Predictor")
st.markdown("### Transparency & Efficiency in Generative AI")

# --- SESSION STATE ---
if 'step' not in st.session_state:
    st.session_state['step'] = 0 
if 'original_results' not in st.session_state:
    st.session_state['original_results'] = {}
if 'improved_results' not in st.session_state:
    st.session_state['improved_results'] = {}
# Store pending toast messages for display after reruns
if 'toast_message' not in st.session_state:
    st.session_state['toast_message'] = None

# --- TOAST MANAGER ---
# If a message was set in a previous run (before a rerun), show it now
if st.session_state['toast_message']:
    msg, icon = st.session_state['toast_message']
    st.toast(msg, icon=icon)
    st.session_state['toast_message'] = None

# --- SIDEBAR (INPUTS) ---
with st.sidebar:
    st.header("Model Architecture")
    layers = st.number_input("Number of Layers", min_value=1, max_value=200, value=12)
    training_time = st.number_input("Training Time (Hours)", min_value=0.1, value=5.0)
    flops_input = st.text_input("FLOPs (e.g. 1.5e18)", value="1.5e18")

    # Add dropdown for model type
    model_type = st.selectbox("Select Model Type", options=["RandomForest", "LinearRegression"], index=1)
    
    st.markdown("---")
    st.info(f"ℹ️ Uses {model_type} on synthetic training data.")

# --- MAIN AREA ---
st.subheader("Enter Prompt Context")
prompt_text = st.text_area("Input Prompt:", height=100, placeholder="Enter your prompt here...")

# --- LOGIC FLOW ---

# BUTTON 1: INITIAL PREDICTION
if st.button("🚀 Analyze Consumption", type="primary"):
    if not prompt_text:
        st.error("Please enter a prompt.")
    else:
        estimator = EnergyEstimator(model_type=model_type)
        results = estimator.estimate(prompt_text, layers, training_time, flops_input)
        
        st.session_state['original_results'] = results
        st.session_state['prompt'] = prompt_text
        st.session_state['step'] = 1
        # TOAST 1: Initial Analysis 
        st.toast('Initial Analysis Complete!', icon='⚡')

# --- VIEW: SHARED RESULTS (Visible in Step 1 AND Step 2) ---
if st.session_state['step'] >= 1:
    st.divider()
    st.subheader("📊 Analysis Report")
    
    res = st.session_state['original_results']
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Energy", f"{res['energy_kwh']} kWh")
    col2.metric("Carbon Footprint", f"{res['carbon_kg']} kgCO2")
    col3.metric("Token Count", res['token_count'])
    
    # Suggestion for original result
    # st.info(f"**System Suggestion:** {res['suggestion']}")
    st.info("**Message: ✅ Initial analysis complete.**" )

    # --- SHOW GRAPH 1 (ALWAYS VISIBLE) ---
    st.markdown("#### Initial Model Performance")
    with st.expander("📈 View Initial Performance Graph", expanded=True):
        estimator = EnergyEstimator(model_type=model_type)
        fig = estimator.get_training_plot(layers)
        st.pyplot(fig)
    
    st.markdown("---")

# --- VIEW 1: BUTTON ONLY ---
if st.session_state['step'] == 1:
    st.write("👉 **Can we do better?** Click below to optimize the prompt and architecture.")
    
    if st.button("✨ Improve & Optimize Prompt"):
        # 1. Simplify Prompt
        simplifier = PromptSimplifier()
        better_prompt = simplifier.optimize(st.session_state['prompt'])
        
        # 2. Estimate with "Improved" inputs
        improved_layers = int(layers * 0.8) if layers > 1 else 1
        improved_time = training_time * 0.8
        
        estimator = EnergyEstimator(model_type=model_type)
        new_results = estimator.estimate(better_prompt, improved_layers, improved_time, flops_input)
        
        st.session_state['improved_results'] = new_results
        st.session_state['better_prompt'] = better_prompt
        st.session_state['step'] = 2 
        # TOAST 2: Set message and rerun
        st.session_state['toast_message'] = ('Optimization Complete! Energy reduced.', '🌱')
        st.rerun()

# --- VIEW 2: COMPARISON RESULT & GRAPH 2 ---
if st.session_state['step'] == 2:
    st.info("**System Suggestion:** ✅ Optimized.")
    
    st.subheader("✅ Optimization Results")
    
    orig = st.session_state['original_results']
    new = st.session_state['improved_results']
    
    saved_energy = orig['energy_kwh'] - new['energy_kwh']
    percent_saved = (saved_energy / orig['energy_kwh']) * 100 if orig['energy_kwh'] > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("New Energy Usage", f"{new['energy_kwh']} kWh", delta=f"-{round(saved_energy, 2)} kWh", delta_color="inverse")
    c2.metric("New Carbon Footprint", f"{new['carbon_kg']} kgCO2", delta="-Low", delta_color="inverse")
    c3.metric("Efficiency Gain", f"{round(percent_saved, 1)}%")

    st.write("#### Prompt Optimization")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.warning("**Original Prompt**")
        st.code(st.session_state['prompt'], language="text")

    with col_b:
        st.success("**Optimized Prompt**")
        st.code(st.session_state['better_prompt'], language="text")
        
    # --- SHOW GRAPH 2 (NEW GRAPH) ---
    st.markdown("#### Optimized Model Performance")
    
    with st.expander("📉 View Optimized Performance Graph", expanded=True):
        estimator = EnergyEstimator(model_type=model_type)
        fig = estimator.get_training_plot()
        st.pyplot(fig)
    
    if st.button("🔄 Reset"):
        st.session_state['step'] = 0
        # TOAST 3: Set message and rerun
        st.session_state['toast_message'] = ('System Reset.', '🔄')
        st.rerun()