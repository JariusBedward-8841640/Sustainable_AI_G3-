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

# --- SIDEBAR (INPUTS) ---
with st.sidebar:
    st.header("Model Architecture")
    layers = st.number_input("Number of Layers", min_value=1, max_value=200, value=12)
    training_time = st.number_input("Training Time (Hours)", min_value=0.1, value=5.0)
    flops_input = st.text_input("FLOPs (e.g. 1.5e18)", value="1.5e18")
    
    st.markdown("---")
    st.info("ℹ️ Uses Linear Regression on synthetic training data.")

# --- MAIN AREA ---
st.subheader("Enter Prompt Context")
prompt_text = st.text_area("Input Prompt:", height=100, placeholder="Enter your prompt here...")

# --- LOGIC FLOW ---

# BUTTON 1: INITIAL PREDICTION
if st.button("🚀 Analyze Consumption", type="primary"):
    if not prompt_text:
        st.error("Please enter a prompt.")
    else:
        estimator = EnergyEstimator()
        results = estimator.estimate(prompt_text, layers, training_time, flops_input)
        
        st.session_state['original_results'] = results
        st.session_state['prompt'] = prompt_text
        st.session_state['step'] = 1 

# --- VIEW 1: INITIAL RESULT ---
if st.session_state['step'] >= 1:
    st.divider()
    st.subheader("📊 Analysis Report")
    
    res = st.session_state['original_results']
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Energy", f"{res['energy_kwh']} kWh")
    col2.metric("Carbon Footprint", f"{res['carbon_kg']} kgCO2")
    col3.metric("Token Count", res['token_count'])
    
    st.info(f"**System Suggestion:** {res['suggestion']}")

    # --- SHOW GRAPH (STEP 1) ---
    # We only show this here if we remain in step 1. 
    # If we are in step 2, we will render it at the bottom to keep the flow clean.
    if st.session_state['step'] == 1:
        with st.expander("📈 View Model Performance Graph", expanded=True):
            estimator = EnergyEstimator()
            fig = estimator.get_training_plot()
            st.pyplot(fig)
    
        st.markdown("---")
        st.write("👉 **Can we do better?** Click below to optimize the prompt and architecture.")
        
        if st.button("✨ Improve & Optimize Prompt"):
            # 1. Simplify Prompt
            simplifier = PromptSimplifier()
            better_prompt = simplifier.optimize(st.session_state['prompt'])
            
            # 2. Estimate with "Improved" inputs
            improved_layers = int(layers * 0.8) if layers > 1 else 1
            improved_time = training_time * 0.8
            
            new_results = estimator.estimate(better_prompt, improved_layers, improved_time, flops_input)
            
            st.session_state['improved_results'] = new_results
            st.session_state['better_prompt'] = better_prompt
            st.session_state['step'] = 2 
            st.rerun()

# --- VIEW 2: COMPARISON RESULT ---
if st.session_state['step'] == 2:
    st.divider()
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
        
    st.balloons()

    # --- SHOW GRAPH AGAIN (STEP 2) ---
    st.markdown("---")
    st.subheader("Model Context")
    with st.expander("📈 View Model Performance Graph", expanded=True):
        estimator = EnergyEstimator()
        fig = estimator.get_training_plot()
        st.pyplot(fig)
    
    if st.button("🔄 Reset"):
        st.session_state['step'] = 0
        st.rerun()