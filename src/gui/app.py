#### Developer: Mostafa Allahmoradi
#### Course: CSCN8010 - Machine Learning
#### Date: December 2025

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

from layout import (
    sidebar_inputs,
    input_prompt_view,
    render_initial_analysis,
    optimization_metrics,
    token_reduction_plot,
    energy_impact_visualization,
    prompt_comparison,
    semantic_validation,
    footer
)

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.prediction.estimator import EnergyEstimator
from src.nlp.simplifier import PromptSimplifier
from src.utils.data_logger import DataLogger

# Initialize data logger
data_logger = DataLogger(db_type="sqlite")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Sustainable AI", layout="wide")
st.title("🌱 Sustainable AI")
st.markdown("### Transparency and Energy-Efficient Prompt/Context Engineering")

# --- SESSION STATE ---
if 'step' not in st.session_state:
    st.session_state['step'] = 0
if 'original_results' not in st.session_state:
    st.session_state['original_results'] = {}
if 'optimized_results' not in st.session_state:
    st.session_state['optimized_results'] = {}
if 'optimization_analysis' not in st.session_state:
    st.session_state['optimization_analysis'] = {}

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
model_type, layers, training_time, flops_input = sidebar_inputs()

# Add NLP Model Info to sidebar
with st.sidebar:
    # NLP Model Info
    st.markdown("---")
    st.markdown("#### 🧠 NLP Optimization")
    
    # Check T5 status and display to user
    try:
        simplifier_check = PromptSimplifier(use_ml_model=True)
        model_status = simplifier_check.get_model_status()
        
        if model_status.get("t5_available"):
            is_trained = model_status.get("is_trained", False)
            if is_trained:
                st.success("✅ T5 Fine-tuned Model Active")
                st.caption("Using trained prompt optimization model")
            else:
                st.info("ℹ️ T5 Base Model Active")
                st.caption("Using base T5 + rules (not fine-tuned)")
                with st.expander("🔧 Fine-tune for better results"):
                    st.markdown("Run training to improve accuracy:")
                    st.code("python scripts/train_t5_model.py --epochs 10", language="bash")
        else:
            st.warning("⚠️ Rule-Based Mode")
            st.caption("T5 unavailable - using rule-based optimization")
            with st.expander("Setup T5 Model"):
                st.code("""pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentencepiece""", language="bash")
    except Exception as e:
        st.info("ℹ️ Using rule-based optimization")
        st.caption(f"T5 check failed: {str(e)[:50]}")

# --- MAIN AREA ---
prompt_text = input_prompt_view()

estimator = EnergyEstimator(model_type=model_type)

# --- LOGIC FLOW ---

# BUTTON 1: INITIAL PREDICTION WITH PROMPT OPTIMIZATION
if st.button("🚀 Analyze Consumption", type="primary"):
    if not prompt_text:
        st.error("Please enter a prompt.")
    else:
        with st.spinner("Analyzing energy consumption and optimizing prompt..."):
            # 1. Energy estimation for original prompt
            original_results = estimator.estimate(prompt_text, layers, training_time, flops_input)

            # 2. Get full prompt optimization analysis
            simplifier = PromptSimplifier(use_ml_model=True)
            optimization_analysis = simplifier.get_full_analysis(prompt_text)

            # 3. Estimate energy for optimized prompt
            optimized_prompt = optimization_analysis.get('optimized', prompt_text)
            optimized_results = estimator.estimate(optimized_prompt, layers, training_time, flops_input)

            # Store results in the session state
            st.session_state['prompt'] = prompt_text
            st.session_state['original_results'] = original_results
            st.session_state['optimization_analysis'] = optimization_analysis
            st.session_state['optimized_results'] = optimized_results

            # 4. Log the analysis session
            try:
                session_id = data_logger.log_analysis(
                    prompt_text=prompt_text,
                    original_tokens=optimization_analysis.get('original_tokens', 0),
                    optimized_tokens=optimization_analysis.get('optimized_tokens', 0),
                    token_reduction_pct=optimization_analysis.get('token_reduction_pct', 0),
                    energy_kwh=float(original_results['energy_kwh']),
                    carbon_kg=float(original_results['carbon_kg']),
                    optimized_energy_kwh=float(optimized_results['energy_kwh']),
                    optimized_carbon_kg=float(optimized_results['carbon_kg']),
                    semantic_similarity=optimization_analysis.get('semantic_similarity', 0),
                    quality_score=optimization_analysis.get('quality_score', 0),
                    model_type=model_type,
                    layers=layers,
                    training_hours=training_time,
                    flops=flops_input,
                    store_full_prompt=False  # Privacy: store hash only
                )
                st.session_state['session_id'] = session_id
            except Exception as e:
                print(f"Logging failed: {e}")

            st.session_state['step'] = 1

        st.toast('Initial Analysis Complete!', icon='⚡')

# --- VIEW: ANALYSIS RESULTS ---
if st.session_state['step'] >= 1:
    render_initial_analysis(
        st.session_state['original_results'],
        st.session_state.get('optimization_analysis', {})
    )

    # --- SHOW ENERGY PERFORMANCE GRAPH ---
    st.markdown("#### Initial Model Performance")
    with st.expander("📈 View Energy Performance Graph", expanded=True):
        fig = estimator.get_training_plot(layers, original_results['energy_kwh'])
        st.pyplot(fig)

    st.markdown("---")

    # =================================================================
    # PROMPT OPTIMIZATION ANALYSIS SECTION
    # =================================================================
    # Get optimized results
    res_optimized = st.session_state.get('optimized_results', {})  
    opt = st.session_state.get('optimization_analysis', {})

    # --- Optimization Metrics Row ---
    optimization_metrics(res_optimized, opt)

    # --- Prompt Comparison ---
    prompt_comparison(st.session_state['prompt'], opt)

    # --- Semantic Validation ---
    semantic_validation(opt)

    # --- Visual: Similarity & Reduction Charts ---
    token_reduction_plot(opt)
    
    # --- Energy Comparison Visualization ---
    energy_impact_visualization(res_optimized)

    with st.expander("📉 View Optimized Performance Graph", expanded=True):
        optimized_results = st.session_state.get('optimized_results', {})
        fig = estimator.get_training_plot(layers, optimized_results.get('energy_kwh', 0))
        st.pyplot(fig)

    st.markdown("---")

    # --- Suggestions ---
    suggestions = opt.get('suggestions', [])
    if suggestions:
        st.markdown("#### 💡 Suggestions")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")

# FOOTER
footer()