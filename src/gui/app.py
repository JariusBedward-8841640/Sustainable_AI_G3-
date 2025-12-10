import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

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
st.title("🌱 Sustainable AI - Energy Predictor")
st.markdown("### Transparency & Efficiency in Generative AI")

# --- SESSION STATE ---
if 'step' not in st.session_state:
    st.session_state['step'] = 0 
if 'original_results' not in st.session_state:
    st.session_state['original_results'] = {}
if 'improved_results' not in st.session_state:
    st.session_state['improved_results'] = {}
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
with st.sidebar:
    st.header("Model Architecture")
    layers = st.number_input("Number of Layers", min_value=1, max_value=200, value=12)
    training_time = st.number_input("Training Time (Hours)", min_value=0.1, value=5.0)
    flops_input = st.text_input("FLOPs (e.g. 1.5e18)", value="1.5e18")

    # Add dropdown for model type
    model_type = st.selectbox("Select Model Type", options=["RandomForest", "LinearRegression"], index=1)
    
    st.markdown("---")
    st.info(f"ℹ️ Uses {model_type} on synthetic training data.")
    
    # NLP Model Info
    st.markdown("---")
    st.markdown("#### 🧠 NLP Optimization")
    st.caption("T5-based prompt optimizer with semantic similarity validation.")

# --- MAIN AREA ---
st.subheader("Enter Prompt Context")
prompt_text = st.text_area("Input Prompt:", height=100, placeholder="Enter your prompt here...")

# --- LOGIC FLOW ---

# BUTTON 1: INITIAL PREDICTION WITH PROMPT OPTIMIZATION
if st.button("🚀 Analyze Consumption", type="primary"):
    if not prompt_text:
        st.error("Please enter a prompt.")
    else:
        with st.spinner("Analyzing energy consumption and optimizing prompt..."):
            # 1. Energy estimation for original prompt
            estimator = EnergyEstimator(model_type=model_type)
            results = estimator.estimate(prompt_text, layers, training_time, flops_input)
            
            # 2. Get full optimization analysis
            simplifier = PromptSimplifier(use_ml_model=True)
            optimization_analysis = simplifier.get_full_analysis(prompt_text)
            
            # Store results
            st.session_state['original_results'] = results
            st.session_state['prompt'] = prompt_text
            st.session_state['optimization_analysis'] = optimization_analysis
            
            # 3. Estimate energy for optimized prompt
            optimized_prompt = optimization_analysis.get('optimized', prompt_text)
            optimized_results = estimator.estimate(optimized_prompt, layers, training_time, flops_input)
            st.session_state['optimized_results'] = optimized_results
            
            # 4. Log the analysis session
            try:
                session_id = data_logger.log_analysis(
                    prompt_text=prompt_text,
                    original_tokens=optimization_analysis.get('original_tokens', 0),
                    optimized_tokens=optimization_analysis.get('optimized_tokens', 0),
                    token_reduction_pct=optimization_analysis.get('token_reduction_pct', 0),
                    energy_kwh=float(results['energy_kwh']),
                    carbon_kg=float(results['carbon_kg']),
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
            
        st.toast('Analysis Complete!', icon='⚡')

# --- VIEW: ANALYSIS RESULTS ---
if st.session_state['step'] >= 1:
    st.divider()
    st.subheader("📊 Analysis Report")
    
    res = st.session_state['original_results']
    opt = st.session_state.get('optimization_analysis', {})
    
    # --- ENERGY METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Energy", f"{res['energy_kwh']} kWh")
    col2.metric("Carbon Footprint", f"{res['carbon_kg']} kgCO2")
    col3.metric("Original Tokens", opt.get('original_tokens', res['token_count']))
    col4.metric("Optimized Tokens", opt.get('optimized_tokens', '-'))
    
    st.info("**Message: ✅ Initial analysis complete.**")

    # --- SHOW ENERGY PERFORMANCE GRAPH ---
    st.markdown("#### Initial Model Performance")
    with st.expander("📈 View Energy Performance Graph", expanded=True):
        estimator = EnergyEstimator(model_type=model_type)
        results = estimator.estimate(prompt_text, layers, training_time, flops_input)
        fig = estimator.get_training_plot(layers)
        st.pyplot(fig)
    
    st.markdown("---")
    
    # =================================================================
    # PROMPT OPTIMIZATION ANALYSIS SECTION
    # =================================================================
    st.subheader("🧠 Prompt Optimization Analysis")
    st.markdown("*AI-powered prompt optimization using T5 transformer model*")
    
    # Get optimized results
    opt_res = st.session_state.get('optimized_results', {})
    
    # --- Energy & Carbon Metrics for Optimized Prompt ---
    st.markdown("#### ⚡ Optimized Prompt Energy Metrics")
    
    e1, e2, e3, e4 = st.columns(4)
    
    # Calculate actual savings
    orig_energy = float(res['energy_kwh'])
    opt_energy_val = float(opt_res.get('energy_kwh', orig_energy))
    energy_saved = orig_energy - opt_energy_val
    energy_saved_pct = (energy_saved / orig_energy * 100) if orig_energy > 0 else 0
    
    orig_carbon = float(res['carbon_kg'])
    opt_carbon_val = float(opt_res.get('carbon_kg', orig_carbon))
    carbon_saved = orig_carbon - opt_carbon_val
    
    e1.metric(
        "Optimized Energy", 
        f"{opt_energy_val:.4f} kWh",
        delta=f"-{energy_saved:.4f} kWh" if energy_saved > 0 else "No change",
        delta_color="inverse" if energy_saved > 0 else "off"
    )
    e2.metric(
        "Optimized Carbon", 
        f"{opt_carbon_val:.4f} kgCO2",
        delta=f"-{carbon_saved:.4f} kgCO2" if carbon_saved > 0 else "No change",
        delta_color="inverse" if carbon_saved > 0 else "off"
    )
    e3.metric(
        "Energy Saved", 
        f"{energy_saved_pct:.1f}%",
        help="Percentage of energy saved using the optimized prompt"
    )
    e4.metric(
        "Carbon Reduction", 
        f"{carbon_saved * 1000:.2f} g CO2",
        help="Grams of CO2 saved using the optimized prompt"
    )
    
    st.markdown("---")
    
    # --- Optimization Metrics Row ---
    m1, m2, m3, m4 = st.columns(4)
    
    token_reduction = opt.get('token_reduction_pct', 0)
    energy_reduction = opt.get('energy_reduction_pct', 0)
    semantic_sim = opt.get('semantic_similarity', 0)
    quality_score = opt.get('quality_score', 0)
    
    m1.metric(
        "Token Reduction", 
        f"{token_reduction}%",
        delta=f"-{opt.get('original_tokens', 0) - opt.get('optimized_tokens', 0)} tokens" if token_reduction > 0 else None,
        delta_color="inverse"
    )
    m2.metric(
        "Est. Energy Savings", 
        f"{energy_reduction}%",
        help="Based on transformer attention complexity (O(n²))"
    )
    m3.metric(
        "Semantic Similarity", 
        f"{semantic_sim}%",
        help="How well the optimized prompt preserves original meaning"
    )
    m4.metric(
        "Quality Score", 
        f"{quality_score}/100",
        help="Overall optimization quality rating"
    )
    
    # --- Prompt Comparison ---
    st.markdown("#### 📝 Prompt Comparison")
    col_orig, col_opt = st.columns(2)
    
    with col_orig:
        st.markdown("**Original Prompt**")
        st.code(st.session_state['prompt'], language="text")
        st.caption(f"Tokens: {opt.get('original_tokens', '-')}")
    
    with col_opt:
        st.markdown("**🌱 Optimized Prompt**")
        optimized_prompt = opt.get('optimized', st.session_state['prompt'])
        st.code(optimized_prompt, language="text")
        st.caption(f"Tokens: {opt.get('optimized_tokens', '-')}")
    
    # --- Semantic Validation ---
    st.markdown("#### 🔍 Semantic Validation")
    
    meaning_preserved = opt.get('meaning_preserved', True)
    similarity_interp = opt.get('similarity_interpretation', 'N/A')
    
    if meaning_preserved:
        st.success(f"✅ **Meaning Preserved:** {similarity_interp}")
    else:
        st.warning(f"⚠️ **Review Needed:** {similarity_interp}")
    
    # --- Visual: Similarity & Reduction Charts ---
    st.markdown("#### 📊 Optimization Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Token Reduction Chart
        fig_tokens = go.Figure()
        fig_tokens.add_trace(go.Bar(
            x=['Original', 'Optimized'],
            y=[opt.get('original_tokens', 0), opt.get('optimized_tokens', 0)],
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[opt.get('original_tokens', 0), opt.get('optimized_tokens', 0)],
            textposition='auto'
        ))
        fig_tokens.update_layout(
            title="Token Count Comparison",
            yaxis_title="Tokens",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig_tokens, use_container_width=True)
    
    with viz_col2:
        # Quality Metrics Gauge
        fig_quality = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=quality_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Optimization Quality"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4ECDC4"},
                'steps': [
                    {'range': [0, 40], 'color': "#FFE66D"},
                    {'range': [40, 70], 'color': "#95E1D3"},
                    {'range': [70, 100], 'color': "#4ECDC4"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig_quality.update_layout(height=300)
        st.plotly_chart(fig_quality, use_container_width=True)
    
    # --- Energy Comparison Visualization ---
    st.markdown("#### 📊 Energy Impact Visualization")
    
    energy_col1, energy_col2 = st.columns(2)
    
    with energy_col1:
        # Use actual energy values from estimator
        actual_orig_energy = float(res['energy_kwh'])
        actual_opt_energy = float(opt_res.get('energy_kwh', actual_orig_energy))
        actual_energy_saved = max(0, actual_orig_energy - actual_opt_energy)
        
        # Energy comparison pie
        fig_energy = go.Figure(data=[go.Pie(
            labels=['Energy Saved', 'Optimized Usage'],
            values=[actual_energy_saved, actual_opt_energy],
            hole=.4,
            marker_colors=['#4ECDC4', '#FF6B6B'],
            textinfo='label+percent',
            hovertemplate='%{label}: %{value:.4f} kWh<extra></extra>'
        )])
        fig_energy.update_layout(
            title="Energy Distribution",
            height=300,
            annotations=[dict(text=f'{actual_opt_energy:.4f}<br>kWh', x=0.5, y=0.5, font_size=12, showarrow=False)]
        )
        st.plotly_chart(fig_energy, use_container_width=True)
    
    with energy_col2:
        # Carbon footprint comparison using actual values
        actual_orig_carbon = float(res['carbon_kg'])
        actual_opt_carbon = float(opt_res.get('carbon_kg', actual_orig_carbon))
        
        fig_carbon = go.Figure()
        fig_carbon.add_trace(go.Bar(
            name='Original',
            x=['Energy (kWh)', 'Carbon (g CO2)'],
            y=[actual_orig_energy, actual_orig_carbon * 1000],
            marker_color='#FF6B6B',
            text=[f'{actual_orig_energy:.4f}', f'{actual_orig_carbon * 1000:.2f}'],
            textposition='auto'
        ))
        fig_carbon.add_trace(go.Bar(
            name='Optimized',
            x=['Energy (kWh)', 'Carbon (g CO2)'],
            y=[actual_opt_energy, actual_opt_carbon * 1000],
            marker_color='#4ECDC4',
            text=[f'{actual_opt_energy:.4f}', f'{actual_opt_carbon * 1000:.2f}'],
            textposition='auto'
        ))
        fig_carbon.update_layout(
            title="Energy & Carbon: Original vs Optimized",
            barmode='group',
            height=300,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig_carbon, use_container_width=True)
    
    # --- Suggestions ---
    suggestions = opt.get('suggestions', [])
    if suggestions:
        st.markdown("#### 💡 Suggestions")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")
    
    st.markdown("---")

# --- VIEW 1: ADDITIONAL OPTIMIZATION BUTTON ---
if st.session_state['step'] == 1:
    st.write("👉 **Want full architecture optimization?** Click below for complete analysis.")
    
    if st.button("✨ Full Architecture Optimization"):
        with st.spinner("Optimizing architecture parameters..."):
            # 1. Get optimized prompt
            opt = st.session_state.get('optimization_analysis', {})
            better_prompt = opt.get('optimized', st.session_state['prompt'])
            
            # 2. Estimate with "Improved" architecture inputs
            improved_layers = int(layers * 0.8) if layers > 1 else 1
            improved_time = training_time * 0.8
            
            estimator = EnergyEstimator(model_type=model_type)
            new_results = estimator.estimate(better_prompt, improved_layers, improved_time, flops_input)
            
            st.session_state['improved_results'] = new_results
            st.session_state['better_prompt'] = better_prompt
            st.session_state['step'] = 2
            
        st.session_state['toast_message'] = ('Full Optimization Complete! 🌱', '🌱')
        st.rerun()

# --- VIEW 2: FULL COMPARISON RESULT ---
if st.session_state['step'] == 2:
    st.subheader("🏆 Full Optimization Results")
    st.success("**System Status:** ✅ Fully Optimized (Prompt + Architecture)")
    
    orig = st.session_state['original_results']
    new = st.session_state['improved_results']
    opt = st.session_state.get('optimization_analysis', {})
    
    saved_energy = orig['energy_kwh'] - new['energy_kwh']
    percent_saved = (saved_energy / orig['energy_kwh']) * 100 if orig['energy_kwh'] > 0 else 0
    
    # Combined metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Final Energy Usage", 
        f"{new['energy_kwh']} kWh", 
        delta=f"-{round(saved_energy, 2)} kWh", 
        delta_color="inverse"
    )
    c2.metric(
        "Final Carbon", 
        f"{new['carbon_kg']} kgCO2", 
        delta=f"-{round(orig['carbon_kg'] - new['carbon_kg'], 2)}", 
        delta_color="inverse"
    )
    c3.metric(
        "Total Efficiency Gain", 
        f"{round(percent_saved, 1)}%"
    )
    c4.metric(
        "Prompt Efficiency", 
        f"{opt.get('token_reduction_pct', 0)}%"
    )

    # Side-by-side prompt comparison
    st.write("#### Prompt Optimization Summary")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.warning("**Original Prompt**")
        st.code(st.session_state['prompt'], language="text")

    with col_b:
        st.success("**Optimized Prompt**")
        st.code(st.session_state['better_prompt'], language="text")
    
    # Final comparison visualization
    st.markdown("#### 📊 Before vs After Comparison")
    
    comparison_data = {
        'Metric': ['Energy (kWh)', 'Carbon (kg CO2)', 'Token Count'],
        'Original': [orig['energy_kwh'], orig['carbon_kg'], opt.get('original_tokens', orig['token_count'])],
        'Optimized': [new['energy_kwh'], new['carbon_kg'], opt.get('optimized_tokens', new['token_count'])]
    }
    
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Bar(
        name='Original',
        x=comparison_data['Metric'],
        y=comparison_data['Original'],
        marker_color='#FF6B6B'
    ))
    fig_comparison.add_trace(go.Bar(
        name='Optimized',
        x=comparison_data['Metric'],
        y=comparison_data['Optimized'],
        marker_color='#4ECDC4'
    ))
    fig_comparison.update_layout(
        title="Full Optimization Impact",
        barmode='group',
        height=400
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
        
    # --- SHOW OPTIMIZED MODEL PERFORMANCE GRAPH ---
    st.markdown("#### Optimized Model Performance")
    
    with st.expander("📉 View Optimized Performance Graph", expanded=True):
        estimator = EnergyEstimator(model_type=model_type)
        results = estimator.estimate(st.session_state['better_prompt'], improved_layers, improved_time, flops_input)
        fig = estimator.get_training_plot(improved_layers)
        st.pyplot(fig)
    
    # Reset button
    if st.button("🔄 Reset Analysis"):
        st.session_state['step'] = 0
        st.session_state['original_results'] = {}
        st.session_state['improved_results'] = {}
        st.session_state['optimization_analysis'] = {}
        st.session_state['toast_message'] = ('System Reset.', '🔄')
        st.rerun()

# --- FOOTER ---
st.markdown("---")
st.caption("🌱 Sustainable AI - Transparency & Energy-Efficient Prompt Engineering with Machine Learning")
st.caption("NLP Module: T5-based prompt optimization with semantic similarity validation")