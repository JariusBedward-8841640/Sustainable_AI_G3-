import streamlit as st
import plotly.graph_objects as go

# -------------------------------------
# SIDEBAR INPUTS
# -------------------------------------
def sidebar_inputs():
    with st.sidebar:
        st.header("Model Architecture")
        layers = st.number_input("Number of Layers", min_value=1, max_value=200, value=12)
        training_time = st.number_input("Training Time (Hours)", min_value=0.1, value=5.0)
        flops_input = st.text_input("FLOPs (e.g. 1.5e18)", value="1.5e18")

        model_type = st.selectbox(
            "Select Model Type",
            options=["RandomForest", "LinearRegression"],
            index=1
        )

        st.markdown("---")
        st.info(f"ℹ️ Uses {model_type} on synthetic training data.")
        st.markdown("---")
        st.markdown("#### 🧠 NLP Optimization")
        st.caption("T5-based prompt optimizer with semantic similarity validation.")

        return model_type, layers, training_time, flops_input


# -------------------------------------
# PROMPT INPUT VIEW
# -------------------------------------
def input_prompt_view():
    st.subheader("Enter Prompt Context")
    return st.text_area("Input Prompt:", height=100, placeholder="Enter your prompt here...")

# -------------------------------------
# INITIAL ANALYSIS METRICS
# -------------------------------------
def render_initial_analysis(original, optimization):
    st.divider()
    st.subheader("📊 Analysis Report")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Energy", f"{original['energy_kwh']} kWh")
    col2.metric("Carbon Footprint", f"{original['carbon_kg']} kgCO2")
    col3.metric("Original Tokens", optimization.get('original_tokens', original['token_count']))
    col4.metric("Optimized Tokens", optimization.get('optimized_tokens', '-'))

    st.info("**Message: ✅ Initial analysis complete.**")


# -------------------------------------
# OPTIMIZATION METRICS (energy, carbon, tokens)
# -------------------------------------
def optimization_metrics(res_optimized, opt):
    st.subheader("🧠 Prompt Optimization Analysis")
    st.markdown("*AI-powered prompt optimization using T5 transformer model*")      

    # --- Energy & Carbon Metrics for Optimized Prompt ---
    st.markdown("#### ⚡ Optimized Prompt Energy Metrics")

    e1, e2, e3, e4 = st.columns(4)

    # Calculate actual savings
    orig_energy = float(st.session_state['original_results']['energy_kwh'])
    opt_energy_val = float(res_optimized.get('energy_kwh', orig_energy))
    energy_saved = orig_energy - opt_energy_val
    energy_saved_pct = (energy_saved / orig_energy * 100) if orig_energy > 0 else 0

    orig_carbon = float(st.session_state['original_results']['carbon_kg'])
    opt_carbon_val = float(res_optimized.get('carbon_kg', orig_carbon))
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

def semantic_validation(opt):
    st.markdown("#### 🔍 Semantic Validation")

    meaning_preserved = opt.get('meaning_preserved', True)
    similarity_interp = opt.get('similarity_interpretation', 'N/A')

    if meaning_preserved:
        st.success(f"✅ **Meaning Preserved:** {similarity_interp}")
    else:
        st.warning(f"⚠️ **Review Needed:** {similarity_interp}")

# -------------------------------------
# TOKEN REDUCTION BAR CHART
# -------------------------------------
def token_reduction_plot(opt_data):
    st.markdown("#### 📊 Optimization Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        fig_tokens = go.Figure()
        fig_tokens.add_trace(go.Bar(
            x=['Original', 'Optimized'],
            y=[opt_data.get('original_tokens', 0), opt_data.get('optimized_tokens', 0)],
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[opt_data.get('original_tokens', 0), opt_data.get('optimized_tokens', 0)],
            textposition='auto'
        ))
        fig_tokens.update_layout(
            title="Token Count Comparison",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_tokens, width='stretch')
    with col2:
        quality_score = opt_data.get('quality_score', 0)
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
        st.plotly_chart(fig_quality, width='stretch')

# -------------------------------------
# ENERGY PIE + CARBON BAR CHART
# -------------------------------------
def energy_impact_visualization(res_optimized):
    st.markdown("#### 📊 Energy Impact Visualization")

    energy_col1, energy_col2 = st.columns(2)

    with energy_col1:
        # Use actual energy values from estimator
        actual_orig_energy = float(st.session_state['original_results']['energy_kwh'])
        actual_opt_energy = float(res_optimized.get('energy_kwh', actual_orig_energy))
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
        st.plotly_chart(fig_energy, width='stretch')

    with energy_col2:
        # Carbon footprint comparison using actual values
        actual_orig_carbon = float(st.session_state['original_results']['carbon_kg'])
        actual_opt_carbon = float(res_optimized.get('carbon_kg', actual_orig_carbon))

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
        st.plotly_chart(fig_carbon, width='stretch')

# -------------------------------------
# PROMPT COMPARISON
# -------------------------------------
def prompt_comparison(original_prompt, opt_data):
    st.markdown("#### 📝 Prompt Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Prompt**")
        st.code(original_prompt, language="text")

    with col2:
        st.markdown("**🌱 Optimized Prompt**")
        st.code(opt_data.get("optimized", original_prompt), language="text")

# -------------------------------------
# FOOTER
# -------------------------------------
def footer():
    st.markdown("---")
    st.caption("🌱 Sustainable AI - Transparency & Energy-Efficient Prompt Engineering")
    st.caption("NLP Module: T5-based prompt optimization with semantic similarity validation")
