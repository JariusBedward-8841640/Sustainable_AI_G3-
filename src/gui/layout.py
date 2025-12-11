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
# TRAINING PERFORMANCE PLOT
# -------------------------------------
def render_training_plot(estimator, layers):
    st.markdown("#### Initial Model Performance")
    with st.expander("📈 View Energy Performance Graph", expanded=True):
        fig = estimator.get_training_plot(layers)
        st.pyplot(fig)


# -------------------------------------
# OPTIMIZATION METRICS (energy, carbon, tokens)
# -------------------------------------
def optimization_metrics(original, opt_results, opt_data):
    st.subheader("🧠 Prompt Optimization Analysis")
    st.markdown("*AI-powered prompt optimization using T5 transformer model*")

    orig_e = float(original['energy_kwh'])
    opt_e = float(opt_results["energy_kwh"])
    saved = orig_e - opt_e
    saved_pct = saved / orig_e * 100 if orig_e else 0

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Optimized Energy", f"{opt_e:.4f} kWh")
    e2.metric("Carbon Reduction", f"{opt_results['carbon_kg']:.4f} kgCO2")
    e3.metric("Energy Saved", f"{saved_pct:.1f}%")
    e4.metric("Token Reduction", f"{opt_data.get('token_reduction_pct', 0)}%")


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
        st.plotly_chart(fig_tokens, use_container_width=True)

    return col2  # return the right column to draw second chart there


# -------------------------------------
# QUALITY GAUGE PLOT
# -------------------------------------
def quality_gauge_plot(col, quality_score):
    with col:
        fig_quality = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=quality_score,
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4ECDC4"},
                'steps': [
                    {'range': [0, 40], 'color': "#FFE66D"},
                    {'range': [40, 70], 'color': "#95E1D3"},
                    {'range': [70, 100], 'color': "#4ECDC4"}
                ]
            }
        ))
        fig_quality.update_layout(height=300)
        st.plotly_chart(fig_quality, use_container_width=True)


# -------------------------------------
# ENERGY PIE + CARBON BAR CHART
# -------------------------------------
def energy_impact_visualization(original, opt_results):
    st.markdown("#### 📊 Energy Impact Visualization")

    col1, col2 = st.columns(2)

    # pie chart
    with col1:
        orig_e = float(original['energy_kwh'])
        opt_e = float(opt_results['energy_kwh'])
        saved = max(0, orig_e - opt_e)

        fig_energy = go.Figure(data=[go.Pie(
            labels=['Energy Saved', 'Optimized Usage'],
            values=[saved, opt_e],
            hole=.4,
            textinfo='label+percent'
        )])
        fig_energy.update_layout(title="Energy Distribution", height=300)
        st.plotly_chart(fig_energy, use_container_width=True)

    # carbon bar chart
    with col2:
        orig_c = float(original['carbon_kg'])
        opt_c = float(opt_results['carbon_kg'])

        fig_carbon = go.Figure()
        fig_carbon.add_trace(go.Bar(
            name='Original',
            x=['Energy (kWh)', 'Carbon (g CO2)'],
            y=[orig_e, orig_c * 1000],
            marker_color='#FF6B6B'
        ))
        fig_carbon.add_trace(go.Bar(
            name='Optimized',
            x=['Energy (kWh)', 'Carbon (g CO2)'],
            y=[opt_e, opt_c * 1000],
            marker_color='#4ECDC4'
        ))
        fig_carbon.update_layout(title="Energy & Carbon: Original vs Optimized", height=300, barmode="group")
        st.plotly_chart(fig_carbon, use_container_width=True)


# -------------------------------------
# PROMPT COMPARISON
# -------------------------------------
def prompt_comparison(original_prompt, opt_data):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Prompt**")
        st.code(original_prompt, language="text")

    with col2:
        st.markdown("**🌱 Optimized Prompt**")
        st.code(opt_data.get("optimized", original_prompt), language="text")

# -------------------------------------
# RESET BUTTON
# -------------------------------------
def reset_button_function():
    st.session_state['step'] = 0
    st.session_state['original_results'] = {}
    st.session_state['optimized_results'] = {}
    st.session_state['optimization_analysis'] = {}
    st.session_state['toast_message'] = ('System Reset.', '🔄')
    st.rerun()


# -------------------------------------
# FOOTER
# -------------------------------------
def footer():
    st.markdown("---")
    st.caption("🌱 Sustainable AI - Transparency & Energy-Efficient Prompt Engineering")
    st.caption("NLP Module: T5-based prompt optimization with semantic similarity validation")
