import streamlit as st
import matplotlib.pyplot as plt
from impeller_sim import ImpellerParams, ImpellerSimulation
from auto_runner import AutomatedRunner
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime

st.set_page_config(page_title="Impeller Flow Simulator", layout="wide")

st.title("Impeller Flow Simulator")
st.markdown("""
This advanced simulator provides real-time visualization and optimization of fluid flow through a parameterized impeller.
The simulation includes physical constraints, automated parameter optimization, and comprehensive performance metrics.
""")

# Initialize session state
if 'runner' not in st.session_state:
    st.session_state.runner = AutomatedRunner()
if 'optimization_running' not in st.session_state:
    st.session_state.optimization_running = False
if 'latest_results' not in st.session_state:
    st.session_state.latest_results = None
if 'latest_analysis' not in st.session_state:
    st.session_state.latest_analysis = None

# Sidebar for automated optimization
st.sidebar.header("Automated Optimization")

# Optimization controls
target_flow_min = st.sidebar.number_input("Min Flow Rate (m/s)", 1.0, 20.0, 5.0)
target_flow_max = st.sidebar.number_input("Max Flow Rate (m/s)", 5.0, 50.0, 30.0)
num_points = st.sidebar.number_input("Number of Flow Points", 2, 10, 6, step=1)
max_iterations = st.sidebar.number_input("Max Iterations per Point", 5, 20, 10, step=1)

if st.sidebar.button("Start Automated Optimization"):
    st.session_state.optimization_running = True
    target_flows = np.linspace(target_flow_min, target_flow_max, num_points)
    
    with st.spinner("Running automated optimization..."):
        results = st.session_state.runner.run_optimization_cycle(
            target_flow_rates=target_flows,
            max_iterations=max_iterations
        )
        analysis = st.session_state.runner.analyze_results()
        
        st.session_state.latest_results = results
        st.session_state.latest_analysis = analysis
        st.session_state.optimization_running = False

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Optimization Results")
    
    if st.session_state.latest_results:
        # Create convergence plot
        fig = go.Figure()
        
        for target_flow, result in st.session_state.latest_results.items():
            flow_entries = [e for e in st.session_state.runner.history 
                          if e['target_flow'] == float(target_flow)]
            errors = [e['error'] for e in flow_entries]
            iterations = list(range(len(errors)))
            
            fig.add_trace(go.Scatter(
                x=iterations,
                y=errors,
                mode='lines+markers',
                name=f'Flow Rate {float(target_flow):.1f} m/s'
            ))
        
        fig.update_layout(
            title="Optimization Convergence",
            xaxis_title="Iteration",
            yaxis_title="Error",
            yaxis_type="log"
        )
        
        st.plotly_chart(fig)
        
        # Show best solution visualization
        if st.session_state.runner.best_params:
            st.subheader("Best Solution Visualization")
            sim = ImpellerSimulation(st.session_state.runner.best_params)
            
            viz_options = st.radio(
                "Visualization Type",
                ["2D Vector Field", "Velocity Magnitude Contour"]
            )
            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            if viz_options == "2D Vector Field":
                sim.visualize_flow(ax)
            else:
                v_x, v_y = sim.calculate_velocity_field()
                v_mag = np.sqrt(v_x**2 + v_y**2)
                contour = ax.contourf(sim.X, sim.Y, v_mag, levels=20, cmap='viridis')
                plt.colorbar(contour, ax=ax, label='Velocity magnitude (m/s)')
                sim.visualize_flow(ax)
                
            st.pyplot(fig)

with col2:
    st.subheader("Analysis Results")
    
    if st.session_state.latest_analysis:
        # Display efficiency statistics
        st.markdown("### Efficiency Statistics")
        eff_stats = st.session_state.latest_analysis['efficiency_stats']
        st.metric("Maximum Efficiency", f"{eff_stats['max']*100:.1f}%")
        st.metric("Average Efficiency", f"{eff_stats['mean']*100:.1f}%")
        st.metric("Efficiency Std Dev", f"{eff_stats['std']*100:.2f}%")
        
        # Display convergence statistics
        st.markdown("### Convergence Analysis")
        for flow_rate, conv_data in st.session_state.latest_analysis['convergence_rate'].items():
            st.markdown(f"**Flow Rate {float(flow_rate):.1f} m/s**")
            st.markdown(f"- Improvement: {conv_data['improvement']:.1f}%")
            st.markdown(f"- Final Error: {conv_data['final_error']:.4f}")
        
        # Show optimization history
        if os.path.exists("optimization_results_latest.json"):
            with open("optimization_results_latest.json", "r") as f:
                history = json.load(f)
            
            st.markdown("### Detailed Results")
            st.json(history)

# Documentation
with st.expander("Documentation & Analysis"):
    st.markdown("""
    ### Automated Optimization Process
    
    1. **Multi-point Optimization**
       - Optimizes for multiple target flow rates
       - Adapts parameters based on performance metrics
       - Ensures stable convergence
    
    2. **Performance Analysis**
       - Tracks convergence rates
       - Monitors efficiency trends
       - Validates physical constraints
    
    3. **Results Storage**
       - Saves optimization history
       - Generates performance reports
       - Enables trend analysis
    
    ### Limitations & Considerations
    
    - Optimization may not converge for all flow rates
    - Physical constraints may limit achievable performance
    - Results are based on simplified flow models
    """)

# Footer with status
st.sidebar.markdown("---")
if st.session_state.optimization_running:
    st.sidebar.warning("Optimization in progress...")
elif st.session_state.latest_results:
    st.sidebar.success("Optimization complete")
else:
    st.sidebar.info("Ready to start optimization")

st.sidebar.markdown("""
### About
This simulator provides a simplified model of fluid flow through an impeller.
The simulation uses basic fluid dynamics principles and is intended for
educational and preliminary design purposes.
""") 