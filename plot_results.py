"""
Impeller Flow Simulator - Results Visualization Module

This module provides comprehensive visualization tools for analyzing impeller simulation results.
It generates:
- Efficiency vs flow rate plots
- Convergence analysis visualizations
- Performance heatmaps
- Interactive dashboards

The visualizations are designed to help understand the performance characteristics
and optimization results of the impeller simulations.

Author: John Benac
Date: January 2024
License: MIT
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_results(results_file):
    with open(results_file, 'r') as f:
        return json.load(f)

def create_efficiency_flow_plot(results):
    plt.figure(figsize=(12, 6))
    flow_rates = []
    efficiencies = []
    categories = []
    
    for case_name, case_data in results.items():
        for flow_rate, result in case_data['results'].items():
            if result['metrics'] is not None:
                flow_rates.append(float(flow_rate))
                efficiencies.append(result['metrics']['efficiency']['overall'] * 100)
                categories.append(case_name)
    
    sns.scatterplot(x=flow_rates, y=efficiencies, hue=categories, s=100)
    plt.plot(flow_rates, efficiencies, '--', alpha=0.3)
    plt.xlabel('Flow Rate (m³/s)')
    plt.ylabel('Efficiency (%)')
    plt.title('Efficiency vs Flow Rate')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/efficiency_flow_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_convergence_plot(results):
    plt.figure(figsize=(12, 6))
    for case_name, case_data in results.items():
        if 'analysis' in case_data and 'convergence_rate' in case_data['analysis']:
            flow_rates = []
            improvements = []
            for flow_rate, conv_data in case_data['analysis']['convergence_rate'].items():
                flow_rates.append(float(flow_rate))
                improvements.append(conv_data['improvement'])
            plt.plot(flow_rates, improvements, 'o-', label=case_name)
    
    plt.xlabel('Flow Rate (m³/s)')
    plt.ylabel('Improvement (%)')
    plt.title('Optimization Improvement by Flow Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/convergence_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_heatmap(results):
    flow_rates = []
    efficiencies = []
    errors = []
    
    for case_data in results.values():
        for flow_rate, result in case_data['results'].items():
            if result['metrics'] is not None:
                flow_rates.append(float(flow_rate))
                efficiencies.append(result['metrics']['efficiency']['overall'] * 100)
                errors.append(result['error'])
    
    # Create 2D grid for heatmap
    x = np.linspace(min(flow_rates), max(flow_rates), 50)
    y = np.linspace(min(efficiencies), max(efficiencies), 50)
    X, Y = np.meshgrid(x, y)
    
    # Create heatmap data
    Z = np.zeros_like(X)
    for i in range(len(flow_rates)):
        Z += np.exp(-((X - flow_rates[i])**2 + (Y - efficiencies[i])**2) / 100)
    
    plt.figure(figsize=(12, 8))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Performance Density')
    plt.scatter(flow_rates, efficiencies, c='red', s=100, label='Test Points')
    plt.xlabel('Flow Rate (m³/s)')
    plt.ylabel('Efficiency (%)')
    plt.title('Performance Heatmap')
    plt.legend()
    plt.savefig('results/performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_map(results):
    plt.figure(figsize=(12, 8))
    
    # Extract flow coefficients, head coefficients, and efficiencies
    flow_coeffs = []
    head_coeffs = []
    efficiencies = []
    
    for case_data in results.values():
        for result in case_data['results'].values():
            if result['metrics'] is not None:
                flow_coeffs.append(result['metrics']['dimensionless']['flow_coefficient'])
                head_coeffs.append(result['metrics']['dimensionless']['head_coefficient'])
                efficiencies.append(result['metrics']['efficiency']['overall'] * 100)
    
    # Add small random noise to prevent identical coordinates
    flow_coeffs = np.array(flow_coeffs)
    head_coeffs = np.array(head_coeffs)
    flow_coeffs += np.random.normal(0, 0.0001, size=len(flow_coeffs))
    head_coeffs += np.random.normal(0, 0.0001, size=len(head_coeffs))
    
    # Create scatter plot of operating points
    scatter = plt.scatter(flow_coeffs, head_coeffs, c=efficiencies, 
                         cmap='viridis', s=100, label='Operating Points')
    plt.colorbar(scatter, label='Efficiency (%)')
    
    # Find and mark best efficiency point
    best_idx = np.argmax(efficiencies)
    plt.scatter(flow_coeffs[best_idx], head_coeffs[best_idx], c='white', s=200, 
                marker='*', label='Best Efficiency Point')
    
    plt.xlabel('Flow Coefficient (φ)')
    plt.ylabel('Head Coefficient (ψ)')
    plt.title('Impeller Performance Map')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/performance_map.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_optimization_progress(results):
    plt.figure(figsize=(12, 8))
    
    # Create subplots for different metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for case_name, case_data in results.items():
        # Plot error progression
        flow_rates = []
        errors = []
        for flow_rate, result in case_data['results'].items():
            flow_rates.append(float(flow_rate))
            errors.append(result['error'])
        
        ax1.plot(flow_rates, errors, '-o', label=case_name)
        
        # Plot efficiency progression
        efficiencies = [result['metrics']['efficiency']['overall'] * 100 
                       for result in case_data['results'].values()]
        ax2.plot(flow_rates, efficiencies, '-o', label=case_name)
    
    # Customize subplots
    ax1.set_title('Error vs Flow Rate')
    ax1.set_xlabel('Flow Rate (m³/s)')
    ax1.set_ylabel('Error (%)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_title('Efficiency vs Flow Rate')
    ax2.set_xlabel('Flow Rate (m³/s)')
    ax2.set_ylabel('Efficiency (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results/optimization_progress.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_blade_loading_plot(results):
    plt.figure(figsize=(12, 8))
    
    # Create subplots for pressure and velocity distributions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    has_blade_data = False
    for case_name, case_data in results.items():
        # Get the last result for each case
        last_result = list(case_data['results'].values())[-1]
        
        if 'blade_data' in last_result and last_result['blade_data'] is not None:
            has_blade_data = True
            # Plot pressure distribution
            meridional_pos = last_result['blade_data']['meridional_pos']
            ax1.plot(meridional_pos, last_result['blade_data']['pressure'], 
                    label=f'{case_name} - Pressure Side')
            ax1.plot(meridional_pos, last_result['blade_data']['suction'], 
                    '--', label=f'{case_name} - Suction Side')
            
            # Plot velocity distribution
            ax2.plot(meridional_pos, last_result['blade_data']['velocity'], 
                    label=case_name)
    
    if not has_blade_data:
        plt.close()
        return
    
    ax1.set_title('Blade Surface Pressure Distribution')
    ax1.set_xlabel('Meridional Position')
    ax1.set_ylabel('Pressure (Pa)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_title('Velocity Distribution Along Blade')
    ax2.set_xlabel('Meridional Position')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results/blade_loading.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_interactive_analysis(results):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance Map', 'Error vs Flow Rate',
                       'Efficiency vs Flow Rate', 'Velocity Triangles')
    )
    
    # Add performance map
    flow_coeffs = []
    head_coeffs = []
    efficiencies = []
    
    for case_data in results.values():
        for result in case_data['results'].values():
            if result['metrics'] is not None:
                flow_coeffs.append(result['metrics']['dimensionless']['flow_coefficient'])
                head_coeffs.append(result['metrics']['dimensionless']['head_coefficient'])
                efficiencies.append(result['metrics']['efficiency']['overall'] * 100)
    
    fig.add_trace(
        go.Scatter(
            x=flow_coeffs,
            y=head_coeffs,
            mode='markers',
            marker=dict(
                size=10,
                color=efficiencies,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Efficiency (%)')
            ),
            text=[f'Efficiency: {e:.1f}%' for e in efficiencies],
            name='Operating Points'
        ),
        row=1, col=1
    )
    
    # Add error vs flow rate
    for case_name, case_data in results.items():
        flow_rates = []
        errors = []
        for flow_rate, result in case_data['results'].items():
            flow_rates.append(float(flow_rate))
            errors.append(result['error'])
        
        fig.add_trace(
            go.Scatter(
                x=flow_rates,
                y=errors,
                mode='lines+markers',
                name=f'{case_name} - Error'
            ),
            row=1, col=2
        )
    
    # Add efficiency vs flow rate
    for case_name, case_data in results.items():
        flow_rates = []
        efficiencies = []
        for flow_rate, result in case_data['results'].items():
            flow_rates.append(float(flow_rate))
            efficiencies.append(result['metrics']['efficiency']['overall'] * 100)
        
        fig.add_trace(
            go.Scatter(
                x=flow_rates,
                y=efficiencies,
                mode='lines+markers',
                name=f'{case_name} - Efficiency'
            ),
            row=2, col=1
        )
    
    # Add velocity triangles placeholder
    # This would be implemented with actual velocity triangle calculations
    
    # Update layout
    fig.update_xaxes(title_text='Flow Coefficient (φ)', row=1, col=1)
    fig.update_yaxes(title_text='Head Coefficient (ψ)', row=1, col=1)
    
    fig.update_xaxes(title_text='Flow Rate (m³/s)', row=1, col=2)
    fig.update_yaxes(title_text='Error (%)', row=1, col=2)
    
    fig.update_xaxes(title_text='Flow Rate (m³/s)', row=2, col=1)
    fig.update_yaxes(title_text='Efficiency (%)', row=2, col=1)
    
    fig.update_layout(
        height=800,
        width=1200,
        title_text='Comprehensive Impeller Analysis',
        showlegend=True
    )
    
    fig.write_html('results/interactive_analysis.html')

def main():
    # Ensure results directory exists
    Path('results').mkdir(exist_ok=True)
    
    # Find the most recent results file
    results_dir = Path('results')
    results_files = list(results_dir.glob('comprehensive_results_*.json'))
    if not results_files:
        print("No results files found!")
        return
    
    latest_file = max(results_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    # Load results
    results = load_results(latest_file)
    
    # Generate all plots
    print("Generating efficiency vs flow rate plot...")
    create_efficiency_flow_plot(results)
    
    print("Generating convergence plot...")
    create_convergence_plot(results)
    
    print("Generating performance heatmap...")
    create_performance_heatmap(results)
    
    print("Generating performance map...")
    create_performance_map(results)
    
    print("Generating optimization progress plots...")
    create_optimization_progress(results)
    
    print("Generating blade loading analysis...")
    create_blade_loading_plot(results)
    
    print("Generating interactive analysis...")
    create_interactive_analysis(results)
    
    print("\nAll visualizations have been generated in the 'results' directory:")
    print("1. efficiency_flow_plot.png")
    print("2. convergence_plot.png")
    print("3. performance_heatmap.png")
    print("4. performance_map.png")
    print("5. optimization_progress.png")
    print("6. blade_loading.png")
    print("7. interactive_analysis.html")

if __name__ == "__main__":
    main() 