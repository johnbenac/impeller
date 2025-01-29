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
                efficiencies.append(result['metrics']['efficiency'] * 100)
                categories.append(case_name)
    
    sns.scatterplot(x=flow_rates, y=efficiencies, hue=categories, s=100)
    plt.plot(flow_rates, efficiencies, '--', alpha=0.3)
    plt.xlabel('Flow Rate (m/s)')
    plt.ylabel('Efficiency (%)')
    plt.title('Efficiency vs Flow Rate')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/efficiency_flow_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_convergence_plot(results):
    plt.figure(figsize=(12, 6))
    for case_name, case_data in results.items():
        flow_rates = []
        improvements = []
        for flow_rate, conv_data in case_data['analysis']['convergence_rate'].items():
            flow_rates.append(float(flow_rate))
            improvements.append(conv_data['improvement'])
        plt.plot(flow_rates, improvements, 'o-', label=case_name)
    
    plt.xlabel('Flow Rate (m/s)')
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
                efficiencies.append(result['metrics']['efficiency'] * 100)
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
    plt.xlabel('Flow Rate (m/s)')
    plt.ylabel('Efficiency (%)')
    plt.title('Performance Heatmap')
    plt.legend()
    plt.savefig('results/performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_interactive_plot(results):
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Efficiency vs Flow Rate',
                                      'Optimization Improvement',
                                      'Error Distribution',
                                      'Performance Overview'))
    
    # Efficiency vs Flow Rate
    flow_rates = []
    efficiencies = []
    categories = []
    for case_name, case_data in results.items():
        case_flows = []
        case_effs = []
        for flow_rate, result in case_data['results'].items():
            if result['metrics'] is not None:
                case_flows.append(float(flow_rate))
                case_effs.append(result['metrics']['efficiency'] * 100)
        fig.add_trace(
            go.Scatter(x=case_flows, y=case_effs, name=case_name,
                      mode='lines+markers'),
            row=1, col=1
        )
    
    # Optimization Improvement
    for case_name, case_data in results.items():
        flow_rates = []
        improvements = []
        for flow_rate, conv_data in case_data['analysis']['convergence_rate'].items():
            flow_rates.append(float(flow_rate))
            improvements.append(conv_data['improvement'])
        fig.add_trace(
            go.Scatter(x=flow_rates, y=improvements, name=case_name,
                      mode='lines+markers'),
            row=1, col=2
        )
    
    # Error Distribution
    all_errors = []
    for case_data in results.values():
        for result in case_data['results'].values():
            all_errors.append(result['error'])
    fig.add_trace(
        go.Histogram(x=all_errors, name='Error Distribution'),
        row=2, col=1
    )
    
    # Performance Overview
    performance_scores = []
    case_names = []
    for case_name, case_data in results.items():
        performance_scores.append(case_data['analysis']['efficiency_stats']['mean'] * 100)
        case_names.append(case_name)
    fig.add_trace(
        go.Bar(x=case_names, y=performance_scores, name='Average Efficiency'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, width=1200, title_text="Comprehensive Performance Analysis")
    fig.write_html("results/interactive_analysis.html")

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
    
    print("Generating interactive analysis...")
    create_interactive_plot(results)
    
    print("\nAll visualizations have been generated in the 'results' directory:")
    print("1. efficiency_flow_plot.png")
    print("2. convergence_plot.png")
    print("3. performance_heatmap.png")
    print("4. interactive_analysis.html")

if __name__ == "__main__":
    main() 