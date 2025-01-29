"""
Automated test runner for the Impeller Flow Simulator.
This script runs a series of test cases and generates comprehensive results.
"""

import logging
import json
from datetime import datetime
import yaml
from pathlib import Path
import numpy as np
from impeller_sim import ImpellerParams, ImpellerSimulation
from plot_results import (create_efficiency_flow_plot, create_convergence_plot,
                         create_performance_heatmap, create_performance_map,
                         create_optimization_progress, create_blade_loading_plot,
                         create_interactive_analysis)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AutomatedTests')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        return super().default(obj)

def load_config():
    """Load test configuration from YAML file."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def run_optimization_cycle(flow_rates, base_params):
    """
    Run optimization for given flow rates.
    
    Args:
        flow_rates: List of flow rates to optimize for
        base_params: Base impeller parameters
        
    Returns:
        Dictionary containing results and analysis
    """
    results = {}
    analysis = {
        'efficiency_stats': {'max': 0.0, 'min': 1.0, 'mean': 0.0},
        'convergence_rate': {}
    }
    
    logger.info(f"Starting optimization cycle for {len(flow_rates)} flow rates\n")
    
    for flow_rate in flow_rates:
        logger.info(f"Optimizing for target flow rate: {flow_rate} m³/s")
        
        # Create simulation with base parameters
        sim = ImpellerSimulation(base_params)
        
        # Calculate initial performance
        sim.calculate_velocity_field()
        initial_metrics = sim.calculate_performance_metrics()
        initial_flow = initial_metrics['flow']['volumetric_flow']
        initial_error = abs(initial_flow - flow_rate) / flow_rate * 100
        initial_efficiency = initial_metrics['efficiency']['overall'] * 100
        
        logger.info(f"Initial error: {initial_error:.4f}%")
        logger.info(f"Initial efficiency: {initial_efficiency:.1f}%")
        
        try:
            # Optimize parameters
            optimized_params = sim.optimize_parameters(flow_rate)
            
            # Calculate final performance
            final_sim = ImpellerSimulation(optimized_params)
            final_sim.calculate_velocity_field()
            final_metrics = final_sim.calculate_performance_metrics()
            
            # Calculate error and improvement
            final_flow = final_metrics['flow']['volumetric_flow']
            final_error = abs(final_flow - flow_rate) / flow_rate * 100
            improvement = initial_error - final_error
            
            # Store results
            results[str(flow_rate)] = {
                'error': final_error,
                'metrics': final_metrics,
                'parameters': {
                    'num_blades': optimized_params.num_blades,
                    'blade_angle_inlet': optimized_params.blade_angle_inlet,
                    'blade_angle_outlet': optimized_params.blade_angle_outlet,
                    'rotational_speed': optimized_params.rotational_speed
                }
            }
            
            # Update analysis
            analysis['convergence_rate'][str(flow_rate)] = {
                'initial_error': initial_error,
                'final_error': final_error,
                'improvement': improvement
            }
            
            efficiency = final_metrics['efficiency']['overall']
            analysis['efficiency_stats']['max'] = max(analysis['efficiency_stats']['max'], efficiency)
            analysis['efficiency_stats']['min'] = min(analysis['efficiency_stats']['min'], efficiency)
            
            logger.info(f"Optimization complete for flow rate {flow_rate} m³/s")
            logger.info(f"Best error achieved: {final_error:.4f}%")
            
        except Exception as e:
            logger.error(f"Optimization failed for flow rate {flow_rate}: {str(e)}")
            results[str(flow_rate)] = None
            continue
    
    # Calculate mean efficiency
    valid_results = [r['metrics']['efficiency']['overall'] 
                    for r in results.values() if r is not None]
    if valid_results:
        analysis['efficiency_stats']['mean'] = np.mean(valid_results)
    
    logger.info("\nAnalysis complete")
    logger.info(f"Best efficiency achieved: {analysis['efficiency_stats']['max']*100:.1f}%")
    
    return results, analysis

def run_test_case(name, flow_rates, base_params):
    """Run a single test case with given parameters."""
    logger.info(f"\nExecuting test case: {name}")
    
    results, analysis = run_optimization_cycle(flow_rates, base_params)
    
    logger.info(f"Test case {name} completed successfully")
    if results:
        best_efficiency = max(r['metrics']['efficiency']['overall'] * 100 
                            for r in results.values() if r is not None)
        logger.info(f"Best efficiency: {best_efficiency:.1f}%")
    
    return {
        'name': name,
        'results': results,
        'analysis': analysis
    }

def generate_visualizations(results_file):
    """Generate visualization plots from results."""
    try:
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        logger.info("Generating visualizations...")
        
        # Create efficiency vs flow rate plot
        create_efficiency_flow_plot(results)
        
        # Create convergence plot
        create_convergence_plot(results)
        
        # Create performance heatmap
        create_performance_heatmap(results)
        
        # Create performance map
        create_performance_map(results)
        
        # Create optimization progress plot
        create_optimization_progress(results)
        
        # Create blade loading analysis
        create_blade_loading_plot(results)
        
        # Create interactive analysis
        create_interactive_analysis(results)
        
        logger.info("\nAll visualizations have been generated in the 'results' directory:")
        logger.info("1. efficiency_flow_plot.png")
        logger.info("2. convergence_plot.png")
        logger.info("3. performance_heatmap.png")
        logger.info("4. performance_map.png")
        logger.info("5. optimization_progress.png")
        logger.info("6. blade_loading.png")
        logger.info("7. interactive_analysis.html")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")

def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    logger.info("Loaded configuration from config.yaml")
    
    # Create results directory if it doesn't exist
    Path('results').mkdir(exist_ok=True)
    
    # Initialize base parameters
    base_params = ImpellerParams()
    
    # Run all test cases
    all_results = {}
    
    # Single point optimizations
    for case in config['test_cases']:
        results = run_test_case(
            case['name'],
            case['flow_rates'],
            base_params
        )
        all_results[case['name']] = results
    
    # Multi-point optimization
    multi_flow_rates = [case['flow_rates'][0] for case in config['test_cases']]
    multi_results = run_test_case(
        'Multi-Point Optimization',
        multi_flow_rates,
        base_params
    )
    all_results['Multi-Point Optimization'] = multi_results
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'results/comprehensive_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4, cls=NumpyEncoder)
    
    logger.info("\nAutomated testing complete")
    logger.info(f"Results saved to {results_file}")
    
    # Print summary statistics
    logger.info("\nSummary Statistics:")
    for case_name, case_data in all_results.items():
        logger.info(f"\n{case_name}:")
        logger.info(f"- Max Efficiency: {case_data['analysis']['efficiency_stats']['max']*100:.1f}%")
        logger.info(f"- Average Efficiency: {case_data['analysis']['efficiency_stats']['mean']*100:.1f}%")
        for flow_rate, result in case_data['results'].items():
            if result is not None:
                logger.info(f"- Flow Rate {flow_rate} m/s:")
                logger.info(f"  * Improvement: {case_data['analysis']['convergence_rate'][flow_rate]['improvement']:.1f}%")
                logger.info(f"  * Final Error: {result['error']:.4f}")
    
    # Generate visualizations
    generate_visualizations(results_file)

if __name__ == "__main__":
    main() 