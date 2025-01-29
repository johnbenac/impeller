import numpy as np
from impeller_sim import ImpellerParams, ImpellerSimulation
import logging
from typing import Dict, List, Tuple
import time
from dataclasses import asdict
import json
from datetime import datetime
import sys
import yaml
import os
from pathlib import Path

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(config: dict) -> logging.Logger:
    """Setup logging based on configuration"""
    log_settings = config['optimization_settings']
    os.makedirs(log_settings['save_results_dir'], exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_settings['log_level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_settings['save_results_dir'], log_settings['log_file'])),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("AutomatedTests")

def create_impeller_params(params_dict: dict) -> ImpellerParams:
    """Create ImpellerParams from dictionary"""
    return ImpellerParams(
        num_blades=params_dict['num_blades'],
        outer_diameter=params_dict['outer_diameter'],
        inner_diameter=params_dict['inner_diameter'],
        blade_angle_inlet=params_dict['blade_angle_inlet'],
        blade_angle_outlet=params_dict['blade_angle_outlet'],
        blade_height=params_dict['blade_height'],
        rotational_speed=params_dict['rotational_speed']
    )

class AutomatedRunner:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("AutomatedRunner")
        self.history = []
        self.best_params = None
        self.best_metrics = None
        self.bounds = config['parameter_bounds']
        self.adjustment_rules = config['adjustment_rules']
        self.performance_targets = config['performance_targets']
        
    def run_optimization_cycle(self, 
                             target_flow_rates: List[float],
                             max_iterations: int,
                             convergence_threshold: float = None) -> Dict:
        """Run automated optimization cycles for multiple target flow rates"""
        if convergence_threshold is None:
            convergence_threshold = self.config['optimization_settings']['convergence_threshold']
            
        self.logger.info(f"Starting optimization cycle for {len(target_flow_rates)} flow rates")
        
        results = {}
        for target_flow in target_flow_rates:
            self.logger.info(f"Optimizing for target flow rate: {target_flow} m/s")
            
            # Start with default parameters from config
            current_params = create_impeller_params(self.config['default_impeller'])
            sim = ImpellerSimulation(current_params)
            
            best_error = float('inf')
            iterations = 0
            
            while iterations < max_iterations:
                try:
                    optimized_params = sim.optimize_parameters(target_flow)
                    new_sim = ImpellerSimulation(optimized_params)
                    metrics = new_sim.calculate_performance_metrics()
                    
                    flow_error = abs(metrics['avg_outlet_velocity'] - target_flow)
                    
                    self.history.append({
                        'iteration': iterations,
                        'target_flow': target_flow,
                        'params': asdict(optimized_params),
                        'metrics': metrics,
                        'error': flow_error
                    })
                    
                    if flow_error < best_error:
                        best_error = flow_error
                        self.best_params = optimized_params
                        self.best_metrics = metrics
                        
                        self.logger.info(f"Improved solution found: Error = {flow_error:.4f}")
                        self.logger.info(f"Efficiency: {metrics['efficiency']*100:.1f}%")
                        
                        if flow_error < convergence_threshold:
                            self.logger.info("Convergence achieved")
                            break
                    
                    current_params = self._adjust_parameters(optimized_params, metrics)
                    sim = ImpellerSimulation(current_params)
                    
                except Exception as e:
                    self.logger.error(f"Optimization error: {str(e)}")
                    break
                
                iterations += 1
            
            results[target_flow] = {
                'params': asdict(self.best_params) if self.best_params else None,
                'metrics': self.best_metrics,
                'error': best_error,
                'iterations': iterations
            }
        
        return results
    
    def _adjust_parameters(self, params: ImpellerParams, metrics: Dict) -> ImpellerParams:
        """Intelligently adjust parameters based on performance metrics and config rules"""
        new_params = ImpellerParams(
            num_blades=params.num_blades,
            outer_diameter=params.outer_diameter,
            inner_diameter=params.inner_diameter,
            blade_angle_inlet=params.blade_angle_inlet,
            blade_angle_outlet=params.blade_angle_outlet,
            blade_height=params.blade_height,
            rotational_speed=params.rotational_speed
        )
        
        # Apply adjustments based on config rules
        if metrics['efficiency'] < self.adjustment_rules['efficiency_threshold']:
            if (new_params.num_blades < self.bounds['num_blades'][1] and 
                new_params.num_blades < params.num_blades + self.adjustment_rules['max_blade_increase']):
                new_params.num_blades += 1
            if new_params.blade_angle_outlet > self.bounds['blade_angle_outlet'][0]:
                new_params.blade_angle_outlet -= self.adjustment_rules['angle_adjustment_step']
        
        if metrics['specific_speed'] < self.performance_targets['target_specific_speed_range'][0]:
            new_speed = new_params.rotational_speed * self.adjustment_rules['speed_adjustment_factor']
            if new_speed <= self.bounds['rotational_speed'][1]:
                new_params.rotational_speed = new_speed
        elif metrics['specific_speed'] > self.performance_targets['target_specific_speed_range'][1]:
            new_speed = new_params.rotational_speed / self.adjustment_rules['speed_adjustment_factor']
            if new_speed >= self.bounds['rotational_speed'][0]:
                new_params.rotational_speed = new_speed
        
        return new_params
    
    def analyze_results(self) -> Dict:
        """Analyze optimization results and generate insights"""
        if not self.history:
            return {}
        
        analysis = {
            'total_iterations': len(self.history),
            'convergence_rate': {},
            'efficiency_stats': {},
            'parameter_sensitivity': {}
        }
        
        for target_flow in set(entry['target_flow'] for entry in self.history):
            flow_entries = [e for e in self.history if e['target_flow'] == target_flow]
            errors = [e['error'] for e in flow_entries]
            analysis['convergence_rate'][target_flow] = {
                'initial_error': errors[0],
                'final_error': errors[-1],
                'improvement': (errors[0] - errors[-1]) / errors[0] * 100
            }
        
        efficiencies = [entry['metrics']['efficiency'] for entry in self.history]
        analysis['efficiency_stats'] = {
            'mean': np.mean(efficiencies),
            'std': np.std(efficiencies),
            'max': np.max(efficiencies),
            'min': np.min(efficiencies)
        }
        
        self.logger.info("Analysis complete")
        self.logger.info(f"Best efficiency achieved: {analysis['efficiency_stats']['max']*100:.1f}%")
        
        return analysis

def main(config_path: str = "config.yaml"):
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Create results directory
    results_dir = Path(config['optimization_settings']['save_results_dir'])
    results_dir.mkdir(exist_ok=True)
    
    # Run test cases
    all_results = {}
    runner = AutomatedRunner(config)
    
    for test_case in config['test_cases']:
        logger.info(f"\nExecuting test case: {test_case['name']}")
        
        try:
            results = runner.run_optimization_cycle(
                target_flow_rates=test_case['flow_rates'],
                max_iterations=test_case['max_iterations']
            )
            analysis = runner.analyze_results()
            
            all_results[test_case['name']] = {
                'results': results,
                'analysis': analysis
            }
            
            logger.info(f"Test case {test_case['name']} completed successfully")
            logger.info(f"Best efficiency: {analysis['efficiency_stats']['max']*100:.1f}%")
            
        except Exception as e:
            logger.error(f"Error in test case {test_case['name']}: {str(e)}")
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"comprehensive_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4, default=str)
    
    logger.info("\nAutomated testing complete")
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    logger.info("\nSummary Statistics:")
    for case_name, case_results in all_results.items():
        logger.info(f"\n{case_name}:")
        analysis = case_results['analysis']
        logger.info(f"- Max Efficiency: {analysis['efficiency_stats']['max']*100:.1f}%")
        logger.info(f"- Average Efficiency: {analysis['efficiency_stats']['mean']*100:.1f}%")
        for flow_rate, conv_data in analysis['convergence_rate'].items():
            logger.info(f"- Flow Rate {float(flow_rate):.1f} m/s:")
            logger.info(f"  * Improvement: {conv_data['improvement']:.1f}%")
            logger.info(f"  * Final Error: {conv_data['final_error']:.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main() 