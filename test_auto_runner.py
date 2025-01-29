"""
Test suite for the AutomatedRunner class and related functionality.

This module tests:
- Configuration loading and validation
- Optimization cycle execution
- Results analysis
- Parameter adjustment logic
- Error handling and edge cases
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import json
import yaml
from auto_runner import AutomatedRunner, load_config, create_impeller_params

class TestAutoRunner(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary config for testing
        self.test_config = {
            'default_impeller': {
                'num_blades': 6,
                'outer_diameter': 0.3,
                'inner_diameter': 0.1,
                'blade_angle_inlet': 30,
                'blade_angle_outlet': 60,
                'blade_height': 0.05,
                'rotational_speed': 1750
            },
            'parameter_bounds': {
                'num_blades': [3, 12],
                'blade_angle_inlet': [10, 60],
                'blade_angle_outlet': [20, 70],
                'rotational_speed': [500, 3000]
            },
            'adjustment_rules': {
                'efficiency_threshold': 0.75,
                'angle_adjustment_step': 2.0,
                'speed_adjustment_factor': 1.1
            },
            'performance_targets': {
                'min_efficiency': 0.70,
                'max_flow_error': 5.0,
                'max_head_error': 5.0
            },
            'optimization_settings': {
                'convergence_threshold': 5.0,
                'save_results_dir': 'test_results',
                'log_level': 'INFO',
                'log_file': 'test_optimization.log'
            }
        }
        
        # Create runner instance
        self.runner = AutomatedRunner(self.test_config)
        
        # Create test results directory
        Path('test_results').mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up test results
        import shutil
        if Path('test_results').exists():
            shutil.rmtree('test_results')

    def test_config_loading(self):
        """Test configuration loading and validation."""
        # Test with valid config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name
        
        loaded_config = load_config(config_path)
        self.assertEqual(loaded_config['default_impeller']['num_blades'], 6)
        self.assertEqual(loaded_config['parameter_bounds']['num_blades'], [3, 12])
        
        Path(config_path).unlink()  # Clean up temp file

    def test_parameter_creation(self):
        """Test impeller parameter creation from config."""
        params = create_impeller_params(self.test_config['default_impeller'])
        self.assertEqual(params.num_blades, 6)
        self.assertEqual(params.outer_diameter, 0.3)
        self.assertEqual(params.blade_angle_inlet, 30)

    def test_optimization_cycle(self):
        """Test optimization cycle execution."""
        # Test with single flow rate
        results = self.runner.run_optimization_cycle(
            target_flow_rates=[0.1],
            max_iterations=5
        )
        
        # Check results structure
        self.assertIn(0.1, results)
        self.assertIn('params', results[0.1])
        self.assertIn('metrics', results[0.1])
        self.assertIn('error', results[0.1])
        
        # Check that error is finite
        self.assertFalse(np.isinf(results[0.1]['error']))

    def test_results_analysis(self):
        """Test results analysis functionality."""
        # Run a simple optimization to generate data
        self.runner.run_optimization_cycle(
            target_flow_rates=[0.1],
            max_iterations=3
        )
        
        # Analyze results
        analysis = self.runner.analyze_results()
        
        # Check analysis structure
        self.assertIn('total_iterations', analysis)
        self.assertIn('convergence_rate', analysis)
        self.assertIn('efficiency_stats', analysis)
        
        # Check that statistics are valid
        self.assertGreaterEqual(analysis['efficiency_stats']['max'], 0)
        self.assertLessEqual(analysis['efficiency_stats']['max'], 1)

    def test_parameter_adjustment(self):
        """Test parameter adjustment logic."""
        # Create test parameters and metrics
        params = create_impeller_params(self.test_config['default_impeller'])
        metrics = {
            'efficiency': {'overall': 0.7},
            'dimensionless': {
                'flow_coefficient': 0.15,
                'head_coefficient': 0.3
            }
        }
        
        # Test adjustment
        new_params = self.runner._adjust_parameters(params, metrics)
        
        # Check that adjustments are within bounds
        self.assertGreaterEqual(new_params.num_blades, self.test_config['parameter_bounds']['num_blades'][0])
        self.assertLessEqual(new_params.num_blades, self.test_config['parameter_bounds']['num_blades'][1])
        self.assertGreaterEqual(new_params.blade_angle_inlet, self.test_config['parameter_bounds']['blade_angle_inlet'][0])
        self.assertLessEqual(new_params.blade_angle_inlet, self.test_config['parameter_bounds']['blade_angle_inlet'][1])

    def test_error_handling(self):
        """Test error handling in optimization process."""
        # Test with invalid flow rate
        with self.assertRaises(ValueError):
            self.runner.run_optimization_cycle(
                target_flow_rates=[-1.0],  # Invalid negative flow rate
                max_iterations=5
            )
        
        # Test with invalid max_iterations
        with self.assertRaises(ValueError):
            self.runner.run_optimization_cycle(
                target_flow_rates=[0.1],
                max_iterations=0  # Invalid iteration count
            )

if __name__ == '__main__':
    unittest.main() 