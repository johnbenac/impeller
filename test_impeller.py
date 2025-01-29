import unittest
import numpy as np
from impeller_sim import ImpellerParams, ImpellerSimulation

class TestImpellerSimulation(unittest.TestCase):
    def setUp(self):
        self.default_params = ImpellerParams()
        self.sim = ImpellerSimulation(self.default_params)

    def test_geometry_initialization(self):
        """Test if geometry initialization produces valid meshgrid"""
        self.assertEqual(self.sim.R.shape, self.sim.THETA.shape)
        self.assertTrue(np.all(self.sim.R >= self.default_params.inner_diameter/2))
        self.assertTrue(np.all(self.sim.R <= self.default_params.outer_diameter/2))
        self.assertTrue(np.all(self.sim.THETA >= 0))
        self.assertTrue(np.all(self.sim.THETA <= 2*np.pi))

    def test_velocity_field(self):
        """Test if velocity field calculations are physically reasonable"""
        v_x, v_y = self.sim.calculate_velocity_field()
        
        # Test shapes
        self.assertEqual(v_x.shape, self.sim.R.shape)
        self.assertEqual(v_y.shape, self.sim.R.shape)
        
        # Test for NaN values
        self.assertFalse(np.any(np.isnan(v_x)))
        self.assertFalse(np.any(np.isnan(v_y)))
        
        # Test velocity magnitudes increase with radius
        v_mag = np.sqrt(v_x**2 + v_y**2)
        for i in range(len(self.sim.r)-1):
            avg_v_inner = np.mean(v_mag[:, i])
            avg_v_outer = np.mean(v_mag[:, i+1])
            self.assertGreater(avg_v_outer, avg_v_inner)

    def test_performance_metrics(self):
        """Test if performance metrics are physically reasonable"""
        metrics = self.sim.calculate_performance_metrics()
        
        # Test presence of all metrics
        self.assertIn('avg_outlet_velocity', metrics)
        self.assertIn('tip_speed', metrics)
        self.assertIn('power_factor', metrics)
        
        # Test positive values
        self.assertGreater(metrics['avg_outlet_velocity'], 0)
        self.assertGreater(metrics['tip_speed'], 0)
        self.assertGreater(metrics['power_factor'], 0)
        
        # Test tip speed calculation
        expected_tip_speed = self.default_params.rotational_speed * np.pi * self.default_params.outer_diameter / 60
        self.assertAlmostEqual(metrics['tip_speed'], expected_tip_speed)

    def test_parameter_validation(self):
        """Test parameter validation and boundary conditions"""
        # Test invalid number of blades
        with self.assertRaises(ValueError):
            params = ImpellerParams(num_blades=0)
            ImpellerSimulation(params)
            
        # Test invalid diameter combination
        with self.assertRaises(ValueError):
            params = ImpellerParams(outer_diameter=0.1, inner_diameter=0.2)
            ImpellerSimulation(params)
            
        # Test invalid angles
        with self.assertRaises(ValueError):
            params = ImpellerParams(blade_angle_inlet=-10)
            ImpellerSimulation(params)

if __name__ == '__main__':
    unittest.main() 