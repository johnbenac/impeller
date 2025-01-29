"""
Impeller Flow Simulator - Test Suite

This module contains unit tests for the impeller simulation components.
It tests:
- Parameter validation
- Geometry initialization
- Velocity field calculations
- Performance metrics
- Physical constraints

Engineering Testing Philosophy:
1. Verification and Validation (V&V):
   - Verification: Code solves equations correctly
   - Validation: Equations model physics correctly
   - Importance of systematic testing in engineering

2. Test-Driven Development in Engineering:
   - Writing tests before implementation
   - Ensuring physical constraints are met
   - Maintaining code reliability

3. Quality Assurance Principles:
   - Boundary condition testing
   - Edge case identification
   - Performance validation

Learning Objectives:
- Understanding engineering testing methodology
- Implementing physical constraints in code
- Validating numerical simulations
- Developing robust engineering software

Further Study Areas:
- Advanced validation techniques
- Statistical analysis of results
- Uncertainty quantification
- Automated testing in engineering

The tests ensure the simulator maintains physical consistency and numerical stability
across different operating conditions.

Author: John Benac
Date: January 2024
License: MIT
"""

# Import Python's built-in testing framework
import unittest
# Import numpy for numerical comparisons
import numpy as np
# Import our impeller simulation code
from impeller_sim import ImpellerParams, ImpellerSimulation

# TestCase class contains all our tests
# unittest.TestCase provides methods for making assertions about code behavior
class TestImpellerSimulation(unittest.TestCase):
    def setUp(self):
        """
        This method runs before each test.
        It creates a fresh impeller simulation with default parameters.
        This ensures each test starts with the same clean state.

        Engineering Testing Principles:
        1. Test Isolation:
           - Each test should be independent
           - Known initial conditions
           - Reproducible results

        2. Default Parameters:
           - Industry standard values
           - Physically reasonable ranges
           - Representative operating conditions
        """
        self.default_params = ImpellerParams()  # Create default parameters
        self.sim = ImpellerSimulation(self.default_params)  # Create simulation

    def test_geometry_initialization(self):
        """
        Test if the computational grid is set up correctly.
        This checks that:
        1. The grid dimensions match
        2. The radius values are within bounds
        3. The angle values are within bounds

        Engineering Significance:
        1. Geometric Validation:
           - Essential for CFD accuracy
           - Mesh quality affects results
           - Grid independence considerations

        2. Physical Constraints:
           - Dimensional consistency
           - Boundary definitions
           - Coordinate system validity

        Learning Topics:
        - Computational grid generation
        - Mesh quality metrics
        - Geometric modeling principles
        """
        # Check that R and THETA matrices have same shape
        self.assertEqual(self.sim.R.shape, self.sim.THETA.shape)
        
        # Check that all radii are between inner and outer radius
        self.assertTrue(np.all(self.sim.R >= self.default_params.inner_diameter/2))
        self.assertTrue(np.all(self.sim.R <= self.default_params.outer_diameter/2))
        
        # Check that all angles are between 0 and 2π
        self.assertTrue(np.all(self.sim.THETA >= 0))
        self.assertTrue(np.all(self.sim.THETA <= 2*np.pi))

    def test_velocity_field(self):
        """
        Test if velocity calculations produce physically reasonable results.
        This checks:
        1. Velocity field has correct dimensions
        2. No invalid (NaN) values
        3. Velocity increases with radius (as expected in a centrifugal impeller)

        Fluid Dynamics Principles:
        1. Conservation Laws:
           - Mass conservation check
           - Momentum conservation
           - Energy conservation implications

        2. Flow Field Properties:
           - Velocity magnitude progression
           - Directional components
           - Continuity requirements

        3. Physical Reasonableness:
           - Expected flow patterns
           - Velocity gradients
           - Boundary conditions

        Advanced Concepts:
        - Reynolds number effects
        - Turbulence considerations
        - Secondary flow patterns
        """
        # Calculate velocity field
        v_x, v_y = self.sim.calculate_velocity_field()
        
        # Test that velocity arrays have same shape as grid
        self.assertEqual(v_x.shape, self.sim.R.shape)
        self.assertEqual(v_y.shape, self.sim.R.shape)
        
        # Test that there are no NaN values in velocity field
        self.assertFalse(np.any(np.isnan(v_x)))
        self.assertFalse(np.any(np.isnan(v_y)))
        
        # Test that velocity magnitude increases with radius
        # This is a key physical characteristic of centrifugal impellers
        v_mag = np.sqrt(v_x**2 + v_y**2)  # Calculate velocity magnitude
        for i in range(len(self.sim.r)-1):
            # Compare average velocity at each radius
            avg_v_inner = np.mean(v_mag[:, i])      # Inner radius
            avg_v_outer = np.mean(v_mag[:, i+1])    # Outer radius
            self.assertGreater(avg_v_outer, avg_v_inner)  # Outer should be faster

    def test_performance_metrics(self):
        """
        Test if performance calculations give reasonable results.
        This checks:
        1. All required metrics are present
        2. All values are positive (physically meaningful)
        3. Tip speed calculation is correct

        Performance Analysis Principles:
        1. Key Performance Indicators (KPIs):
           - Efficiency metrics
           - Power consumption
           - Flow characteristics

        2. Dimensional Analysis:
           - Non-dimensional parameters
           - Scaling relationships
           - Similarity principles

        3. Operating Limits:
           - Material constraints
           - Flow regime boundaries
           - Efficiency thresholds

        Industry Applications:
        - Performance testing standards
        - Acceptance criteria
        - Operating range verification
        """
        # Calculate performance metrics
        metrics = self.sim.calculate_performance_metrics()
        
        # Test that all required metrics exist in results
        self.assertIn('avg_outlet_velocity', metrics)
        self.assertIn('tip_speed', metrics)
        self.assertIn('power_factor', metrics)
        
        # Test that all metrics have physically meaningful (positive) values
        self.assertGreater(metrics['avg_outlet_velocity'], 0)
        self.assertGreater(metrics['tip_speed'], 0)
        self.assertGreater(metrics['power_factor'], 0)
        
        # Test tip speed calculation against manual calculation
        # This verifies the basic physics is correct
        expected_tip_speed = self.default_params.rotational_speed * np.pi * self.default_params.outer_diameter / 60
        self.assertAlmostEqual(metrics['tip_speed'], expected_tip_speed)

    def test_parameter_validation(self):
        """
        Test if parameter validation catches invalid inputs.
        This checks that the code properly rejects:
        1. Invalid number of blades
        2. Invalid diameter combinations
        3. Invalid blade angles

        Design Validation Principles:
        1. Parameter Constraints:
           - Physical limitations
           - Manufacturing constraints
           - Operating requirements

        2. Error Handling:
           - Early detection of issues
           - Clear error messages
           - Fail-safe design

        3. Design Rules:
           - Industry standards
           - Best practices
           - Safety factors

        Educational Value:
        - Understanding design limits
        - Engineering constraints
        - Safety considerations
        """
        # Test case 1: Invalid number of blades
        with self.assertRaises(ValueError):
            params = ImpellerParams(num_blades=0)  # Try to create impeller with 0 blades
            ImpellerSimulation(params)
            
        # Test case 2: Invalid diameter combination
        with self.assertRaises(ValueError):
            # Try to create impeller where inner diameter > outer diameter
            params = ImpellerParams(outer_diameter=0.1, inner_diameter=0.2)
            ImpellerSimulation(params)
            
        # Test case 3: Invalid blade angle
        with self.assertRaises(ValueError):
            # Try to create impeller with negative inlet angle
            params = ImpellerParams(blade_angle_inlet=-10)
            ImpellerSimulation(params)

# This allows running the tests directly from command line
if __name__ == '__main__':
    unittest.main() 