import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImpellerParams:
    """Parameters defining the impeller geometry"""
    num_blades: int = 6
    outer_diameter: float = 0.3  # meters
    inner_diameter: float = 0.1  # meters
    blade_angle_inlet: float = 30.0  # degrees
    blade_angle_outlet: float = 60.0  # degrees
    blade_height: float = 0.05  # meters
    rotational_speed: float = 1500  # RPM

    def __post_init__(self):
        """Validate parameters after initialization"""
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate parameter ranges and physical constraints"""
        if self.num_blades < 2:
            raise ValueError("Number of blades must be at least 2")
        if self.outer_diameter <= self.inner_diameter:
            raise ValueError("Outer diameter must be greater than inner diameter")
        if not (0 <= self.blade_angle_inlet <= 90):
            raise ValueError("Inlet blade angle must be between 0 and 90 degrees")
        if not (0 <= self.blade_angle_outlet <= 90):
            raise ValueError("Outlet blade angle must be between 0 and 90 degrees")
        if self.blade_height <= 0:
            raise ValueError("Blade height must be positive")
        if self.rotational_speed <= 0:
            raise ValueError("Rotational speed must be positive")

class ImpellerSimulation:
    def __init__(self, params: ImpellerParams):
        self.params = params
        self.initialize_geometry()
        self._validate_simulation()
        
    def _validate_simulation(self):
        """Validate simulation setup and physical constraints"""
        # Check mesh resolution
        if len(self.r) < 20 or len(self.theta) < 40:
            raise ValueError("Insufficient mesh resolution")
        
        # Check for numerical stability
        tip_speed = self.params.rotational_speed * np.pi * self.params.outer_diameter / 60
        if tip_speed > 200:  # m/s, typical limit for centrifugal impellers
            logger.warning("High tip speed may lead to numerical instability")

    def initialize_geometry(self):
        """Initialize the computational grid for the impeller"""
        self.r = np.linspace(self.params.inner_diameter/2, 
                            self.params.outer_diameter/2, 50)
        self.theta = np.linspace(0, 2*np.pi, 100)
        self.R, self.THETA = np.meshgrid(self.r, self.theta)
        self.X = self.R * np.cos(self.THETA)
        self.Y = self.R * np.sin(self.THETA)
        
    def calculate_velocity_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate velocity field using Euler equation with slip factor"""
        # Convert RPM to rad/s
        omega = self.params.rotational_speed * 2 * np.pi / 60
        
        # Calculate slip factor using Stodola correlation
        slip_factor = 1 - np.pi * np.sin(np.radians(self.params.blade_angle_outlet)) / self.params.num_blades
        
        # Tangential velocity component with slip factor
        v_theta = omega * self.R * slip_factor
        
        # Radial velocity component with blade angle variation
        angle_variation = np.interp(self.r, 
                                  [self.params.inner_diameter/2, self.params.outer_diameter/2],
                                  [self.params.blade_angle_inlet, self.params.blade_angle_outlet])
        v_r = omega * self.R * np.tan(np.radians(angle_variation[:, np.newaxis].T))
        
        # Convert to Cartesian coordinates
        v_x = v_r * np.cos(self.THETA) - v_theta * np.sin(self.THETA)
        v_y = v_r * np.sin(self.THETA) + v_theta * np.cos(self.THETA)
        
        return v_x, v_y
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        v_x, v_y = self.calculate_velocity_field()
        
        # Calculate average velocity magnitude at outlet
        outer_idx = -1
        v_mag_outlet = np.sqrt(v_x[:, outer_idx]**2 + v_y[:, outer_idx]**2)
        avg_outlet_velocity = np.mean(v_mag_outlet)
        
        # Calculate tip speed
        tip_speed = self.params.rotational_speed * np.pi * self.params.outer_diameter / 60
        
        # Calculate power factor
        power_factor = tip_speed**3
        
        # Calculate specific speed (dimensionless)
        omega = self.params.rotational_speed * 2 * np.pi / 60
        flow_coefficient = avg_outlet_velocity / tip_speed
        specific_speed = omega * np.sqrt(flow_coefficient) / (tip_speed**0.75)
        
        # Calculate hydraulic efficiency (simplified model)
        slip_factor = 1 - np.pi * np.sin(np.radians(self.params.blade_angle_outlet)) / self.params.num_blades
        efficiency = slip_factor * (1 - (self.params.inner_diameter/self.params.outer_diameter)**2)
        
        return {
            'avg_outlet_velocity': avg_outlet_velocity,
            'tip_speed': tip_speed,
            'power_factor': power_factor,
            'specific_speed': specific_speed,
            'efficiency': efficiency
        }
    
    def optimize_parameters(self, target_flow_rate: float) -> ImpellerParams:
        """Optimize impeller parameters for a target flow rate"""
        def objective(x):
            """Objective function for optimization"""
            try:
                test_params = ImpellerParams(
                    num_blades=int(x[0]),
                    blade_angle_inlet=x[1],
                    blade_angle_outlet=x[2],
                    rotational_speed=x[3]
                )
                sim = ImpellerSimulation(test_params)
                metrics = sim.calculate_performance_metrics()
                flow_error = abs(metrics['avg_outlet_velocity'] - target_flow_rate)
                efficiency_penalty = (1 - metrics['efficiency']) * 100
                return flow_error + efficiency_penalty
            except ValueError:
                return 1e6  # High penalty for invalid parameters
        
        # Initial guess
        x0 = [self.params.num_blades,
              self.params.blade_angle_inlet,
              self.params.blade_angle_outlet,
              self.params.rotational_speed]
        
        # Parameter bounds
        bounds = [(3, 12),      # num_blades
                 (10, 60),      # inlet angle
                 (20, 70),      # outlet angle
                 (500, 3000)]   # rotational speed
        
        result = minimize(objective, x0, bounds=bounds, method='SLSQP')
        
        if result.success:
            optimized_params = ImpellerParams(
                num_blades=int(round(result.x[0])),
                blade_angle_inlet=result.x[1],
                blade_angle_outlet=result.x[2],
                rotational_speed=result.x[3],
                outer_diameter=self.params.outer_diameter,
                inner_diameter=self.params.inner_diameter,
                blade_height=self.params.blade_height
            )
            return optimized_params
        else:
            raise RuntimeError("Parameter optimization failed")
    
    def visualize_flow(self, ax=None):
        """Visualize the flow field with enhanced graphics"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
            
        v_x, v_y = self.calculate_velocity_field()
        
        # Calculate velocity magnitude for coloring
        v_mag = np.sqrt(v_x**2 + v_y**2)
        
        # Plot velocity field with color-coded magnitude
        skip = 5
        quiver = ax.quiver(self.X[::skip, ::skip], self.Y[::skip, ::skip],
                          v_x[::skip, ::skip], v_y[::skip, ::skip],
                          v_mag[::skip, ::skip],
                          scale=50, cmap='viridis')
        plt.colorbar(quiver, ax=ax, label='Velocity magnitude (m/s)')
        
        # Plot impeller outline
        circle_outer = plt.Circle((0, 0), self.params.outer_diameter/2,
                                fill=False, color='black', linestyle='--')
        circle_inner = plt.Circle((0, 0), self.params.inner_diameter/2,
                                fill=False, color='black', linestyle='--')
        ax.add_artist(circle_outer)
        ax.add_artist(circle_inner)
        
        # Plot blades with curvature
        for i in range(self.params.num_blades):
            angle = i * 2 * np.pi / self.params.num_blades
            r = np.linspace(self.params.inner_diameter/2, self.params.outer_diameter/2, 50)
            theta = np.linspace(angle, 
                              angle + np.radians(self.params.blade_angle_outlet),
                              50)
            x_blade = r * np.cos(theta)
            y_blade = r * np.sin(theta)
            ax.plot(x_blade, y_blade, 'k-', linewidth=2)
        
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Impeller Flow Field')
        
        return ax 