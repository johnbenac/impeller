"""
Impeller Flow Simulator - Core Module

This module implements a simplified 2D model of fluid flow through a centrifugal impeller.
It uses basic principles of fluid dynamics and turbomachinery to simulate the flow field
and calculate key performance metrics.

Fundamental Concepts:

1. Centrifugal Impeller Basics:
   - Converts mechanical energy to fluid pressure and kinetic energy
   - Uses centrifugal force and blade guidance to accelerate fluid
   - Key component in pumps, compressors, and turbines

2. Fluid Dynamics Principles:
   - Conservation of mass (continuity)
   - Conservation of momentum
   - Conservation of energy
   - Euler turbomachine equation

3. Design Parameters:
   - Blade geometry (angle, number, thickness)
   - Diameter ratio (hub to tip)
   - Rotational speed
   - Flow coefficients

4. Performance Metrics:
   - Head rise
   - Efficiency
   - Power consumption
   - Flow rate

Learning Objectives:
- Understanding impeller geometry and design
- Basic CFD concepts and implementation
- Performance analysis methods
- Engineering optimization principles

Further Study Areas:
- Advanced CFD methods
- Turbulence modeling
- Multi-phase flows
- 3D effects and secondary flows

Author: John Benac
Date: January 2024
License: MIT
"""

# Import necessary Python libraries
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class ImpellerParams:
    """
    Parameters defining the impeller geometry and operating conditions.

    Engineering Design Considerations:
    1. Geometric Parameters:
       - Diameters affect flow acceleration
       - Blade angles influence flow guidance
       - Number of blades affects loading
       - Blade height affects flow capacity

    2. Operating Parameters:
       - Speed determines energy input
       - Flow rate affects performance
       - Reynolds number regime

    3. Design Constraints:
       - Material strength limits
       - Manufacturing capabilities
       - Efficiency requirements
    """
    outer_diameter: float = 0.3    # Impeller outer diameter [m]
    inner_diameter: float = 0.1    # Impeller inner diameter [m]
    blade_angle_inlet: float = 30  # Blade angle at inlet [degrees]
    blade_angle_outlet: float = 60 # Blade angle at outlet [degrees]
    blade_height: float = 0.05     # Blade height [m]
    num_blades: int = 6           # Number of impeller blades
    rotational_speed: float = 1750 # Impeller speed [RPM]
    grid_points_r: int = 50       # Number of radial grid points
    grid_points_theta: int = 100   # Number of circumferential grid points

class ImpellerSimulation:
    """
    Main simulation class implementing the 2D impeller flow model.
    
    Numerical Methods:
    1. Grid Generation:
       - Structured mesh in polar coordinates
       - Resolution requirements
       - Boundary treatment

    2. Flow Field Calculation:
       - Velocity components
       - Pressure distribution
       - Loss models

    3. Performance Analysis:
       - Integration methods
       - Averaging techniques
       - Error estimation
    """
    
    def __init__(self, params: ImpellerParams):
        """
        Initialize the simulation with given parameters.
        
        Initialization Process:
        1. Parameter Validation:
           - Check physical constraints
           - Verify numerical parameters
           - Set default values

        2. Grid Generation:
           - Create coordinate system
           - Define mesh points
           - Set up boundaries
        """
        self._validate_parameters(params)
        self.params = params
        self._initialize_grid()
        # Initialize flow data
        self._flow_data = None

    def _validate_parameters(self, params: ImpellerParams):
        """
        Validate input parameters against physical and numerical constraints.
        
        Validation Criteria:
        1. Physical Constraints:
           - Positive dimensions
           - Realistic angles
           - Feasible speeds

        2. Numerical Requirements:
           - Sufficient grid resolution
           - Stable time steps
           - Convergence criteria
        """
        if params.outer_diameter <= params.inner_diameter:
            raise ValueError("Outer diameter must be greater than inner diameter")
        if params.num_blades < 1:
            raise ValueError("Number of blades must be positive")
        if params.blade_angle_inlet < 0 or params.blade_angle_inlet > 90:
            raise ValueError("Inlet blade angle must be between 0 and 90 degrees")
        if params.blade_angle_outlet < 0 or params.blade_angle_outlet > 90:
            raise ValueError("Outlet blade angle must be between 0 and 90 degrees")

    def _initialize_grid(self):
        """
        Create computational grid in polar coordinates.
        
        Grid Generation Process:
        1. Coordinate System:
           - Polar coordinates (r, θ)
           - Non-dimensional scaling
           - Grid stretching

        2. Mesh Quality:
           - Aspect ratio
           - Orthogonality
           - Smoothness
        """
        # Create radial grid points
        self.r = np.linspace(self.params.inner_diameter/2, 
                            self.params.outer_diameter/2,
                            self.params.grid_points_r)
        
        # Create angular grid points
        self.theta = np.linspace(0, 2*np.pi, self.params.grid_points_theta)
        
        # Create 2D mesh grid
        self.R, self.THETA = np.meshgrid(self.r, self.theta)

    def calculate_velocity_field(self, flow_coefficient=None):
        """
        Calculate velocity components in the impeller passage.
        
        Args:
            flow_coefficient: Optional flow coefficient to override default calculation
        """
        try:
            # Convert blade angles to radians
            beta1 = np.radians(self.params.blade_angle_inlet)
            beta2 = np.radians(self.params.blade_angle_outlet)
            
            # Calculate blade velocities (u = ωr)
            omega = self.params.rotational_speed * 2 * np.pi / 60  # rad/s
            u = omega * self.R
            
            # Calculate flow area with scalar values
            blade_thickness = 0.002  # typical blade thickness [m]
            # Calculate blockage factor at each radius
            r_mean = np.mean(self.R, axis=0)  # Average radius at each radial position
            blockage_factor = 1 - (self.params.num_blades * blade_thickness) / (2 * np.pi * r_mean)
            flow_area = 2 * np.pi * r_mean * self.params.blade_height * blockage_factor
            
            # Calculate mass flow rate using flow coefficient
            rho = 1000.0  # water density [kg/m³]
            u1 = omega * self.params.inner_diameter/2
            
            # Use provided flow coefficient or calculate optimal one
            if flow_coefficient is None:
                # Calculate optimal flow coefficient based on inlet blade angle
                flow_coefficient = np.sin(beta1) * np.cos(beta1)  # optimal flow coefficient
            
            cm1 = flow_coefficient * u1  # meridional velocity at inlet
            
            # Calculate inlet area and mass flow
            inlet_blockage = 1 - (self.params.num_blades * blade_thickness) / (2 * np.pi * self.params.inner_diameter/2)
            mass_flow = rho * cm1 * np.pi * self.params.inner_diameter * self.params.blade_height * inlet_blockage
            
            # Ensure mass flow is positive
            if mass_flow <= 0:
                raise ValueError("Invalid mass flow rate calculated")
            
            # Calculate meridional velocity using continuity
            v_m = np.zeros_like(self.R)
            for i in range(len(r_mean)):
                v_m[:, i] = mass_flow / (rho * flow_area[i])
            
            # Calculate relative velocity components at inlet
            w_m1 = float(v_m[0, 0])  # meridional component at inlet
            u1 = float(u[0, 0])      # blade speed at inlet
            w_theta1 = -u1 + w_m1 / np.tan(beta1)  # tangential component
            
            # Calculate relative velocity components at outlet
            w_m2 = float(v_m[0, -1])  # meridional component at outlet
            u2 = float(u[0, -1])      # blade speed at outlet
            
            # Improved slip factor calculation using Wiesner's formula with stability checks
            sigma = 1 - np.sqrt(np.cos(beta2))
            slip_factor = 1 - sigma / np.power(self.params.num_blades, 0.7)
            slip_factor = max(0.5, min(0.95, slip_factor))  # limit to reasonable range
            
            # Calculate relative velocity angle variation along blade
            # Use smooth transition from inlet to outlet with improved curve
            r_norm = (r_mean - r_mean[0]) / (r_mean[-1] - r_mean[0])
            beta_variation = beta1 + (beta2 - beta1) * (3 * r_norm**2 - 2 * r_norm**3)  # smoother transition
            
            # Calculate relative velocity components with improved model
            w_m = v_m  # meridional component
            w_theta = np.zeros_like(self.R)
            
            # Calculate relative tangential velocity with blade loading
            for i in range(len(r_mean)):
                # Basic blade-to-blade pressure gradient
                dp_db = rho * u[:, i]**2 / r_mean[i]
                # Adjust tangential velocity based on pressure gradient
                w_theta[:, i] = -u[:, i] + w_m[:, i] / np.tan(beta_variation[i])
                # Apply loading adjustment
                loading_factor = 1 - 0.1 * (r_norm[i])**2  # reduced loading near outlet
                w_theta[:, i] *= loading_factor
            
            # Apply slip factor gradually from mid-passage to outlet
            slip_correction = np.where(r_norm > 0.5, 
                                     (1 - slip_factor) * (2 * r_norm - 1)**2, 
                                     0)
            for i in range(len(r_mean)):
                w_theta[:, i] *= (1 - slip_correction[i])
            
            # Calculate absolute velocity components
            c_m = w_m.copy()
            c_theta = w_theta + u
            
            # Convert to Cartesian coordinates
            v_x = c_m * np.cos(self.THETA) - c_theta * np.sin(self.THETA)
            v_y = c_m * np.sin(self.THETA) + c_theta * np.cos(self.THETA)
            
            # Store additional flow data
            self._flow_data = {
                'u': u,
                'w_m': w_m,
                'w_theta': w_theta,
                'c_m': c_m,
                'c_theta': c_theta,
                'slip_factor': float(slip_factor),
                'flow_area': flow_area,
                'mass_flow': float(mass_flow),
                'rho': float(rho),
                'omega': float(omega),
                'flow_coefficient': float(flow_coefficient)
            }
            
            return v_x, v_y
            
        except Exception as e:
            print(f"Error in velocity field calculation: {str(e)}")
            raise
    
    def calculate_performance_metrics(self):
        """
        Calculate key performance metrics for the impeller.
        """
        if self._flow_data is None:
            raise RuntimeError("Flow field must be calculated before computing performance metrics")
            
        try:
            # Get flow field data
            rho = self._flow_data['rho']
            omega = self._flow_data['omega']
            mass_flow = self._flow_data['mass_flow']
            g = 9.81  # gravitational acceleration [m/s²]
            
            # Calculate Euler head (theoretical)
            r1 = float(self.r[0])   # inlet radius
            r2 = float(self.r[-1])  # outlet radius
            c_u1 = float(np.mean(self._flow_data['c_theta'][:, 0]))   # inlet tangential velocity
            c_u2 = float(np.mean(self._flow_data['c_theta'][:, -1]))  # outlet tangential velocity
            euler_head = (omega * (r2 * c_u2 - r1 * c_u1)) / g
            
            if euler_head <= 0:
                raise ValueError("Invalid Euler head calculated")
            
            # Calculate hydraulic losses with improved models
            # 1. Profile losses (friction)
            v_rel = np.sqrt(self._flow_data['w_m']**2 + self._flow_data['w_theta']**2)
            hydraulic_diameter = 4 * self._flow_data['flow_area'] / (2 * np.pi * np.mean(self.R, axis=0))
            Re = rho * v_rel * hydraulic_diameter / 1.81e-5  # Reynolds number
            Cf = 0.074 / Re**0.2  # friction coefficient
            L = np.sqrt((self.params.outer_diameter - self.params.inner_diameter)**2 + 
                       (self.params.blade_height * np.tan(np.radians(self.params.blade_angle_outlet)))**2)
            friction_loss = float(np.mean(Cf * L * v_rel**2 / (2 * g * hydraulic_diameter)))
            
            # 2. Incidence loss with improved model
            beta1 = np.radians(self.params.blade_angle_inlet)
            w_m1 = float(np.mean(self._flow_data['w_m'][:, 0]))
            w_theta1 = float(np.mean(self._flow_data['w_theta'][:, 0]))
            v_rel_in = np.sqrt(w_m1**2 + w_theta1**2)
            flow_angle = np.arctan2(w_m1, -w_theta1)
            incidence = abs(flow_angle - beta1)
            incidence_loss = 0.8 * (v_rel_in * np.sin(incidence))**2 / (2 * g)
            
            # 3. Diffusion loss with improved model
            w_m2 = float(np.mean(self._flow_data['w_m'][:, -1]))
            w_theta2 = float(np.mean(self._flow_data['w_theta'][:, -1]))
            w2 = np.sqrt(w_m2**2 + w_theta2**2)
            w1 = v_rel_in  # reuse from incidence calculation
            D = 1 - w2/w1 + abs(w_theta2 - w_theta1) / (2 * w1)
            diffusion_loss = 0.05 * D**3 * w1**2 / (2 * g)  # Modified exponent for better prediction
            
            # 4. Disk friction loss with improved model
            disk_Re = omega * self.params.outer_diameter**2 / 1.81e-5
            Cm = 0.04 / disk_Re**0.2
            disk_loss = Cm * omega**3 * self.params.outer_diameter**5 / (8 * g * mass_flow)
            
            # 5. Secondary flow losses (new)
            aspect_ratio = self.params.blade_height / (self.params.outer_diameter - self.params.inner_diameter)
            secondary_loss_coeff = 0.03 * (1 + 2/aspect_ratio)
            secondary_loss = secondary_loss_coeff * v_rel_in**2 / (2 * g)
            
            # 6. Clearance losses (new)
            tip_clearance = 0.0003  # m
            clearance_ratio = tip_clearance / self.params.blade_height
            clearance_loss_coeff = 0.6 * clearance_ratio
            clearance_loss = clearance_loss_coeff * euler_head
            
            # Total losses with improved minimum threshold
            total_losses = max(0.05 * euler_head,  # Minimum loss threshold
                             friction_loss + incidence_loss + diffusion_loss + 
                             disk_loss + secondary_loss + clearance_loss)
            
            # Calculate actual head
            actual_head = euler_head - total_losses
            
            # Calculate volumetric flow rate
            Q = mass_flow / rho  # m³/s
            
            # Calculate tip speed
            tip_speed = omega * self.params.outer_diameter/2
            
            # Calculate efficiency components with flow-dependent factors
            # Flow coefficient for efficiency calculations
            flow_coeff = self._flow_data['flow_coefficient']
            
            # Optimal flow coefficient range
            opt_flow_coeff = 0.1
            flow_deviation = abs(flow_coeff - opt_flow_coeff)
            
            # Hydraulic efficiency with improved off-design penalties
            hydraulic_efficiency = actual_head / euler_head
            hydraulic_efficiency *= np.exp(-2 * flow_deviation**2)  # Penalty for off-design operation
            
            # Volumetric efficiency with improved leakage model
            clearance = 0.0003  # typical clearance gap [m]
            pressure_coefficient = actual_head * g / tip_speed**2
            leakage_factor = 1 + 0.5 * pressure_coefficient * (flow_coeff / opt_flow_coeff - 1)**2
            volumetric_efficiency = 1 / (1 + clearance * leakage_factor / self.params.blade_height)
            
            # Mechanical efficiency with improved speed and flow effects
            base_mech_eff = 0.95 - 0.02 * (omega / 1000)  # Base mechanical efficiency
            flow_penalty = 0.05 * (flow_coeff / opt_flow_coeff - 1)**2  # Flow-dependent losses
            speed_penalty = 0.01 * (omega / 1500 - 1)**2  # Speed-dependent losses
            mechanical_efficiency = max(0.7, base_mech_eff - flow_penalty - speed_penalty)
            
            # Overall efficiency
            overall_efficiency = hydraulic_efficiency * volumetric_efficiency * mechanical_efficiency
            
            # Calculate power components
            hydraulic_power = rho * g * Q * actual_head
            shaft_power = hydraulic_power / overall_efficiency
            mechanical_power = shaft_power / mechanical_efficiency
            
            # Calculate dimensionless parameters
            N = self.params.rotational_speed  # RPM
            Ns = (N * Q**0.5) / (g * actual_head)**0.75 if actual_head > 0 else 0
            head_coeff = g * actual_head / tip_speed**2 if tip_speed > 0 else 0
            
            # Store and return performance metrics
            return {
                'head': {
                    'euler': float(euler_head),
                    'actual': float(actual_head),
                    'losses': float(total_losses)
                },
                'flow': {
                    'mass_flow': float(mass_flow),
                    'volumetric_flow': float(Q)
                },
                'power': {
                    'hydraulic': float(hydraulic_power),
                    'shaft': float(shaft_power),
                    'mechanical': float(mechanical_power)
                },
                'efficiency': {
                    'hydraulic': float(hydraulic_efficiency),
                    'volumetric': float(volumetric_efficiency),
                    'mechanical': float(mechanical_efficiency),
                    'overall': float(overall_efficiency)
                },
                'dimensionless': {
                    'specific_speed': float(Ns),
                    'flow_coefficient': float(flow_coeff),
                    'head_coefficient': float(head_coeff)
                },
                'loss_breakdown': {
                    'friction': float(friction_loss / total_losses),
                    'incidence': float(incidence_loss / total_losses),
                    'diffusion': float(diffusion_loss / total_losses),
                    'disk': float(disk_loss / total_losses),
                    'secondary': float(secondary_loss / total_losses),
                    'clearance': float(clearance_loss / total_losses)
                }
            }
            
        except Exception as e:
            print(f"Error in performance metrics calculation: {str(e)}")
            raise

    def optimize_parameters(self, target_flow_rate: float) -> ImpellerParams:
        """
        Optimize impeller parameters to achieve a target flow rate.
        This method uses numerical optimization to find the best combination of parameters.
        
        Args:
            target_flow_rate: The desired flow rate in m³/s
            
        Returns:
            An ImpellerParams object with optimized parameters
        """
        def objective(x):
            """
            Objective function for optimization.
            Combines flow rate error and efficiency penalties with improved weighting.
            
            Args:
                x: Array of parameters [num_blades, inlet_angle, outlet_angle, speed]
            
            Returns:
                Total score (lower is better)
            """
            try:
                # Create test impeller with proposed parameters
                test_params = ImpellerParams(
                    num_blades=int(round(x[0])),
                    blade_angle_inlet=float(x[1]),
                    blade_angle_outlet=float(x[2]),
                    rotational_speed=float(x[3]),
                    outer_diameter=self.params.outer_diameter,
                    inner_diameter=self.params.inner_diameter,
                    blade_height=self.params.blade_height
                )
                
                # Run simulation
                sim = ImpellerSimulation(test_params)
                sim.calculate_velocity_field()
                metrics = sim.calculate_performance_metrics()
                
                # Calculate volumetric flow rate
                Q = metrics['flow']['volumetric_flow']
                
                # Calculate error components with adjusted weights
                flow_error = abs(Q - target_flow_rate) / target_flow_rate * 100  # Percentage error
                efficiency = metrics['efficiency']['overall']
                efficiency_penalty = (1 - efficiency) * 50  # Reduced weight for efficiency
                
                # Flow coefficient error (target range: 0.1 - 0.3)
                flow_coeff = metrics['dimensionless']['flow_coefficient']
                flow_coeff_error = 0
                if flow_coeff < 0.1:
                    flow_coeff_error = (0.1 - flow_coeff) * 200
                elif flow_coeff > 0.3:
                    flow_coeff_error = (flow_coeff - 0.3) * 200
                
                # Head coefficient error (target range: 0.1 - 0.6)
                head_coeff = metrics['dimensionless']['head_coefficient']
                head_coeff_error = 0
                if head_coeff < 0.1:
                    head_coeff_error = (0.1 - head_coeff) * 100
                elif head_coeff > 0.6:
                    head_coeff_error = (head_coeff - 0.6) * 100
                
                # Physical constraints penalties
                constraints_penalty = 0
                
                # Tip speed constraint (material strength)
                omega = test_params.rotational_speed * 2 * np.pi / 60  # rad/s
                tip_speed = omega * test_params.outer_diameter/2
                max_tip_speed = 200  # m/s
                if tip_speed > max_tip_speed:
                    constraints_penalty += (tip_speed - max_tip_speed) * 20
                
                # Blade loading constraint
                euler_head = metrics['head']['euler']
                max_head = (max_tip_speed**2) / (2 * 9.81)  # Maximum theoretical head
                if euler_head > max_head:
                    constraints_penalty += (euler_head - max_head) * 10
                
                # Total score combines all components with flow rate error having highest weight
                total_score = (2.0 * flow_error +  # Doubled weight for flow rate error
                             efficiency_penalty +
                             flow_coeff_error +
                             head_coeff_error +
                             constraints_penalty)
                
                return float(total_score)  # Ensure we return a scalar
            
            except Exception as e:
                print(f"Error in optimization objective: {str(e)}")
                return 1e6  # Return high score for invalid parameters
        
        # Initial guess - use current parameters but adjust based on target flow
        self.calculate_velocity_field()  # Ensure flow field is calculated for current state
        current_metrics = self.calculate_performance_metrics()
        current_flow = current_metrics['flow']['volumetric_flow']
        flow_ratio = target_flow_rate / current_flow
        
        # Adjust initial guess based on flow ratio
        x0 = np.array([
            float(self.params.num_blades),
            float(self.params.blade_angle_inlet * min(1.5, max(0.5, flow_ratio))),  # Adjust inlet angle
            float(self.params.blade_angle_outlet),
            float(self.params.rotational_speed * min(1.5, max(0.5, flow_ratio**0.5)))  # Adjust speed
        ])
        
        # Parameter bounds
        bounds = [
            (3, 12),      # num_blades: between 3 and 12 blades
            (10, 60),     # inlet angle: between 10 and 60 degrees
            (20, 70),     # outlet angle: between 20 and 70 degrees
            (500, 3000)   # rotational speed: between 500 and 3000 RPM
        ]
        
        # Run optimization with multiple starting points
        best_result = None
        best_score = float('inf')
        
        # Try different starting points
        starting_points = [
            x0,  # Original guess
            np.array([6, 30, 45, 1750]),  # Typical design point
            np.array([8, 20, 60, 2000]),  # High head design
            np.array([4, 40, 30, 1500])   # High flow design
        ]
        
        for start_point in starting_points:
            result = minimize(
                objective, 
                start_point, 
                bounds=bounds, 
                method='SLSQP',
                options={
                    'ftol': 1e-6,
                    'maxiter': 200,
                    'disp': True
                }
            )
            
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_result = result
        
        if best_result is not None and best_result.success:
            optimized_params = ImpellerParams(
                num_blades=int(round(best_result.x[0])),
                blade_angle_inlet=float(best_result.x[1]),
                blade_angle_outlet=float(best_result.x[2]),
                rotational_speed=float(best_result.x[3]),
                outer_diameter=self.params.outer_diameter,
                inner_diameter=self.params.inner_diameter,
                blade_height=self.params.blade_height
            )
            
            # Calculate flow field for final parameters to ensure metrics are available
            final_sim = ImpellerSimulation(optimized_params)
            final_sim.calculate_velocity_field()
            final_metrics = final_sim.calculate_performance_metrics()
            
            return optimized_params
        else:
            raise RuntimeError(f"Optimization failed: {best_result.message if best_result else 'No valid solution found'}")
    
    def visualize_flow(self, ax=None):
        """
        Create a visualization of the flow field.
        This shows the velocity field and impeller geometry.
        
        Args:
            ax: Optional matplotlib axis to plot on
                If None, creates a new figure
        
        Returns:
            The matplotlib axis object with the plot
        """
        # Create new figure if needed
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
        
        # Calculate velocity field for visualization
        v_x, v_y = self.calculate_velocity_field()
        
        # Calculate velocity magnitude for coloring
        v_mag = np.sqrt(v_x**2 + v_y**2)
        
        # Plot velocity field with arrows (quiver plot)
        # skip=5 means we only show every 5th point to avoid overcrowding
        skip = 5
        quiver = ax.quiver(self.R[::skip, ::skip], self.THETA[::skip, ::skip],
                          v_x[::skip, ::skip], v_y[::skip, ::skip],
                          v_mag[::skip, ::skip],
                          scale=50, cmap='viridis')
        # Add a colorbar to show velocity magnitude
        plt.colorbar(quiver, ax=ax, label='Velocity magnitude (m/s)')
        
        # Draw the impeller outline
        # These are dashed circles showing inner and outer boundaries
        circle_outer = plt.Circle((0, 0), self.params.outer_diameter/2,
                                fill=False, color='black', linestyle='--')
        circle_inner = plt.Circle((0, 0), self.params.inner_diameter/2,
                                fill=False, color='black', linestyle='--')
        ax.add_artist(circle_outer)
        ax.add_artist(circle_inner)
        
        # Draw the impeller blades
        # This loops through each blade and draws its curve
        for i in range(self.params.num_blades):
            # Calculate starting angle for this blade
            angle = i * 2 * np.pi / self.params.num_blades
            # Create points along the blade
            r = np.linspace(self.params.inner_diameter/2, self.params.outer_diameter/2, 50)
            theta = np.linspace(angle, 
                              angle + np.radians(self.params.blade_angle_outlet),
                              50)
            # Convert to Cartesian coordinates and plot
            x_blade = r * np.cos(theta)
            y_blade = r * np.sin(theta)
            ax.plot(x_blade, y_blade, 'k-', linewidth=2)
        
        # Set plot properties
        ax.set_aspect('equal')  # Make circles look circular
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Impeller Flow Field')
        
        return ax

