# Impeller Fluid Flow Simulator and Optimizer

A comprehensive tool for simulating, optimizing, and visualizing fluid flow through parameterized impellers. This project combines advanced fluid dynamics simulation with visualization tools to help engineers and researchers optimize impeller designs.

## Features
- **Parameterized Impeller Design**
  - Configurable blade count, angles, and dimensions
  - Real-time geometry updates
  - Automated parameter optimization

- **Advanced Simulation**
  - Fluid flow simulation using simplified Navier-Stokes equations
  - Performance metrics tracking and analysis
  - Convergence monitoring

- **Rich Visualization Suite**
  - Performance maps with efficiency contours
  - Blade loading analysis
  - Flow field visualization
  - Optimization progress tracking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/johnbenac/impeller.git
cd impeller
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Simulations and Generating Visualizations
1. Configure simulation parameters in `config.yaml`
2. Run the automated optimization and visualization:
```bash
python auto_runner.py
```

This will:
- Run all configured test cases
- Optimize impeller parameters
- Generate comprehensive performance visualizations

All results and visualizations will be saved in the `results` directory.

## Visualization Suite

### 1. Performance Map
![Performance Map](results/performance_map.png)
- Dimensionless performance characteristics (ψ-φ diagram)
- Efficiency contours showing optimal operating regions
- Operating points with efficiency values
- Best Efficiency Point (BEP) marker

### 2. Blade Loading Analysis
![Blade Loading](results/blade_loading.png)
- Pressure distribution along blade surfaces
- Velocity profiles at key meridional positions
- Comparison between pressure and suction sides
- Flow separation and loading indicators

### 3. Optimization Progress
![Optimization Progress](results/optimization_progress.png)
- Evolution of design parameters during optimization
- Convergence of objective function
- Efficiency improvements over iterations
- Parameter sensitivity analysis

### 4. Interactive Analysis Dashboard
The interactive dashboard (`results/interactive_analysis.html`) provides:
- Dynamic performance map exploration
- Real-time parameter correlation analysis
- Blade loading animations
- Velocity triangle visualizations

## Performance Metrics

### Key Indicators
- Maximum Efficiency: 84.4%
- Operating Range: 0.05-0.15 m³/s
- Optimal Flow Coefficient: 0.1-0.12

### Operating Recommendations
1. **Best Efficiency Point**: 
   - Flow Coefficient (φ): 0.11
   - Head Coefficient (ψ): 0.32
   - Efficiency: 84.4%

2. **Operating Ranges**: 
   - Normal Operation: φ = 0.08-0.14
   - Extended Operation: φ = 0.05-0.15
   - Avoid: φ > 0.15 (high flow instability)

3. **Design Parameters at BEP**:
   - Number of Blades: 4
   - Inlet Angle: 10.0°
   - Outlet Angle: 38.4°
   - Rotational Speed: 3000 RPM

## Project Structure
```
├── auto_runner.py         # Automated simulation runner
├── config.yaml           # Configuration parameters
├── impeller_sim.py       # Core simulation logic
├── plot_results.py       # Visualization generator
├── requirements.txt      # Project dependencies
├── test_impeller.py      # Test suite
└── results/              # Generated visualizations and data
```

## Contributing
Contributions are welcome! Please read our contributing guidelines and code of conduct before submitting pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this tool in your research, please cite:
```bibtex
@software{impeller_simulator,
  author = {Benac, John},
  title = {Impeller Fluid Flow Simulator and Optimizer},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/johnbenac/impeller}
}
```

## Contact
For questions, issues, or contributions, please:
- Open an issue on [GitHub](https://github.com/johnbenac/impeller/issues)
- Submit a pull request
- Contact the maintainer at [GitHub](https://github.com/johnbenac) 