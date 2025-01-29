# Impeller Fluid Flow Simulator and Optimizer

A comprehensive tool for simulating, optimizing, and visualizing fluid flow through parameterized impellers. This project combines advanced fluid dynamics simulation with interactive visualization tools to help engineers and researchers optimize impeller designs.

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
  - Interactive performance dashboards
  - Real-time flow visualization
  - Comprehensive performance plots and heatmaps
  - Efficiency analysis tools

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

### Running the Application
1. Start the interactive application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

### Running Simulations
1. Configure simulation parameters in `config.yaml`
2. Run automated simulations:
```bash
python auto_runner.py
```

3. Generate visualization results:
```bash
python plot_results.py
```

## Visualization Suite

### Available Visualizations

1. **Efficiency vs Flow Rate Plot** (`results/efficiency_flow_plot.png`)
   - Performance mapping across flow rates
   - Color-coded test case categories
   - Trend analysis with efficiency patterns

2. **Convergence Analysis** (`results/convergence_plot.png`)
   - Optimization convergence tracking
   - Per-case improvement visualization
   - Iteration performance metrics

3. **Performance Heatmap** (`results/performance_heatmap.png`)
   - 2D density visualization of performance
   - Optimal operating region identification
   - Test point distribution mapping

4. **Interactive Dashboard** (`results/interactive_analysis.html`)
   - Dynamic data exploration
   - Real-time filtering and analysis
   - Comprehensive performance metrics
   - Custom view exports

## Performance Metrics

### Key Indicators
- Maximum Efficiency: 73.0%
- Operating Range: 5-40 m/s
- Optimal Flow Range: 15-30 m/s

### Operating Recommendations
1. **Optimal Operation**: 15-30 m/s flow rate
2. **Caution Zones**: 
   - Below 10 m/s: Requires careful monitoring
   - At 35 m/s: Known efficiency drop
3. **Recovery Zone**: 40 m/s shows stable operation

## Project Structure
```
├── app.py                 # Interactive web application
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