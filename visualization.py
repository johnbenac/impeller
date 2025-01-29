import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def create_performance_map(flow_coeffs, head_coeffs, efficiencies, operating_points, best_point):
    """Create improved performance map with efficiency contours."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create smoother contours using interpolation
    phi_grid = np.linspace(min(flow_coeffs), max(flow_coeffs), 100)
    psi_grid = np.linspace(min(head_coeffs), max(head_coeffs), 100)
    PHI, PSI = np.meshgrid(phi_grid, psi_grid)
    
    # Interpolate efficiencies for smoother contours
    from scipy.interpolate import griddata
    EFF = griddata((flow_coeffs, head_coeffs), efficiencies, (PHI, PSI), method='cubic')
    
    # Create custom colormap
    colors = ['#2c3e50', '#3498db', '#2ecc71', '#f1c40f']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    # Plot contours with improved aesthetics
    levels = np.linspace(40, 75, 15)
    contours = ax.contour(PHI, PSI, EFF, levels=levels, cmap=cmap, alpha=0.7)
    plt.clabel(contours, inline=True, fontsize=8, fmt='%1.1f%%')
    
    # Plot operating points with better styling
    scatter = ax.scatter(operating_points[:, 0], operating_points[:, 1], 
                        c=operating_points[:, 2], cmap=cmap, 
                        s=100, marker='o', edgecolor='white', linewidth=1,
                        zorder=5)
    
    # Highlight best efficiency point
    ax.scatter(best_point[0], best_point[1], c='red', s=150, marker='*',
              label='Best Efficiency Point', zorder=6, edgecolor='white', linewidth=1)
    
    # Improve grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Flow Coefficient (φ)', fontsize=10)
    ax.set_ylabel('Head Coefficient (ψ)', fontsize=10)
    ax.set_title('Performance Map with Efficiency Contours', fontsize=12, pad=15)
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(scatter)
    cbar.set_label('Efficiency (%)', fontsize=10)
    
    plt.tight_layout()
    return fig

def create_performance_heatmap(flow_rates, efficiencies, test_points):
    """Create improved performance density heatmap."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create smoother heatmap using kernel density estimation
    from scipy.stats import gaussian_kde
    x = np.linspace(5, 40, 100)
    y = np.linspace(50, 75, 100)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    kernel = gaussian_kde(np.vstack([flow_rates, efficiencies]))
    Z = np.reshape(kernel(positions), X.shape)
    
    # Create custom colormap for performance density
    colors = ['#2c3e50', '#3498db', '#2ecc71', '#f1c40f']
    cmap = LinearSegmentedColormap.from_list("performance", colors, N=100)
    
    # Plot heatmap with improved aesthetics
    im = ax.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()],
                  aspect='auto', origin='lower', cmap=cmap)
    
    # Plot test points with smaller markers
    ax.scatter(test_points[:, 0], test_points[:, 1], c='red', s=50,
              alpha=0.6, label='Test Points', edgecolor='white', linewidth=0.5)
    
    # Improve axes and labels
    ax.set_xlabel('Flow Rate (m³/s)', fontsize=10)
    ax.set_ylabel('Efficiency (%)', fontsize=10)
    ax.set_title('Performance Heatmap', fontsize=12, pad=15)
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(im)
    cbar.set_label('Performance Density', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_optimization_progress(improvements, flow_rates, test_cases):
    """Create improved optimization progress visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create color palette for different test cases
    colors = sns.color_palette("husl", len(test_cases))
    
    # Plot each test case with improved styling
    for i, (test_case, color) in enumerate(zip(test_cases, colors)):
        mask = improvements['test_case'] == test_case
        x = flow_rates[mask]
        y = improvements['improvement'][mask]
        
        # Add smoothing to the curves
        from scipy.interpolate import make_interp_spline
        if len(x) > 3:  # Only smooth if we have enough points
            x_smooth = np.linspace(x.min(), x.max(), 200)
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(x_smooth)
            ax.plot(x_smooth, y_smooth, '-', color=color, label=test_case, linewidth=2)
        else:
            ax.plot(x, y, '-o', color=color, label=test_case, linewidth=2)
    
    # Improve grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Flow Rate (m³/s)', fontsize=10)
    ax.set_ylabel('Improvement (%)', fontsize=10)
    ax.set_title('Optimization Improvement by Flow Rate', fontsize=12, pad=15)
    
    # Improve legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    return fig

def plot_efficiency_components(flow_coeffs, efficiency_components):
    """Create improved efficiency components plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create smoother curves using spline interpolation
    from scipy.interpolate import make_interp_spline
    x_smooth = np.linspace(min(flow_coeffs), max(flow_coeffs), 200)
    
    # Plot each efficiency component with improved styling and descriptions
    components = {
        'hydraulic': 'Hydraulic (Flow Losses)',
        'volumetric': 'Volumetric (Internal Leakage)',
        'mechanical': 'Mechanical (Friction)',
        'overall': 'Overall Efficiency'
    }
    colors = {
        'hydraulic': '#3498db',
        'volumetric': '#2c3e50',
        'mechanical': '#e74c3c',
        'overall': '#2ecc71'
    }
    
    # Sort data points and remove duplicates for each component
    for component, label in components.items():
        y = efficiency_components[component]
        # Get unique x,y pairs to avoid overlapping lines
        unique_pairs = list(dict.fromkeys(zip(flow_coeffs, y)))
        x_unique = [pair[0] for pair in unique_pairs]
        y_unique = [pair[1] for pair in unique_pairs]
        
        if len(x_unique) > 3:  # Only smooth if we have enough points
            spl = make_interp_spline(x_unique, y_unique, k=3)
            y_smooth = spl(x_smooth)
            # Add reasonable limits to prevent unrealistic values
            y_smooth = np.clip(y_smooth, 0, 100)
            
            ax.plot(x_smooth, y_smooth, '-', color=colors[component], 
                   label=label, linewidth=2)
        else:
            ax.plot(x_unique, y_unique, 'o-', color=colors[component],
                   label=label, linewidth=2)
    
    # Improve grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Flow Rate (m³/s)', fontsize=10)
    ax.set_ylabel('Efficiency (%)', fontsize=10)
    ax.set_title('Pump Efficiency Components\nBreakdown of Loss Mechanisms', 
                fontsize=12, pad=15)
    
    # Set reasonable y-axis limits
    ax.set_ylim([0, 100])
    
    # Improve legend with descriptions
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
             title='Efficiency Components\n(Sources of Loss)',
             title_fontsize=10,
             borderaxespad=0,
             frameon=True,
             fancybox=True,
             shadow=True)
    
    # Add explanatory text
    ax.text(0.02, -0.15,
            'Hydraulic: Losses due to fluid flow friction and turbulence\n' +
            'Volumetric: Losses due to internal fluid recirculation\n' +
            'Mechanical: Losses due to bearing friction and seals\n' +
            'Overall: Combined effect of all loss mechanisms',
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_convergence(errors, flow_rates, test_cases):
    """Create improved convergence plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create color palette for different test cases
    colors = sns.color_palette("husl", len(test_cases))
    
    # Plot each test case with improved styling and offset
    offset = 0
    for i, (test_case, color) in enumerate(zip(test_cases, colors)):
        mask = errors['test_case'] == test_case
        x = flow_rates[mask]
        y = errors['error'][mask]
        
        # Add smoothing to the curves
        if len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 200)
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(x_smooth)
            ax.plot(x_smooth, y_smooth + offset, '-', color=color,
                   label=test_case, linewidth=2)
        else:
            ax.plot(x, y + offset, '-o', color=color,
                   label=test_case, linewidth=2)
        
        offset += 0.5  # Add offset to separate overlapping lines
    
    # Improve grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Flow Rate (m³/s)', fontsize=10)
    ax.set_ylabel('Error (%)', fontsize=10)
    ax.set_title('Error vs Flow Rate', fontsize=12, pad=15)
    
    # Improve legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    return fig 