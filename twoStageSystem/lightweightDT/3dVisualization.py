import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

# Load and prepare data
data = pd.read_csv('aggregate_metrics.csv')
data['sample_rate_numeric'] = data['sample_rate'].apply(lambda x: float(x.replace('Hz', '')))

# Filter data
valid_sample_rates = ['0.5Hz', '1Hz', '2Hz']
valid_window_sizes = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
data = data[data['sample_rate'].isin(valid_sample_rates) & data['window_minutes'].isin(valid_window_sizes)]

# Normalize color scale across all plots
all_mcc_values = pd.concat([data['binary_mcc'], data['multiclass_mcc'], data['combined_mcc']])
vmin, vmax = all_mcc_values.min(), all_mcc_values.max()
norm = Normalize(vmin=vmin, vmax=vmax)

def create_surface_plot(metric_name, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(data['sample_rate_numeric'], data['window_minutes'], 
                        data[metric_name], c=data[metric_name],
                        cmap=cm.coolwarm, norm=norm, marker='o', s=50, alpha=0.7)
    
    try:
        x_range = np.linspace(data['sample_rate_numeric'].min(), data['sample_rate_numeric'].max(), 20)
        y_range = np.linspace(data['window_minutes'].min(), data['window_minutes'].max(), 20)
        X, Y = np.meshgrid(x_range, y_range)
        Z = griddata((data['sample_rate_numeric'], data['window_minutes']),
                    data[metric_name], (X, Y), method='cubic', fill_value=np.nan)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, norm=norm, alpha=0.7,
                             linewidth=0, antialiased=True)
    except Exception as e:
        print(f"Could not generate smooth surface for {metric_name}: {e}")
    
    ax.set_xlabel('Sample Rate (Hz)')
    ax.set_ylabel('Window Minutes')
    ax.set_zlabel('MCC Score')
    ax.set_title(title)
    ax.set_xticks([0.5, 1.0, 2.0])
    ax.set_xticklabels(['0.5Hz', '1Hz', '2Hz'])
    ax.set_yticks(valid_window_sizes)
    ax.view_init(elev=30, azim=-130)
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    # Add colorbar
    fig.colorbar(scatter, shrink=0.5, aspect=5, label='MCC Score')
    plt.tight_layout()
    return fig

# Create three separate figures
binary_fig = create_surface_plot('binary_mcc', 'Binary Classification MCC Performance\n(Lightweight Decision Tree)')
multiclass_fig = create_surface_plot('multiclass_mcc', 'Multiclass Classification MCC Performance\n(Extra Trees)')
combined_fig = create_surface_plot('combined_mcc', 'Combined System MCC Performance\n(Binary + Multiclass)')

plt.show()
