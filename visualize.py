import numpy as np
import pandas as pd
import re
import os
import sys
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm

# --- Configuration ---
DATA_FILE = '1019_data.csv'
INITIAL_DATA_DIR = 'initial_data'
OUTPUT_DIR = 'visualizations'

# --- Data Loading Utility ---
def load_all_data(function_id):
    """Loads and combines initial and new data for a given function."""
    try:
        # Load initial data
        x_initial_path = os.path.join(INITIAL_DATA_DIR, f'function_{function_id}', 'initial_inputs.npy')
        y_initial_path = os.path.join(INITIAL_DATA_DIR, f'function_{function_id}', 'initial_outputs.npy')
        print(x_initial_path, y_initial_path)
        X_initial = np.load(x_initial_path)
        Y_initial = np.load(y_initial_path)

        # Load new data from CSV
        data = pd.read_csv(DATA_FILE)
        input_col, output_col = f'f{function_id}', f'f{function_id}_output'
        
        if input_col not in data.columns or output_col not in data.columns:
            print(f"Warning: No new data found for function {function_id} in {DATA_FILE}. Using initial data only.")
            return X_initial, Y_initial

        # Handle potentially empty new data columns
        data = data.dropna(subset=[input_col, output_col])
        if data.empty:
            print(f"Warning: New data for function {function_id} is empty. Using initial data only.")
            return X_initial, Y_initial

        # Robust parsing for different string formats in the CSV
        def parse_input_string(s):
            # Remove brackets, quotes, and handle various spacing
            s_cleaned = s.strip('[]" ')
            # Split by spaces or commas, filter out empty strings
            numbers = [float(num) for num in re.split(r'[ ,]+', s_cleaned) if num]
            return numbers

        parsed_inputs = data[input_col].apply(parse_input_string)
        X_new = np.array(parsed_inputs.tolist())
        Y_new = data[output_col].values

        # Combine data
        X_sample = np.vstack([X_initial, X_new])
        Y_sample = np.concatenate([Y_initial, Y_new])
        
        return X_sample, Y_sample

    except Exception as e:
        print(f"CRITICAL: Could not load data for function {function_id}. Error: {e}")
        sys.exit(1)

# --- Acquisition Functions (for visualization) ---
def ei_acquisition(X, model, best_y, xi=0.1):
    X = np.atleast_2d(X)
    mean, std = model.predict(X, return_std=True)
    std = np.maximum(std, 1e-9)
    z = (mean - best_y - xi) / std
    return (mean - best_y - xi) * norm.cdf(z) + std * norm.pdf(z)

def ucb_acquisition(X, model, kappa=1.96):
    X = np.atleast_2d(X)
    mean, std = model.predict(X, return_std=True)
    return mean + kappa * std

# --- Visualization Functions ---

def plot_2d_function(function_id, X_sample, Y_sample, kernel, acq_func, title):
    """Generates and saves a 2D visualization of the GP model and acquisition function."""
    print(f"Generating 2D plot for Function {function_id}...")
    
    # Fit the GP model
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, normalize_y=True)
    gpr.fit(X_sample, Y_sample)

    # Create a grid to plot on
    grid_res = 100
    x1 = np.linspace(0, 1, grid_res)
    x2 = np.linspace(0, 1, grid_res)
    xx, yy = np.meshgrid(x1, x2)
    X_grid = np.vstack([xx.ravel(), yy.ravel()]).T

    # Get predictions
    y_pred, y_std = gpr.predict(X_grid, return_std=True)
    y_pred = y_pred.reshape(xx.shape)
    
    # Calculate acquisition values
    if acq_func.__name__ == 'ei_acquisition':
        best_y = np.max(Y_sample)
        acq_values = acq_func(X_grid, gpr, best_y).reshape(xx.shape)
    else: # UCB
        acq_values = acq_func(X_grid, gpr).reshape(xx.shape)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    fig.suptitle(f'Function {function_id}: {title}', fontsize=18, weight='bold')

    # Panel 1: Predicted Mean Surface
    contour1 = ax1.contourf(xx, yy, y_pred, levels=50, cmap='viridis')
    fig.colorbar(contour1, ax=ax1, label='Predicted Mean Output')
    ax1.scatter(X_sample[:, 0], X_sample[:, 1], c='red', s=50, edgecolors='k', zorder=3, label='Sampled Points')
    best_idx = np.argmax(Y_sample)
    ax1.scatter(X_sample[best_idx, 0], X_sample[best_idx, 1], c='cyan', s=150, edgecolors='k', marker='*', zorder=4, label='Best Point Found')
    ax1.set_title('GP Predicted Mean Surface', fontsize=14)
    ax1.set_xlabel('Input Dimension 1')
    ax1.set_ylabel('Input Dimension 2')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Panel 2: Acquisition Function Surface
    contour2 = ax2.contourf(xx, yy, acq_values, levels=50, cmap='plasma')
    fig.colorbar(contour2, ax=ax2, label='Acquisition Value')
    ax2.set_title('Acquisition Function Surface', fontsize=14)
    ax2.set_xlabel('Input Dimension 1')
    ax2.set_ylabel('Input Dimension 2')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, f'function_{function_id}_2d_summary.png'), dpi=150)
    plt.close(fig)

def plot_convergence(function_id, Y_sample, title):
    """Generates and saves a convergence plot."""
    print(f"Generating convergence plot for Function {function_id}...")
    
    best_y_so_far = np.maximum.accumulate(Y_sample)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(Y_sample)), best_y_so_far, marker='o', linestyle='-', color='b')
    ax.set_title(f'Function {function_id}: {title}\nConvergence Plot', fontsize=16)
    ax.set_xlabel('Number of Queries')
    ax.set_ylabel('Best Value Found So Far')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'function_{function_id}_convergence.png'), dpi=120)
    plt.close(fig)

def plot_parallel_coordinates(function_id, X_sample, Y_sample, title):
    """Generates and saves a parallel coordinate plot for the best points."""
    print(f"Generating parallel coordinates plot for Function {function_id}...")
    
    n_dims = X_sample.shape[1]
    n_top_points = min(5, len(Y_sample)) # Show up to the top 5 points
    
    # Get indices of the top N points
    top_indices = np.argsort(Y_sample)[-n_top_points:][::-1]
    
    # Create a DataFrame for plotting
    df_data = {f'Dim {i+1}': X_sample[top_indices, i] for i in range(n_dims)}
    df_data['Output'] = Y_sample[top_indices]
    df = pd.DataFrame(df_data)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    pd.plotting.parallel_coordinates(df, 'Output', colormap='viridis', ax=ax)
    ax.set_title(f'Function {function_id}: {title}\nParallel Coordinates of Top {n_top_points} Points', fontsize=16)
    ax.set_ylabel('Input Value (0 to 1)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Function Output', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'function_{function_id}_parallel_coords.png'), dpi=120)
    plt.close(fig)

# --- Main Execution Block ---
def main():
    """Main function to generate all visualizations."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: '{OUTPUT_DIR}'")

    # --- Define function-specific settings ---
    function_configs = {
        1: {'type': '2d', 'kernel': 1.0 * Matern(nu=2.5), 'acq': ei_acquisition, 'title': 'Contamination Sources'},
        2: {'type': '2d', 'kernel': 1.0 * Matern(nu=1.5) + WhiteKernel(), 'acq': ucb_acquisition, 'title': 'Noisy Model'},
        3: {'type': 'high_dim', 'title': 'Drug Discovery'},
        4: {'type': 'high_dim', 'title': 'Inaccurate Modelling'},
        5: {'type': 'high_dim', 'title': 'Chemical Yield'},
        6: {'type': 'high_dim', 'title': 'Cake Recipe'},
        7: {'type': 'high_dim', 'title': 'ML Hyperparameters'},
        8: {'type': 'high_dim', 'title': 'High-Dimensional Search'}
    }

    # --- Generate plots for each function ---
    for i in range(1, 9):
        print("\n" + "="*50)
        print(f"Processing Function {i}...")
        X, Y = load_all_data(i)
        config = function_configs[i]
        
        if config['type'] == '2d':
            plot_2d_function(i, X, Y, config['kernel'], config['acq'], config['title'])
        
        # Always generate convergence and parallel plots for all functions
        plot_convergence(i, Y, config['title'])
        if X.shape[1] > 2: # Parallel coordinates are most useful for >2D
             plot_parallel_coordinates(i, X, Y, config['title'])

    print("\n" + "="*50)
    print("All visualizations have been generated and saved in the 'visualizations' folder.")

if __name__ == '__main__':
    main()
