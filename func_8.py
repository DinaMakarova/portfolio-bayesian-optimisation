import numpy as np
import pandas as pd
import re
import sys
import os
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize

def load_data(file_path, function_id):
    """Loads and combines initial and new data for a given function."""
    initial_data_dir = 'initial_data'
    x_initial_path = os.path.join(initial_data_dir, f'function_{function_id}', 'initial_inputs.npy')
    y_initial_path = os.path.join(initial_data_dir, f'function_{function_id}', 'initial_outputs.npy')
    X_initial = np.load(x_initial_path)
    Y_initial = np.load(y_initial_path)

    data = pd.read_csv(file_path).tail(1)
    input_col, output_col = f'f{function_id}', f'f{function_id}_output'
    
    if input_col not in data.columns or output_col not in data.columns or data.empty:
        print(f"Warning: No new data found for function {function_id}. Using initial data only.")
        return X_initial, Y_initial

    last_row = data.iloc[0]
    s = last_row[input_col]
    # Handle potential extra quotes in the string
    s_cleaned = s.strip('[]" ')
    parsed_input = np.array([float(num) for num in s_cleaned.split()]).reshape(1, -1)
    
    X_new = parsed_input
    Y_new = last_row[[output_col]].values
    
    print(f"Loaded 1 new data point for Function {function_id}:")
    print(f"  - New Input (X): {X_new.flatten()}")
    print(f"  - New Output (Y): {Y_new.flatten()}")

    X_sample = np.vstack([X_initial, X_new])
    Y_sample = np.concatenate([Y_initial, Y_new])
    
    return X_sample, Y_sample

def load_state(state_file):
    """Loads the optimizer state from a file."""
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return json.load(f)
    else:
        # Default state if file doesn't exist
        return {'best_y_so_far': -np.inf, 'stagnation_counter': 0}

def save_state(state_file, state):
    """Saves the optimizer state to a file."""
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=4)

def main():
    function_id = 8
    data_file = '1019_data.csv'
    state_file = 'func_8_state.json'
    STAGNATION_LIMIT = 3 # Number of weeks without improvement to trigger an escape

    # --- Load persistent state ---
    state = load_state(state_file)
    print(f"\nLoaded state: Best Y so far = {state['best_y_so_far']:.6f}, Stagnation Counter = {state['stagnation_counter']}")

    # --- 1. Load Data ---
    X_sample, Y_sample = load_data(data_file, function_id)
    n_dims = X_sample.shape[1]

    # --- 2. Define GP Model and Acquisition Function ---
    kernel = 1.0 * Matern(nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=10)

    def ucb_acquisition(X, model, kappa=1.96):
        X = np.atleast_2d(X)
        mean, std = model.predict(X, return_std=True)
        return mean + kappa * std

    # --- 3. Fit the Model ---
    print("\nFitting GP model to combined data...")
    gpr.fit(X_sample, Y_sample)
    print(f"GP model fitted. Learned kernel: {gpr.kernel_}")

    # --- 4. Update State and Define Search Bounds ---
    current_best_y_idx = np.argmax(Y_sample)
    current_best_y = Y_sample[current_best_y_idx]
    current_best_x = X_sample[current_best_y_idx]
    
    # Update stagnation counter
    if current_best_y > state['best_y_so_far']:
        print(f"\nNew best point found! Value: {current_best_y:.6f}. Resetting stagnation counter.")
        state['best_y_so_far'] = current_best_y
        state['stagnation_counter'] = 0
    else:
        state['stagnation_counter'] += 1
        print(f"\nNo improvement found. Best remains {state['best_y_so_far']:.6f}. Stagnation counter is now {state['stagnation_counter']}.")

    # Check for stagnation and decide search strategy
    if state['stagnation_counter'] >= STAGNATION_LIMIT:
        print(f"\nSTAGNATION LIMIT REACHED! Escaping trust region for a global search.")
        search_bounds = [(0.0, 1.0)] * n_dims
        # Reset counter after escaping to allow for a new local search next time
        state['stagnation_counter'] = 0 
    else:
        print(f"\nDefining trust region around best point: {np.round(current_best_x, 4)}")
        trust_region_width = 0.2
        search_bounds = []
        for i in range(n_dims):
            low = max(0.0, current_best_x[i] - trust_region_width / 2.0)
            high = min(1.0, current_best_x[i] + trust_region_width / 2.0)
            search_bounds.append((low, high))
    
    print(f"Using search bounds: {search_bounds}")

    # --- 5. Find Next Point ---
    def find_next_query_point(model, bounds):
        best_x = None
        best_acq_value = -np.inf
        n_restarts = 75 # Increased restarts for this important function
        lows = [b[0] for b in bounds]
        highs = [b[1] for b in bounds]
        for sp in np.random.uniform(low=lows, high=highs, size=(n_restarts, n_dims)):
            res = minimize(fun=lambda x: -ucb_acquisition(x, model), x0=sp, bounds=bounds, method='L-BFGS-B')
            if -res.fun >= best_acq_value:
                best_acq_value = -res.fun
                best_x = res.x
        return best_x

    print("\nOptimizing acquisition function within defined bounds...")
    next_query = find_next_query_point(gpr, search_bounds)

    # --- 6. Save State and Output ---
    save_state(state_file, state)
    print(f"State saved to {state_file}.")
    
    print("\n" + "="*50)
    print(f"Next recommended query point: {'-'.join(map(str, np.round(next_query, 6)))}")
    print("="*50)

if __name__ == "__main__":
    main()
