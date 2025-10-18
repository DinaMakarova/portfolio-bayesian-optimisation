import numpy as np
import pandas as pd
import re
import sys
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import minimize

def load_data(file_path, function_id):
    """Loads and combines initial and new data for a given function."""
    # Load initial data
    initial_data_dir = 'initial_data'
    x_initial_path = os.path.join(initial_data_dir, f'function_{function_id}', 'initial_inputs.npy')
    y_initial_path = os.path.join(initial_data_dir, f'function_{function_id}', 'initial_outputs.npy')
    X_initial = np.load(x_initial_path)
    Y_initial = np.load(y_initial_path)

    # Load the single new data point from the CSV
    data = pd.read_csv(file_path).tail(1) # Only process the last row
    input_col, output_col = f'f{function_id}', f'f{function_id}_output'
    
    if input_col not in data.columns or output_col not in data.columns or data.empty:
        print(f"Warning: No new data found for function {function_id}. Using initial data only.")
        return X_initial, Y_initial

    # Process the last row
    last_row = data.iloc[0]
    s = last_row[input_col]
    parsed_input = np.array([float(num) for num in s.strip('[]').split()]).reshape(1, -1)
    
    X_new = parsed_input
    Y_new = last_row[[output_col]].values
    
    print(f"Loaded 1 new data point for Function {function_id}:")
    print(f"  - New Input (X): {X_new.flatten()}")
    print(f"  - New Output (Y): {Y_new.flatten()}")

    # Combine initial and new data
    X_sample = np.vstack([X_initial, X_new])
    Y_sample = np.concatenate([Y_initial, Y_new])
    
    return X_sample, Y_sample

def main():
    function_id = 2
    data_file = '1019_data.csv'

    # --- 1. Load Data ---
    X_sample, Y_sample = load_data(data_file, function_id)

    # --- 2. Define GP Model and Acquisition Function ---
    kernel = 1.0 * Matern(nu=1.5) + WhiteKernel(
        noise_level=0.1, noise_level_bounds=(1e-7, 1e+1)
    )
    
    gpr = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=20
    )

    def ucb_acquisition(X, model, kappa=1.96):
        X = np.atleast_2d(X)
        mean, std = model.predict(X, return_std=True)
        return mean + kappa * std

    # --- 3. Fit the Model ---
    print("\nFitting GP model with calibrated noise handling...")
    gpr.fit(X_sample, Y_sample)
    print(f"GP model fitted. Learned kernel: {gpr.kernel_}")

    # --- 4. Find Next Point ---
    search_bounds = [(0.0, 1.0), (0.0, 1.0)]

    def find_next_query_point(model, bounds):
        best_x = None
        best_acq_value = -np.inf
        n_restarts = 50
        for sp in np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(n_restarts, 2)):
            res = minimize(fun=lambda x: -ucb_acquisition(x, model), x0=sp, bounds=bounds, method='L-BFGS-B')
            if -res.fun >= best_acq_value:
                best_acq_value = -res.fun
                best_x = res.x
        return best_x

    print("\nOptimizing acquisition function to find next query...")
    next_query = find_next_query_point(gpr, search_bounds)
    
    print("\n" + "="*50)
    print(f"Next recommended query point: {'-'.join(map(str, np.round(next_query, 6)))}")
    print("="*50)

if __name__ == "__main__":
    main()
