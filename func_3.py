import numpy as np
import pandas as pd
import re
import sys
import os
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
    parsed_input = np.array([float(num) for num in s.strip('[]').split()]).reshape(1, -1)
    
    X_new = parsed_input
    Y_new = last_row[[output_col]].values
    
    print(f"Loaded 1 new data point for Function {function_id}:")
    print(f"  - New Input (X): {X_new.flatten()}")
    print(f"  - New Output (Y): {Y_new.flatten()}")

    X_sample = np.vstack([X_initial, X_new])
    Y_sample = np.concatenate([Y_initial, Y_new])
    
    return X_sample, Y_sample

def clip_query(query):
    """Clips the query to be strictly between 0 and 1."""
    return np.clip(query, 1e-6, 0.999999)

def main():
    function_id = 3
    data_file = '1019_data.csv'

    # --- Define the irrelevant dimension based on prior ARD results ---
    # Our ARD kernel analysis consistently showed the first dimension (index 0)
    # has a huge length-scale, indicating it's irrelevant.
    IRRELEVANT_DIM_INDEX = 0
    FIXED_DIM_VALUE = 0.5 # We fix its value to the midpoint.
    print(f"STRATEGY: Fixing dimension {IRRELEVANT_DIM_INDEX} to {FIXED_DIM_VALUE} based on ARD analysis.")

    # --- 1. Load Data ---
    # We load the full 3D data, but will use a 2D slice for the GP model.
    X_sample_3d, Y_sample = load_data(data_file, function_id)
    
    # Create the 2D dataset for the model by removing the irrelevant dimension.
    X_sample_2d = np.delete(X_sample_3d, IRRELEVANT_DIM_INDEX, axis=1)

    # --- 2. Define GP Model and Acquisition Function ---
    kernel = 1.0 * Matern(length_scale_bounds=(1e-7, 1e3), nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=15)

    def ucb_acquisition(X, model, kappa=1.96):
        X = np.atleast_2d(X)
        mean, std = model.predict(X, return_std=True)
        return mean + kappa * std

    # --- 3. Fit the Model on 2D Data ---
    print(f"\nFitting GP model on reduced 2D data (Dimensions other than {IRRELEVANT_DIM_INDEX})...")
    gpr.fit(X_sample_2d, Y_sample)
    print(f"GP model fitted. Learned kernel: {gpr.kernel_}")

    # --- 4. Find Next Point in 2D, then reconstruct to 3D ---
    # The search is now more efficient as it's only over 2 dimensions.
    search_bounds_2d = [(0.0, 1.0), (0.0, 1.0)]

    def find_next_query_point(model, bounds):
        best_x = None
        best_acq_value = -np.inf
        n_restarts = 50
        for sp in np.random.uniform(0, 1, size=(n_restarts, 2)):
            res = minimize(fun=lambda x: -ucb_acquisition(x, model), x0=sp, bounds=bounds, method='L-BFGS-B')
            if -res.fun >= best_acq_value:
                best_acq_value = -res.fun
                best_x = res.x
        return best_x

    print("\nOptimizing acquisition function in 2D space...")
    next_query_2d = find_next_query_point(gpr, search_bounds_2d)
    
    # Reconstruct the 3D point by inserting the fixed value back.
    next_query_3d = np.insert(next_query_2d, IRRELEVANT_DIM_INDEX, FIXED_DIM_VALUE)
    clipped_query = clip_query(next_query_3d) # Clip the query
    print("\n" + "="*50)
    print(f"Next recommended query point (3D): {'-'.join(map(str, np.round(clipped_query, 6)))}")
    print("="*50)

if __name__ == "__main__":
    main()
