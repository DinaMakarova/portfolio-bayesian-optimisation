import numpy as np
import pandas as pd
import re
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize

def load_new_data(file_path, function_id):
    """Loads, parses, and validates new data, raising an error on failure."""
    try:
        data = pd.read_csv(file_path)
        input_col, output_col = f'f{function_id}', f'f{function_id}_output'
        if input_col not in data.columns or output_col not in data.columns:
            raise ValueError(f"Columns for function {function_id} not found.")

        # Process each row in the input column to handle multiple data points
        parsed_inputs = data[input_col].apply(
            lambda s: [float(num) for num in s.strip('[]').split()]
        )
        
        # Convert the list of lists into a 2D numpy array
        X_new = np.array(parsed_inputs.tolist())
        Y_new = data[output_col].values
        
        if X_new.shape[0] != Y_new.shape[0]:
            raise ValueError(
                f"Mismatch in number of samples between input ({X_new.shape[0]})"
                f" and output ({Y_new.shape[0]}) for function {function_id}."
            )

        print(f"Loaded {X_new.shape[0]} new data point(s) for Function {function_id}:")
        print(f"  - New Inputs (X):\n{X_new}")
        print(f"  - New Outputs (Y): {Y_new}")
        return X_new, Y_new
    except FileNotFoundError:
        raise ValueError(f"CRITICAL: Data file not found at {file_path}. Cannot proceed.")
    except Exception as e:
        raise ValueError(f"CRITICAL: Error loading data for function {function_id}. Details: {e}")

def clip_query(query):
    """Clips the query to be strictly between 0 and 1."""
    return np.clip(query, 1e-6, 0.999999)

# --- Main Execution Block ---
try:
    # --- 1. Load Data ---
    X_initial = np.load('initial_data/function_7/initial_inputs.npy')
    Y_initial = np.load('initial_data/function_7/initial_outputs.npy')
    X_new, Y_new = load_new_data('1019_data.csv', 7)

    X_sample = np.vstack([X_initial, X_new])
    Y_sample = np.concatenate([Y_initial, Y_new])

    # --- 2. Define the Model and Acquisition Function ---
    kernel = 1.0 * Matern(nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, normalize_y=True, n_restarts_optimizer=10)
    def ucb_acquisition(X, model, kappa=1.96):
        X = np.atleast_2d(X)
        mean, std = model.predict(X, return_std=True)
        return mean + kappa * std

    # --- 3. Fit the Model ---
    print("Fitting GP model to combined data...")
    gpr.fit(X_sample, Y_sample)
    print(f"GP model fitted. Learned kernel: {gpr.kernel_}")

    # --- 4. Find the Next Point using Informed Bounds ---
    n_dims = X_sample.shape[1]
    search_bounds = [
        (0.0, 0.4), (0.0, 0.75), (0.5, 1.0),
        (0.5, 1.0), (0.0, 1.0), (0.0, 0.5)
    ]
    print(f"Using research-informed search bounds: {search_bounds}")

    def find_next_query_point(model, bounds):
        best_x = None
        best_acq_value = -np.inf
        n_restarts = 50 # Increased n_restarts
        lows = [b[0] for b in bounds]
        highs = [b[1] for b in bounds]
        for sp in np.random.uniform(low=lows, high=highs, size=(n_restarts, n_dims)):
            res = minimize(fun=lambda x: -ucb_acquisition(x, model), x0=sp, bounds=bounds, method='L-BFGS-B')
            if -res.fun >= best_acq_value:
                best_acq_value = -res.fun
                best_x = res.x
        return best_x

    print("Optimizing UCB acquisition function within informed bounds...")
    next_query = find_next_query_point(gpr, search_bounds)
    clipped_query = clip_query(next_query) # Clip the query
    formatted_query = "-".join([f"{x:.6f}" for x in clipped_query])
    print("\n" + "="*50)
    print(f"Next recommended query point: {formatted_query}")
    print("="*50)

except ValueError as e:
    print(e)
    sys.exit(1)