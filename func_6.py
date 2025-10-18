import numpy as np
import pandas as pd
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def load_data_safely(file_path, function_id):
    """Safely loads and parses data with improved error handling."""
    try:
        data = pd.read_csv(file_path)
        input_col, output_col = f'f{function_id}', f'f{function_id}_output'
        
        if input_col not in data.columns or output_col not in data.columns:
            raise ValueError(f"Required columns not found for function {function_id}")

        all_inputs = []
        all_outputs = []
        
        for idx, row in data.iterrows():
            try:
                input_str = str(row[input_col]).strip('[]"').replace(',', ' ')
                parsed_input = [float(x) for x in input_str.split()]
                if len(parsed_input) == 5:  # 5D function (5 ingredients)
                    all_inputs.append(parsed_input)
                    all_outputs.append(row[output_col])
            except (ValueError, TypeError) as e:
                print(f"Warning: Skipping row {idx} due to parsing error: {e}")
                continue
        
        if not all_inputs:
            raise ValueError("No valid data points found")
            
        X_new = np.array(all_inputs)
        Y_new = np.array(all_outputs)
        
        print(f"Successfully loaded {len(X_new)} data points for Function {function_id}")
        print(f"Output range: [{Y_new.min():.3f}, {Y_new.max():.3f}]")
        
        return X_new, Y_new
        
    except Exception as e:
        raise ValueError(f"Critical error loading data: {e}")

def pareto_efficient_global_optimization(X, model, reference_point=-2.0, n_samples=200):
    """
    ParEGO-inspired acquisition for multi-objective optimization.
    Assumes the output is a weighted sum of multiple objectives.
    """
    X = np.atleast_2d(X)
    
    # Get model predictions
    mean, std = model.predict(X, return_std=True)
    std = np.maximum(std, 1e-9)
    
    # Multi-objective handling:
    # Since we want to get as close to zero as possible, we model this as
    # minimizing the absolute distance from zero with uncertainty
    
    # Expected improvement towards zero
    # P(|f(x)| < |best_so_far|) where best_so_far is closest to zero
    abs_mean = np.abs(mean)
    abs_reference = abs(reference_point)
    
    # Calculate probability of improvement (getting closer to zero)
    z = (abs_reference - abs_mean) / std
    prob_improvement = norm.cdf(z)
    
    # Expected absolute improvement
    expected_improvement = (abs_reference - abs_mean) * norm.cdf(z) + std * norm.pdf(z)
    
    # Multi-objective scalarization with uncertainty
    # Weight by uncertainty to encourage exploration in uncertain regions
    scalarized_acquisition = expected_improvement * (1 + std)
    
    return scalarized_acquisition

def constraint_aware_acquisition(X, model, best_y, constraint_penalty=100.0):
    """
    Acquisition function aware that we're optimizing a cake recipe.
    Penalize extreme ingredient combinations that don't make culinary sense.
    """
    X = np.atleast_2d(X)
    
    # Standard Expected Improvement towards zero
    mean, std = model.predict(X, return_std=True)
    std = np.maximum(std, 1e-9)
    
    # We want to minimize |mean| (get closer to zero)
    abs_mean = np.abs(mean)
    abs_best = abs(best_y) if best_y != 0 else 1e-6
    
    # Expected improvement in absolute terms (closer to zero is better)
    z = (abs_best - abs_mean) / std
    ei = (abs_best - abs_mean) * norm.cdf(z) + std * norm.pdf(z)
    
    # Culinary constraints (soft penalties for unrealistic recipes)
    penalties = np.zeros(len(X))
    
    for i, x in enumerate(X):
        penalty = 0
        
        # Penalty for extreme values (recipes should be balanced)
        extreme_penalty = np.sum(np.maximum(0, x - 0.95)**2) + np.sum(np.maximum(0, 0.05 - x)**2)
        penalty += extreme_penalty * 50
        
        # Penalty for too many ingredients at maximum (unrealistic)
        max_ingredients = np.sum(x > 0.9)
        if max_ingredients > 2:
            penalty += (max_ingredients - 2) * 20
        
        # Penalty for too many ingredients at minimum (too simple)
        min_ingredients = np.sum(x < 0.1)
        if min_ingredients > 3:
            penalty += (min_ingredients - 3) * 10
        
        penalties[i] = penalty
    
    # Apply penalties
    constrained_ei = ei - penalties
    
    return constrained_ei

def diversified_search(model, bounds, best_y, n_candidates=1000, n_selected=5):
    """
    Generate diverse candidate points and select the best ones.
    """
    n_dims = len(bounds)
    
    # Generate candidate points with different strategies
    candidates = []
    
    # 1. Random sampling
    random_candidates = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(n_candidates//2, n_dims)
    )
    candidates.extend(random_candidates)
    
    # 2. Latin Hypercube Sampling for better space coverage
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=n_dims, seed=42)
    lhc_samples = sampler.random(n_candidates//2)
    
    # Scale to bounds
    lhc_candidates = []
    for sample in lhc_samples:
        scaled_sample = []
        for i, (low, high) in enumerate(bounds):
            scaled_sample.append(low + sample[i] * (high - low))
        lhc_candidates.append(scaled_sample)
    
    candidates.extend(lhc_candidates)
    candidates = np.array(candidates)
    
    # Evaluate acquisition function
    acquisition_values = constraint_aware_acquisition(candidates, model, best_y)
    
    # Select top candidates
    top_indices = np.argsort(acquisition_values)[-n_selected:]
    top_candidates = candidates[top_indices]
    
    # Return the best one
    return top_candidates[-1]

def clip_query(query, bounds):
    """Clips query to valid bounds with small margin."""
    clipped = []
    for i, (low, high) in enumerate(bounds):
        clipped.append(np.clip(query[i], low + 1e-6, high - 1e-6))
    return np.array(clipped)

# Main execution
try:
    print("=" * 70)
    print("FUNCTION 6: Multi-objective Cake Recipe Optimization")
    print("Objectives: Flavor, Consistency, Calories, Waste, Cost -> Sum closest to 0")
    print("=" * 70)
    
    # Load initial data
    X_initial = np.load('initial_data/function_6/initial_inputs.npy')
    Y_initial = np.load('initial_data/function_6/initial_outputs.npy')
    
    # Load new data
    X_new, Y_new = load_data_safely('1019_data.csv', 6)
    
    # Combine data
    X_sample = np.vstack([X_initial, X_new])
    Y_sample = np.concatenate([Y_initial, Y_new])
    
    print(f"\nTotal data points: {len(X_sample)}")
    
    # Find best result (closest to zero)
    abs_values = np.abs(Y_sample)
    best_idx = np.argmin(abs_values)
    best_y = Y_sample[best_idx]
    best_x = X_sample[best_idx]
    
    print(f"Current best result: {best_y:.6f} (absolute: {abs(best_y):.6f})")
    print(f"Best recipe ingredients: {np.round(best_x, 3)}")
    
    # Analyze ingredient importance
    print("\nIngredient analysis:")
    for i in range(5):
        ingredient_values = X_sample[:, i]
        print(f"  Ingredient {i+1}: range [{ingredient_values.min():.3f}, {ingredient_values.max():.3f}], "
              f"mean {ingredient_values.mean():.3f}")
    
    # Fit GP model with appropriate kernel for multi-objective
    print("\nFitting GP model for multi-objective optimization...")
    kernel = 1.0 * Matern(nu=2.5, length_scale_bounds=(0.01, 2.0))
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=20
    )
    gpr.fit(X_sample, Y_sample)
    print(f"GP kernel: {gpr.kernel_}")
    
    # Define search bounds (recipe constraints)
    search_bounds = [(0.0, 1.0)] * 5  # All ingredients 0-100%
    
    print("\nUsing constraint-aware multi-objective acquisition...")
    
    # Find next query using diversified multi-objective search
    next_query = diversified_search(gpr, search_bounds, best_y)
    
    # Clip and format
    clipped_query = clip_query(next_query, search_bounds)
    formatted_query = "-".join([f"{x:.6f}" for x in clipped_query])
    
    # Provide interpretation
    print(f"\nNext recipe prediction:")
    ingredient_names = ["Flour", "Sugar", "Eggs", "Butter", "Liquid"]  # Example names
    for i, (name, amount) in enumerate(zip(ingredient_names, clipped_query)):
        print(f"  {name}: {amount:.1%}")
    
    print("\n" + "=" * 50)
    print(f"Next recommended query: {formatted_query}")
    print("Strategy: Multi-objective ParEGO with culinary constraints")
    print("=" * 50)
    
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)