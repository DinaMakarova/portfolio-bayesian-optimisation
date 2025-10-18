import numpy as np
import pandas as pd
import sys
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, Product, WhiteKernel
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
from scipy.stats import qmc, norm
import warnings
warnings.filterwarnings('ignore')

def load_data_safely(file_path, function_id):
    """Load new data from CSV with error handling."""
    try:
        data = pd.read_csv(file_path)
        input_col, output_col = f'f{function_id}', f'f{function_id}_output'

        if input_col not in data.columns or output_col not in data.columns:
            print(f"Warning: Required columns not found for function {function_id}")
            return np.array([]), np.array([])

        all_inputs, all_outputs = [], []

        for idx, row in data.iterrows():
            try:
                input_str = str(row[input_col]).strip()

                # Handle different input formats
                if input_str.startswith('[') and input_str.endswith(']'):
                    input_str = input_str[1:-1]

                input_str = input_str.replace(',', ' ')
                parsed_input = [float(x) for x in input_str.split() if x.strip()]

                if len(parsed_input) == 2:  # 2D function
                    all_inputs.append(parsed_input)
                    all_outputs.append(row[output_col])

            except (ValueError, TypeError, AttributeError) as e:
                print(f"Warning: Skipping row {idx} due to parsing error: {e}")
                continue

        if not all_inputs:
            print(f"No valid new data points found for function {function_id}")
            return np.array([]), np.array([])

        X_new = np.array(all_inputs)
        y_new = np.array(all_outputs)

        print(f"Successfully loaded {len(X_new)} new data points for Function {function_id}")
        return X_new, y_new

    except Exception as e:
        print(f"Error loading new data: {e}")
        return np.array([]), np.array([])

def load_and_combine_data(csv_file_path, function_id):
    """Load and combine initial and new data for a given function."""

    # Load initial data
    try:
        X_initial = np.load('initial_data/function_1/initial_inputs.npy')
        Y_initial = np.load('initial_data/function_1/initial_outputs.npy')
        print(f" Loaded {len(X_initial)} initial data points")
    except Exception as e:
        print(f"Warning: Could not load initial data: {e}")
        print("Using fallback initial data")
        X_initial = np.array([[0.1, 0.1], [0.9, 0.9], [0.5, 0.5], [0.3, 0.7], [0.7, 0.3]])
        Y_initial = np.array([1e-30, 1e-29, 1e-31, 1e-28, 1e-32])

    # Load new data from CSV
    X_new, Y_new = load_data_safely(csv_file_path, function_id)

    if len(X_new) > 0:
        # Combine initial and new data
        X_combined = np.vstack([X_initial, X_new])
        Y_combined = np.concatenate([Y_initial, Y_new])
        print(f"Combined data: {len(X_initial)} initial + {len(X_new)} new = {len(X_combined)} total points")
    else:
        # Use only initial data if no new data
        X_combined = X_initial
        Y_combined = Y_initial
        print(f"Using initial data only: {len(X_combined)} points")

    return X_combined, Y_combined

class SparseMultiModalOptimizer:
    """Specialized optimizer for sparse multi-modal problems like radiation detection"""

    def __init__(self, bounds, n_modes=2):
        self.bounds = bounds
        self.n_modes = n_modes
        self.discovered_regions = []
        self.stagnation_counter = 0
        self.iteration = 0

    def sparse_acquisition_function(self, X, model, best_y, explored_points, xi=0.01):
        """Multi-scale acquisition function for sparse problems"""
        X = np.atleast_2d(X)
        mean, std = model.predict(X, return_std=True)
        std = np.maximum(std, 1e-12)  # Prevent division by zero

        # Expected Improvement base
        improvement = mean - best_y - xi
        z = improvement / std
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)

        # Diversity bonus - encourage exploration away from sampled points
        diversity_bonus = self.compute_diversity_bonus(X, explored_points)

        # Multi-scale bonus - encourage different scales of exploration
        multiscale_bonus = self.compute_multiscale_bonus(X, model)

        # Combine acquisitions
        total_acquisition = ei + 0.3 * diversity_bonus + 0.2 * multiscale_bonus

        return total_acquisition

    def compute_diversity_bonus(self, X, explored_points, min_distance=0.05):
        """Compute bonus for exploring away from existing points"""
        if len(explored_points) == 0:
            return np.ones(X.shape[0])

        distances = []
        for x in X:
            min_dist = np.min([np.linalg.norm(x - exp_point) for exp_point in explored_points])
            distances.append(min_dist)

        distances = np.array(distances)
        # Give higher bonus for points farther from explored regions
        return np.tanh(distances / min_distance)

    def compute_multiscale_bonus(self, X, model):
        """Encourage exploration at multiple scales"""
        # Create a grid of scales for exploration
        scales = [0.1, 0.2, 0.3, 0.5]  # Different exploration scales
        bonuses = []

        for x in X:
            scale_bonus = 0
            for scale in scales:
                # Sample around this point at different scales
                neighbors = self.sample_neighbors(x, scale, n_samples=5)
                neighbor_stds = model.predict(neighbors, return_std=True)[1]
                # Higher uncertainty at this scale = higher bonus
                scale_bonus += np.mean(neighbor_stds)
            bonuses.append(scale_bonus)

        return np.array(bonuses)

    def sample_neighbors(self, center, scale, n_samples=5):
        """Sample neighbors around a center point at given scale"""
        neighbors = []
        for _ in range(n_samples):
            neighbor = center + np.random.normal(0, scale, size=len(center))
            # Clip to bounds
            for i, (low, high) in enumerate(self.bounds):
                neighbor[i] = np.clip(neighbor[i], low, high)
            neighbors.append(neighbor)
        return np.array(neighbors)

    def latin_hypercube_sampling(self, n_samples):
        """Generate diverse initial samples using Latin Hypercube"""
        sampler = qmc.LatinHypercube(d=len(self.bounds), seed=42 + self.iteration)
        samples = sampler.random(n_samples)

        # Scale to bounds
        scaled_samples = []
        for sample in samples:
            scaled = []
            for i, (low, high) in enumerate(self.bounds):
                scaled.append(low + sample[i] * (high - low))
            scaled_samples.append(scaled)

        return np.array(scaled_samples)

    def global_search_candidates(self, n_candidates=1000):
        """Generate diverse candidates for global search"""
        candidates = []

        # 1/3 Latin Hypercube
        lhs_samples = self.latin_hypercube_sampling(n_candidates // 3)
        candidates.extend(lhs_samples)

        # 1/3 Random uniform
        random_samples = np.random.uniform(
            [b[0] for b in self.bounds],
            [b[1] for b in self.bounds],
            size=(n_candidates // 3, len(self.bounds))
        )
        candidates.extend(random_samples)

        # 1/3 Around discovered regions with noise
        if self.discovered_regions:
            region_samples = []
            samples_per_region = (n_candidates // 3) // len(self.discovered_regions)
            for region in self.discovered_regions:
                for _ in range(samples_per_region):
                    noisy_sample = region + np.random.normal(0, 0.1, len(region))
                    # Clip to bounds
                    for i, (low, high) in enumerate(self.bounds):
                        noisy_sample[i] = np.clip(noisy_sample[i], low, high)
                    region_samples.append(noisy_sample)
            candidates.extend(region_samples)
        else:
            # If no regions discovered, use more random samples
            extra_random = np.random.uniform(
                [b[0] for b in self.bounds],
                [b[1] for b in self.bounds],
                size=(n_candidates // 3, len(self.bounds))
            )
            candidates.extend(extra_random)

        return np.array(candidates)

    def update_discovered_regions(self, X, y, threshold_percentile=95):
        """Update list of high-value regions"""
        if len(y) < 5:
            return

        threshold = np.percentile(y, threshold_percentile)
        high_value_indices = y >= threshold

        if np.sum(high_value_indices) > 0:
            high_value_points = X[high_value_indices]

            for point in high_value_points:
                # Check if this is a new region
                is_new = True
                for existing in self.discovered_regions:
                    if np.linalg.norm(point - existing) < 0.1:  # Within 0.1 distance
                        is_new = False
                        break

                if is_new and len(self.discovered_regions) < self.n_modes:
                    self.discovered_regions.append(point)
                    print(f" New high-value region discovered: [{point[0]:.3f}, {point[1]:.3f}]")

    def get_next_candidate(self, X, y, model):
        """Get next candidate point for evaluation"""
        self.iteration += 1

        # Update discovered regions
        self.update_discovered_regions(X, y)

        # Check for stagnation
        if len(y) > 3:
            recent_best = np.max(y[-3:])
            if len(y) > 6:
                previous_best = np.max(y[-6:-3])
                if abs(recent_best - previous_best) < 1e-15:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0

        # If stagnating, do global search
        if self.stagnation_counter > 2:
            print(" Detected stagnation - performing global search")
            candidates = self.global_search_candidates(2000)
            best_y = np.max(y)

            # Evaluate acquisition for all candidates
            acquisitions = self.sparse_acquisition_function(
                candidates, model, best_y, X
            )

            best_idx = np.argmax(acquisitions)
            self.stagnation_counter = 0  # Reset counter
            return candidates[best_idx]

        # Normal optimization
        candidates = self.global_search_candidates(1000)
        best_y = np.max(y)

        acquisitions = self.sparse_acquisition_function(
            candidates, model, best_y, X
        )

        best_idx = np.argmax(acquisitions)
        return candidates[best_idx]

def main():
    print("=== FUNCTION 1: SPARSE RADIATION DETECTION OPTIMIZER ===")

    # Load and combine initial + new data
    X, y = load_and_combine_data('1019_data.csv', 1)

    print(f" Final dataset - X shape: {X.shape}, y range: [{np.min(y):.2e}, {np.max(y):.2e}]")

    # Check if all values are essentially zero (radiation not detected)
    max_val = np.max(np.abs(y))
    if max_val < 1e-20:
        print("  WARNING: All values near machine precision - radiation sources not found!")
        print(" Deploying emergency sparse optimization protocol...")

    # Define bounds for 2D radiation detection
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    # Initialize sparse multi-modal optimizer
    optimizer = SparseMultiModalOptimizer(bounds, n_modes=2)

    # Multi-scale kernel for sparse problems
    kernel = (
        RBF(length_scale=0.1, length_scale_bounds=(0.01, 0.5)) *  # Fine scale
        Matern(length_scale=0.5, nu=2.5, length_scale_bounds=(0.1, 1.0)) +  # Coarse scale
        WhiteKernel(noise_level=1e-20, noise_level_bounds=(1e-25, 1e-15))  # Numerical noise
    )

    # Fit Gaussian Process
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-20,  # Very small regularization for sparse problems
        n_restarts_optimizer=10,
        random_state=42
    )

    try:
        gp.fit(X, y)
        print("GP model fitted successfully")

        # Get next candidate
        next_candidate = optimizer.get_next_candidate(X, y, gp)
        print(f"Next recommended query point: {next_candidate[0]:.6f}-{next_candidate[1]:.6f}")
        print(f"Current best value: {np.max(y):.6e}")
        print(f" Discovered regions: {len(optimizer.discovered_regions)}")

        # Predict at next candidate
        pred_mean, pred_std = gp.predict([next_candidate], return_std=True)
        print(f" Model prediction: {pred_mean[0]:.6e} ± {pred_std[0]:.6e}")

        return next_candidate

    except Exception as e:
        print(f" Error in optimization: {e}")
        # Fallback to global search
        candidates = np.random.uniform(0, 1, size=(1000, 2))
        next_candidate = candidates[np.random.randint(1000)]
        print(f"Fallback random candidate: [{next_candidate[0]:.6f}, {next_candidate[1]:.6f}]")
        return next_candidate

if __name__ == "__main__":
    result = main()
