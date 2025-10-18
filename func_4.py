import numpy as np
import pandas as pd
import sys
import json
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
from scipy.stats import norm, qmc
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

                if len(parsed_input) == 4:  # 4D function
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
        X_initial = np.load('initial_data/function_4/initial_inputs.npy')
        Y_initial = np.load('initial_data/function_4/initial_outputs.npy')
        print(f" Loaded {len(X_initial)} initial data points")
    except Exception as e:
        print(f"Warning: Could not load initial data: {e}")
        print("Using fallback initial data")
        X_initial = np.array([[0.4, 0.4, 0.4, 0.4], [0.3, 0.5, 0.4, 0.45], [0.45, 0.42, 0.38, 0.43]])
        Y_initial = np.array([0.1, -0.2, 0.3])

    # Load new data from CSV
    X_new, Y_new = load_data_safely(csv_file_path, function_id)

    if len(X_new) > 0:
        # Combine initial and new data
        X_combined = np.vstack([X_initial, X_new])
        Y_combined = np.concatenate([Y_initial, Y_new])
        print(f" Combined data: {len(X_initial)} initial + {len(X_new)} new = {len(X_combined)} total points")
    else:
        # Use only initial data if no new data
        X_combined = X_initial
        Y_combined = Y_initial
        print(f" Using initial data only: {len(X_combined)} points")

    return X_combined, Y_combined

class EnsembleBusinessOptimizer:
    """Ensemble optimizer for complex business modeling problems"""

    def __init__(self, bounds):
        self.bounds = bounds
        self.iteration = 0
        self.stagnation_counter = 0
        self.performance_history = []
        self.clustering_threshold = 0.05  # Threshold for detecting clustering
        self.global_restart_triggered = False

    def detect_clustering(self, X, threshold=None):
        """Detect if recent points are clustered (exploration failure)"""
        if threshold is None:
            threshold = self.clustering_threshold

        if len(X) < 5:
            return False

        # Check last 5 points
        recent_points = X[-5:]

        # Calculate pairwise distances
        max_distance = 0
        for i in range(len(recent_points)):
            for j in range(i+1, len(recent_points)):
                dist = np.linalg.norm(recent_points[i] - recent_points[j])
                max_distance = max(max_distance, dist)

        # Also check standard deviation across dimensions
        std_per_dim = np.std(recent_points, axis=0)
        max_std = np.max(std_per_dim)

        is_clustered = max_distance < threshold and max_std < threshold

        if is_clustered:
            print(f"  Clustering detected! Max distance: {max_distance:.4f}, Max std: {max_std:.4f}")

        return is_clustered

    def ensemble_predict(self, X, gp_model, rf_model):
        """Make ensemble predictions combining GP and RF"""
        X = np.atleast_2d(X)

        # GP predictions with uncertainty
        gp_mean, gp_std = gp_model.predict(X, return_std=True)

        # RF predictions (no uncertainty)
        rf_pred = rf_model.predict(X)

        # Ensemble weighting based on GP uncertainty
        # High uncertainty -> trust RF more, Low uncertainty -> trust GP more
        weights_gp = 1.0 / (1.0 + gp_std)  # High std -> low weight
        weights_rf = 1.0 - weights_gp

        # Normalize weights
        total_weights = weights_gp + weights_rf
        weights_gp = weights_gp / total_weights
        weights_rf = weights_rf / total_weights

        ensemble_mean = weights_gp * gp_mean + weights_rf * rf_pred
        ensemble_std = gp_std  # Use GP uncertainty

        return ensemble_mean, ensemble_std

    def adaptive_acquisition(self, X, gp_model, rf_model, best_y, phase="balanced"):
        """Adaptive acquisition function that switches strategy based on phase"""
        X = np.atleast_2d(X)

        # Get ensemble predictions
        mean, std = self.ensemble_predict(X, gp_model, rf_model)
        std = np.maximum(std, 1e-9)

        if phase == "exploration":
            # High exploration - Upper Confidence Bound with high kappa
            kappa = 2.5
            acquisition = mean + kappa * std

        elif phase == "exploitation":
            # High exploitation - Probability of Improvement
            xi = 0.001  # Very small xi for aggressive exploitation
            z = (mean - best_y - xi) / std
            acquisition = norm.cdf(z)

        elif phase == "recovery":
            # Recovery from poor performance - Expected Improvement with diversity
            xi = 0.1  # Higher xi to encourage improvement
            improvement = mean - best_y - xi
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)

            # Add diversity bonus
            diversity_bonus = np.random.random(len(mean)) * 0.1
            acquisition = ei + diversity_bonus

        else:  # "balanced"
            # Balanced Expected Improvement
            xi = 0.01
            improvement = mean - best_y - xi
            z = improvement / std
            acquisition = improvement * norm.cdf(z) + std * norm.pdf(z)

        return acquisition

    def global_restart_search(self, n_candidates=2000):
        """Perform global restart when trapped in local optima"""
        print(" Executing GLOBAL RESTART - searching entire space")

        candidates = []

        # 1/2 Latin Hypercube Sampling for good coverage
        sampler = qmc.LatinHypercube(d=len(self.bounds), seed=42 + self.iteration)
        lhs_samples = sampler.random(n_candidates // 2)

        for sample in lhs_samples:
            scaled = []
            for i, (low, high) in enumerate(self.bounds):
                scaled.append(low + sample[i] * (high - low))
            candidates.append(scaled)

        # 1/2 Random uniform sampling
        random_samples = np.random.uniform(
            [b[0] for b in self.bounds],
            [b[1] for b in self.bounds],
            size=(n_candidates // 2, len(self.bounds))
        )
        candidates.extend(random_samples)

        self.global_restart_triggered = True
        return np.array(candidates)

    def determine_optimization_phase(self, y):
        """Determine what phase of optimization we're in"""
        if len(y) < 5:
            return "exploration"

        # Check recent performance
        recent_best = np.max(y[-5:])
        overall_best = np.max(y)

        # Check for severe degradation (like the -5.5 drop)
        if len(y) > 10:
            previous_best = np.max(y[-10:-5])
            if recent_best < previous_best - 2.0:  # Severe drop
                return "recovery"

        # Check improvement trend
        if len(y) >= 10:
            recent_trend = np.mean(y[-5:]) - np.mean(y[-10:-5])
            if recent_trend > 0.1:
                return "exploitation"  # Improving, exploit more
            elif recent_trend < -0.5:
                return "recovery"  # Degrading badly

        # Check stagnation
        if len(y) >= 8:
            recent_var = np.var(y[-5:])
            if recent_var < 0.01:  # Very low variance = stagnation
                return "exploration"

        return "balanced"

    def get_next_candidate(self, X, y, gp_model, rf_model):
        """Get next candidate with adaptive strategy"""
        self.iteration += 1
        self.performance_history.append(np.max(y) if len(y) > 0 else -np.inf)

        # Detect clustering (exploration failure)
        clustering_detected = self.detect_clustering(X)

        # Determine optimization phase
        phase = self.determine_optimization_phase(y)
        print(f" Optimization phase: {phase}")

        # Force global restart if clustering detected or severe performance issues
        best_y = np.max(y)
        if clustering_detected or (len(y) > 5 and best_y < -5.0):
            candidates = self.global_restart_search()
            phase = "recovery"
        else:
            # Normal candidate generation
            n_candidates = 1500
            candidates = []

            # Mix of sampling strategies
            # 1/3 Latin Hypercube
            sampler = qmc.LatinHypercube(d=len(self.bounds), seed=42 + self.iteration)
            lhs_samples = sampler.random(n_candidates // 3)

            for sample in lhs_samples:
                scaled = []
                for i, (low, high) in enumerate(self.bounds):
                    scaled.append(low + sample[i] * (high - low))
                candidates.append(scaled)

            # 1/3 Random uniform
            random_samples = np.random.uniform(
                [b[0] for b in self.bounds],
                [b[1] for b in self.bounds],
                size=(n_candidates // 3, len(self.bounds))
            )
            candidates.extend(random_samples)

            # 1/3 Around best points with noise
            if len(X) > 0:
                best_indices = np.argsort(y)[-min(3, len(y)):]  # Top 3 points
                best_points = X[best_indices]

                samples_per_point = (n_candidates // 3) // len(best_points)
                for point in best_points:
                    for _ in range(samples_per_point):
                        noisy_point = point + np.random.normal(0, 0.1, len(point))
                        # Clip to bounds
                        for i, (low, high) in enumerate(self.bounds):
                            noisy_point[i] = np.clip(noisy_point[i], low, high)
                        candidates.append(noisy_point)

            candidates = np.array(candidates)

        # Evaluate acquisition function
        acquisitions = self.adaptive_acquisition(
            candidates, gp_model, rf_model, best_y, phase
        )

        # Select best candidate
        best_idx = np.argmax(acquisitions)
        next_candidate = candidates[best_idx]

        print(f"Selected from {len(candidates)} candidates in {phase} phase")
        print(f"Next recommended query: {'-'.join([f'{x:.6f}' for x in next_candidate])}")

        return next_candidate

def main():
    print("=== FUNCTION 4: BUSINESS MODEL ENSEMBLE OPTIMIZER ===")

    # Load and combine initial + new data
    X, y = load_and_combine_data('1019_data.csv', 4)

    print(f" Final dataset - X shape: {X.shape}, y range: [{np.min(y):.3f}, {np.max(y):.3f}]")

    # Check for severe performance degradation
    if len(y) > 3 and np.min(y) < -5.0:
        print(" SEVERE PERFORMANCE DEGRADATION DETECTED!")
        print(" Latest value:", y[-1])
        print(" Best value ever:", np.max(y))

    # Define bounds for 4D business model
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    # Initialize ensemble optimizer
    optimizer = EnsembleBusinessOptimizer(bounds)

    # Fit ensemble models
    # GP with mixed kernels for complex landscapes
    kernel = (
        Matern(length_scale=0.3, nu=2.5, length_scale_bounds=(0.1, 0.8)) +
        RBF(length_scale=0.5, length_scale_bounds=(0.2, 1.0)) +
        WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-3, 0.1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        n_restarts_optimizer=15,
        random_state=42
    )

    # Random Forest for complex patterns
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    try:
        # Fit models
        gp.fit(X, y)
        rf.fit(X, y)
        print(" Ensemble models fitted successfully")

        # Get next candidate
        next_candidate = optimizer.get_next_candidate(X, y, gp, rf)

        # Ensemble prediction at next candidate
        pred_mean, pred_std = optimizer.ensemble_predict([next_candidate], gp, rf)

        print(f" Ensemble prediction: {pred_mean[0]:.4f} ± {pred_std[0]:.4f}")
        print(f" Current best: {np.max(y):.4f}")

        if optimizer.global_restart_triggered:
            print(" Global restart was triggered - exploring new regions")

        return next_candidate

    except Exception as e:
        print(f" Error in optimization: {e}")

if __name__ == "__main__":
    result = main()
