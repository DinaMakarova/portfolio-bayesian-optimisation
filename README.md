# Bayesian Optimization Competition: Multi-Function Black-Box Optimization

## NON-TECHNICAL EXPLANATION OF THE PROJECT

This project tackles the challenge of finding optimal solutions to complex problems where we can only test limited options. This system learns from each test to intelligently predict where better solutions might exist. The framework optimises eight different synthetic functions ranging from radiation detection to chemical yield maximization, using machine learning to make each query count. Results show that customising strategies to each problem's characteristics (like reducing unnecessary dimensions or focusing search regions) dramatically outperforms one-size-fits-all approaches.

## DATA

**Source:** Synthetic competition dataset provided by the capstone program (August-October 2025)

**Dataset Composition:**
- Eight black-box functions with varying dimensionality (2D to 8D)
- Initial seed data: ~50 pre-sampled points per function (inputs and outputs stored as NumPy arrays)
- Competition queries: 8 weekly queries per function collected in `1019_data.csv`
- Total data points per function: 58 observations

**Data Format:**
- Inputs: Normalized values in [0, 1] range for all dimensions
- Outputs: Real-valued function evaluations (scales vary by function)
- Function objectives: Mix of maximization (F1, F2, F4, F5, F7, F8) and zero-targeting (F3, F6)

**Data Access:** Initial data provided in `/initial_data/function_{1-8}/` directories; query results stored in CSV format with timestamps.

**Citation:** Data generated for educational purposes by the course instructors.

## MODEL

**Primary Model:** Gaussian Process Regression with function-specific kernel architectures

**Why Gaussian Processes?**
- Natural uncertainty quantification for acquisition function optimization
- Flexibility to model diverse function landscapes through kernel selection
- Sample efficiency critical for query-limited scenarios (8 queries per function)
- Interpretable hyperparameters (length scales) reveal problem structure

**Model Variants by Function:**
- **F1 (Radiation Detection):** Multi-scale kernel (RBF × Matérn + WhiteKernel) for sparse peak detection
- **F2 (Noisy Models):** Matérn(ν=1.5) + WhiteKernel for explicit noise modeling
- **F3 (Drug Discovery):** ARD Matérn kernel enabling dimensional relevance detection
- **F4 (Business Modeling):** Ensemble (GP + Random Forest) for multi-modal landscape
- **F5-F8:** Trust region-based Matérn kernels with stagnation detection

**Framework:** Custom implementation using scikit-learn's `GaussianProcessRegressor` for maximum flexibility.

## HYPERPARAMETER OPTIMIZATION

**Kernel Hyperparameters:**
- **Length scales:** Optimized via marginal likelihood maximization (bounds: 0.01-1000.0)
- **Noise levels:** WhiteKernel noise_level bounds (1e-25 to 1e-3) tuned per function
- **Smoothness:** Matérn ν parameter (1.5 or 2.5) selected based on expected smoothness

**Acquisition Function Parameters:**
- **UCB exploration-exploitation (κ):** Initially 2.5, decayed to 1.5 for exploitation phases
- **EI exploration threshold (ξ):** 0.01-0.2 depending on multi-modality
- **Trust region width:** 0.2 for unimodal functions (F5, F8)

**Optimization Strategy:**
- Multi-start optimization: 10-100 restarts depending on dimensionality
- Latin Hypercube Sampling for high-dimensional exploration (F6-F8)
- ARD kernel hyperparameters reveal dimensional importance (F3)

**Selection Method:** Validated through weekly query performance; strategies adjusted based on convergence patterns and visualization analysis.

## RESULTS

**Strong Performers:**
- **Function 3 (Drug Discovery):** -0.038 (78% improvement toward zero target via dimensional reduction)
- **Function 5 (Chemical Yield):** 7,073 (90% of best, +142% improvement via trust regions)
- **Function 8 (High-dimensional):** 9.857

**Critical Challenges:**
- **Function 1:** Failure due to extreme sparsity
- **Functions 2, 4, 7:** Significant degradation from peak values due to insufficient exploitation safeguards

**Key Insights:**
1. **Domain-specific adaptation is decisive:** Tailored strategies (F3, F5, F8) vastly outperformed generic approaches
2. **Act on ARD insights:** Fixing irrelevant dimensions in F3 accelerated convergence
3. **Maintain achieved performance:** Functions that degraded from peaks (F2, F4, F7) wasted early gains
4. **Visualization is essential:** Parallel coordinate plots revealed dimensional patterns invisible in metrics


**Performance Summary:**

| Function | Objective | Initial | Best | Latest | Status |
|----------|-----------|---------|------|--------|--------|
| F1 | Maximize | ~0 | ~0 | ~0 |  failure |
| F2 | Maximize | 0.21 | 0.67 | -0.16 | Degraded |
| F3 | → Zero | -0.18 | -0.04 | -0.04 | Strong progress |
| F4 | Maximize | -0.14 | 0.63 | 0.04 | High volatility |
| F5 | Maximize | 3,234 | 7,836 | 7,073 | Strong performance |
| F6 | → Zero | -0.70 | -0.27 | -0.39 | Moderate |
| F7 | Maximize | 1.25 | 1.69 | 1.00 | Needs improvement |
| F8 | Maximize | 9.80 | 9.96 | 9.86 | Near-optimal |

**Lessons Learned:**
- Query-limited optimization demands persistent exploitation of discovered regions
- Bayesian optimization has fundamental limits for needle-in-haystack problems
- Ensemble methods require careful calibration to avoid volatility
- Safeguards against regression are essential when exploring after strong performance

## PROJECT STRUCTURE

```
portfolio-bayesian-optimisation/
├── README.md                    # This file
├── data_sheet.md               # Dataset documentation
├── model_card.md               # Model specifications and limitations
├── Bayesian-optimization_report.md  # Detailed weekly progress log
├── main.py                     # Master execution script
├── func_1.py to func_8.py     # Function-specific optimization scripts
├── visualize.py                # Convergence and diagnostic plotting
├── 1019_data.csv              # Competition query results
├── initial_data/               # Seed data for each function
│   └── function_{1-8}/
│       ├── initial_inputs.npy
│       └── initial_outputs.npy
└── image.jpg                   # Function visualization example
```

## HOW TO RUN

```bash
# Run all function optimizations
python main.py

# Run specific function
python func_3.py

# Generate visualizations
python visualize.py
```

**Requirements:** Python 3.8+, NumPy, pandas, scikit-learn, scipy, matplotlib

---

**Competition Period:** August - October 2025  
**Framework:** Custom Bayesian Optimization using scikit-learn
