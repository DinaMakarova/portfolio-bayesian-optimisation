# Model Card: Bayesian Optimization Framework for Multi-Function Black-Box Optimization

## Model Description

**Input:** 
- Normalized d-dimensional vectors in [0,1]^d where d ∈ {2,3,4,5,6,8}
- Historical query data: previous input-output pairs {(x₁,y₁), ..., (xₙ,yₙ)}
- Function-specific bounds and constraints

**Output:** 
- Recommended next query point x* ∈ [0,1]^d
- Predicted function value μ(x*) with uncertainty σ(x*)
- Acquisition function landscape for visualization

**Model Architecture:**

**Primary Component:** Gaussian Process Regression with function-specific kernel configurations

**Kernel Architectures:**
- **Standard:** Matérn kernel with ν ∈ {1.5, 2.5} for smooth non-analytic functions
- **Noise-aware:** Matérn + WhiteKernel for explicit observation noise modeling (F2)
- **Multi-scale:** (RBF × Matérn) + WhiteKernel for sparse multi-modal functions (F1)
- **ARD (Automatic Relevance Determination):** Matérn with per-dimension length scales for dimensional relevance detection (F3)
- **Ensemble:** GP + Random Forest (60-40 weighting) for extreme multi-modality (F4)

**Acquisition Functions:**
- Upper Confidence Bound (UCB): μ(x) + κ·σ(x) where κ ∈ [1.5, 2.5]
- Expected Improvement (EI): 𝔼[max(0, μ(x) - y_best)]
- Custom diversity-penalized acquisition for sparse functions

**Optimisation Strategy:**
- Multi-start optimization: 10-100 random restarts based on dimensionality
- Latin Hypercube Sampling for high-dimensional exploration
- Trust region constraints (width 0.2) for exploitation phases
- Stagnation detection with automatic global search triggers

**Implementation:** Python 3.8+ using scikit-learn 1.0+, scipy 1.7+, numpy 1.21+

## Performance

**Overall Competition Results:**

| Metric | Value | Notes |
|--------|-------|-------|
| Strong performers | 3/8 functions | F3, F5, F8 achieved 90%+ of target |
| Critical failures | 3/8 functions | F1, F2, F4 showed degradation or failure |
| Moderate performance | 2/8 functions | F6, F7 with improvement but suboptimal |

**Function-Specific Performance:**

**Successes:**
- **Function 3:** 78% improvement toward zero target (-0.18 → -0.04) via dimensional reduction
- **Function 5:** +142% improvement (3,234 → 7,073) maintaining 90% of peak through trust regions
- **Function 8:** 99% of best value (9.86/9.96) via stagnation detection in 8D space

**Failures:**
- **Function 1:** Unable to detect radiation sources (all values ~10^-150)
- **Function 2:** Degraded from 0.67 to -0.16 (lost all gains)
- **Function 4:** High volatility, dropped from 0.63 to 0.04 (7% retention)

**Evaluation Metrics:**
- **Absolute improvement:** Change from initial to best observed value
- **Retention rate:** Latest value as percentage of best achieved
- **Convergence rate:** Queries required to reach 90% of final best
- **Dimensional efficiency:** Performance relative to curse of dimensionality

**Evaluation Dataset:** 8 functions × 58 observations = 464 total evaluations over 8-week competition (August-October 2025)


## Limitations

**Fundamental Limitations:**

1. **Extreme Sparsity:** Model completely fails when non-zero regions occupy small amount of domain (Function 1). Gaussian Processes assume some level of smoothness; needle-in-haystack problems require alternative methods (grid search, genetic algorithms).

2. **Query Budget Constraint:** With only a few queries per function, insufficient data to reliably model complex multi-modal landscapes (Functions 4, 7). Standard BO literature assumes 50-100+ queries.

3. **Exploitation-Exploration Balance:** No safeguards against abandoning high-performing regions led to 3 functions degrading from peak values. Model lacks "return to best" logic.

4. **High-Dimensional Curse:** Performance degrades in 8D (Function 8 improved only 1.6%) despite near-optimal final value - would fail in 10D+ without dimensionality reduction.

5. **Ensemble Instability:** GP+RF ensemble for Function 4 showed high variance; weighting scheme not adaptive to local uncertainty patterns.

**Known Failure Modes:**

- **Premature Convergence:** Trust regions can trap optimizer if activated too early (seen in preliminary F5 tests)
- **Noise Overfitting:** Without WhiteKernel, model fits smooth curves through noisy observations (corrected in F2)
- **ARD Misinterpretation:** Irrelevant dimensions must be fixed, not just observed—passive ARD insufficient
- **Acquisition Optimization:** Multi-modal acquisition functions may return local optima; insufficient restarts in high dimensions

**Computational Constraints:**

- GP training scales O(n³) with data points; becomes slow beyond 200 observations
- Acquisition optimization expensive in 6D+ (10-15 seconds per query on standard hardware)
- Memory requirements scale O(n²) for covariance matrix storage

## Trade-offs

**Exploration vs. Exploitation:**

Early aggressive exploitation (Functions 2, 4, 7) wasted promising regions when exploration resumed. Late-stage exploration (Functions 5, 8) risked losing achieved gains. Optimal balance depends on:
- Number of remaining queries
- Distance from theoretical optimum (unknown in practice)
- Function landscape characteristics (unknown initially)

**Recommendation:** Reserve 20% of query budget for continued exploitation of best region even during exploration phases.

**Model Complexity vs. Interpretability:**

- **Simple Matérn kernels:** Interpretable length scales but limited expressiveness (struggled with F1)
- **Multi-scale kernels:** Better modeling but 6+ hyperparameters hard to optimise with limited data
- **Ensemble methods:** Improved F4 but lost uncertainty calibration and interpretability

**Recommendation:** Start simple; add complexity only when specific failure mode identified.

**Function-Specific vs. Unified Framework:**

- **Separate scripts per function:** Maximum customization but code drift and maintenance burden
- **Unified framework:** Cleaner codebase but requires extensive configuration logic

**Trade-off made:** Chose separation for competition flexibility; would choose unified framework for production.

**Computational Cost vs. Optimization Quality:**

- Increasing optimizer restarts from 10 to 100 improved acquisition optimization but 10× slower
- Dense sampling (500+ candidates) better for exploration but memory-intensive in 8D

**Trade-off made:** Adaptive restarts based on dimensionality (10 for 2D-4D, 50 for 5D-6D, 100 for 8D).

**Uncertainty Quantification vs. Convergence Speed:**

- High κ (UCB=2.5) maintains exploration but slow convergence
- Low κ (UCB=1.5) fast convergence but risks missing global optimum

**Trade-off made:** Dynamic decay from κ=2.5 to κ=1.5 over competition period.

## Ethical Considerations

Not applicable. This is a synthetic optimization framework for educational purposes with no real-world deployment, no personal data, and no social impact.

If deployed for real applications, considerations would include:
- **Chemical optimization (F5):** Safety constraints on hazardous experiments
- **Drug discovery (F3):** Clinical trial ethics and regulatory compliance
- **Business modeling (F4):** Fair competition and antitrust implications
- **Radiation detection (F1):** Security and public safety protocols

## Intended Use

**Primary Use Case:** Educational demonstration of Bayesian optimization strategies under query constraints.

**Recommended Applications:**
- Hyperparameter tuning for expensive-to-train models
- Chemical/materials science optimization with limited experiments
- A/B testing with high-cost interventions
- Simulation optimization where function evaluations are expensive

**Not Recommended For:**
- Real-time systems (GP training too slow)
- Extremely high-dimensional spaces (>10D without dimensionality reduction)
- Non-smooth discontinuous functions (violates GP smoothness assumptions)
- Safety-critical applications without extensive validation

## Maintenance and Versioning
**Current Version:** 1.0 (Final competition submission)

**Maintenance Status:** No ongoing maintenance planned. Code frozen as educational artifact.