# Datasheet: Bayesian Optimization Competition Dataset

## Motivation

**For what purpose was the dataset created?**

This dataset was created for an educational capstone competition to learn about Bayesian optimization strategies under realistic query constraints. The dataset simulates real-world black-box optimization scenarios where function evaluations are expensive and limited (e.g., chemical experiments, hyperparameter tuning, A/B testing).

**Who created the dataset and who funded it?**

Created by course instructors for the Machine Learning and AI capstone program (2025). The functions are synthetic (algorithmically generated) to enable controlled competition conditions while mimicking practical optimization challenges.

## Composition

**What do the instances represent?**

Each instance represents a single function evaluation comprising:
- **Input:** A point in normalized d-dimensional space [0,1]
- **Output:** The scalar function value at that input location

The dataset contains eight distinct functions modeling different domains:
1. **Function 1 (2D):** Radiation source detection - extremely sparse with near-zero baseline
2. **Function 2 (2D):** Log-likelihood estimation - noisy observations
3. **Function 3 (3D):** Drug adverse reaction minimization - target value of zero
4. **Function 4 (4D):** Business model comparison - multi-modal landscape
5. **Function 5 (4D):** Chemical yield maximization - smooth unimodal
6. **Function 6 (5D):** Recipe optimization - multi-objective (target zero)
7. **Function 7 (6D):** ML hyperparameter tuning - moderate dimensionality
8. **Function 8 (8D):** High-dimensional standard optimization

**How many instances are there?**

- **Initial seed data:** ~50 instances per function (400 total)
- **Competition queries:** 8 queries per function (64 total)
- **Total per function:** 58 observations
- **Grand total:** 464 function evaluations

**Is there missing data?**

I missed the first few weeks of the competition which resulted in less data points.

**Does the dataset contain confidential data?**

No. All data is synthetically generated for educational purposes. No real-world sensitive information, personal data, or proprietary information is included.

## Collection Process

**How was the data acquired?**

- **Initial data:** Provided by course instructors
- **Competition queries:** Submitted query points weekly; evaluations returned by automated competition system
- **Query frequency:** One query per function per week
- **Evaluation method:** Deterministic function evaluation (except F2 which includes controlled noise)

**Over what time frame was data collected?**

- **Initial data:** Generated before competition started
- **Competition data:** August - October, 2025 (8 weekly query rounds)
- **Data frozen:** October 2025

## Preprocessing/Cleaning/Labeling

**Was any preprocessing applied?**

- **Input normalization:** All function inputs normalized to [0, 1] for consistency
- **Output scaling:** Left in natural units to preserve function characteristics
  - F1: Order ~10^-150 (near-zero)
  - F2: Range approximately [-0.2, 0.7]
  - F3: Range approximately [-0.2, 0.0] (target zero)
  - F4: Range approximately [-5.5, 0.7]
  - F5: Range approximately [3000, 8000]
  - F6: Range approximately [-1.2, -0.2] (target zero)
  - F7: Range approximately [0.9, 1.7]
  - F8: Range approximately [9.8, 10.0]

**Was raw data saved?**

Yes. Initial seed data saved as NumPy arrays (`.npy` files). Competition queries stored in CSV format with timestamps, student IDs, inputs, and outputs.

## Uses

**What tasks has the dataset been used for?**

- Teaching Bayesian optimization methodology
- Comparing acquisition function strategies (UCB, EI, PI)
- Demonstrating dimensional reduction techniques (ARD kernels)
- Illustrating trust region optimization for exploitation
- Exploring multi-modal optimization challenges
- Understanding noise modeling in Gaussian Processes

**What tasks should the dataset NOT be used for?**

- Real-world prediction tasks (synthetic data)
- Benchmarking against published optimization algorithms (competition-specific design)
- Financial or safety-critical decision making
- Generalizing to truly unknown function properties

**Are there tasks where the dataset should be used with caution?**

Yes. Function 1's extreme sparsity makes it unsuitable for demonstrating standard Bayesian optimization success. Using it as a "success story" would misrepresent the method's capabilities.

## Distribution

**How is the dataset distributed?**

- Initial data: Provided by course instructors
- Format: NumPy arrays (.npy) for initial data; CSV for query history
- License: Educational use only (not publicly licensed)

**When was the dataset first distributed?**

- Initial data: When competition started
- Complete dataset: October 2025 (competition end)

## Maintenance

**Who maintains the dataset?**

Course instructors maintain the dataset. No ongoing updates planned post-competition.

**Will the dataset be updated?**

No. Dataset is frozen as of October 2025, representing the final competition state.

**If the dataset relates to people, are applicable limits on retention period?**

Not applicable - dataset contains no personal information beyond anonymised student IDs for competition tracking.

**Will older versions be accessible?**

Weekly snapshots exist showing progressive query history. 

## Legal & Ethical Considerations

**Data Protection Compliance:** Not applicable (synthetic data, no personal information)

**Ethical Review:** Not required (educational synthetic dataset)

**Intellectual Property:** Functions are original synthetic creations by course instructors

**Bias Considerations:** Functions designed to test diverse optimization scenarios; no demographic or social bias present in synthetic mathematical functions
