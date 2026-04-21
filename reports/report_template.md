# Report Template (Max 6 Pages)

## 1. Problem Setup

- Application: Image Classification (Fashion-MNIST)
- MOO framework: Optuna with NSGA-II
- Objectives (exactly 3):
  - Maximize accuracy
  - Minimize inference time
  - Minimize model parameters
- Decision variables and ranges: copy from [configs/default.yaml](../configs/default.yaml)

## 2. Experimental Setup

- Dataset splits and subset sizes
- Hardware details (CPU/GPU)
- Trial budget and total wall-clock time
- Random seed and reproducibility details

## 3. Quantitative Analysis

- Pareto front figures:
  - 2D scatter (inference time vs model size)
  - 3D scatter (accuracy, inference time, model size)
  - Parallel coordinates
- Metrics:
  - Hypervolume (reference point definition)
  - Spacing metric
  - Generational distance (if comparing algorithms)

## 4. Qualitative Analysis

- Main trade-offs observed
- One non-obvious trade-off with explanation
- Practical recommendations for deployment scenarios
- Strengths and weaknesses of NSGA-II in this setup

## 5. Pareto Point Tabulation (At least 4)

Use [outputs/pareto_selected.csv](../outputs/pareto_selected.csv) and include:

- Trial ID
- Accuracy
- Inference time
- Model parameters
- Relevant decision variable settings

## 6. Optional Algorithm Comparison

Compare with another MOO algorithm (e.g., qParEGO or MOEA/D):

- Setup differences
- Pareto front quality comparison
- Metric comparison

## 7. Appendix: One Pareto-Optimal Solution in Detail

- Full decision variable vector
- Exact objective values
- Why this point is useful in practice
