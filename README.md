# Multi-Objective Optimization for Fashion-MNIST

This repository implements **Application 1: Image Classification** from the assignment PDF using **Fashion-MNIST** and **Optuna NSGA-II**.

## What This Project Optimizes

Exactly 3 objectives are optimized simultaneously:

1. **Maximize classification accuracy** (implemented as minimizing `1 - accuracy`).
2. **Minimize inference time** (ms per sample).
3. **Minimize model size** (number of trainable parameters).

The code also logs **noise robustness accuracy** (Gaussian perturbation) for qualitative trade-off discussion.

## Decision Variables

The MOO search space includes:

- Number of convolutional layers
- Channels per conv layer
- Number of fully connected layers
- Hidden units per FC layer
- Learning rate (log scale)
- Batch size
- Number of training epochs
- Dropout rate
- Optimizer type (Adam/SGD)
- Input resolution (resized image)

All ranges are configurable in [configs/default.yaml](configs/default.yaml).

## Project Structure

- [configs/default.yaml](configs/default.yaml): Search ranges and experiment settings
- [scripts/run_optimization.py](scripts/run_optimization.py): Runs NSGA-II optimization loop
- [scripts/analyze_results.py](scripts/analyze_results.py): Extracts Pareto front, plots, and metrics
- [scripts/select_solutions.py](scripts/select_solutions.py): Picks at least 4 representative Pareto points
- [src/data.py](src/data.py): Fashion-MNIST loading/splitting/subsampling
- [src/model.py](src/model.py): Dynamic CNN generated from decision variables
- [src/train_eval.py](src/train_eval.py): Train + evaluate (clean, noisy, inference timing)
- [src/moo.py](src/moo.py): Optuna objective + trial logging
- [src/pareto.py](src/pareto.py): Pareto extraction and visualization
- [src/metrics.py](src/metrics.py): Hypervolume/spacing/GD utilities

## Environment Setup

If you already have a conda env named `vrenv`, install dependencies there:

```bash
conda activate vrenv
pip install -r requirements.txt
```

Or create a fresh environment:

```bash
conda env create -f environment.yml
conda activate moml-fashion
```

## How to Run

### One-Command Pipeline

```bash
bash scripts/run_pipeline.sh
```

Optional arguments:

```bash
bash scripts/run_pipeline.sh [config_path] [outdir] [topk]
```

Example:

```bash
bash scripts/run_pipeline.sh configs/default.yaml outputs 4
```

This runs optimization, analysis, and Pareto solution selection in sequence.

1. Optimize:

```bash
python scripts/run_optimization.py --config configs/default.yaml
```

2. Analyze and plot Pareto front:

```bash
python scripts/analyze_results.py --trials outputs/all_trials.csv --outdir outputs
```

3. Select at least 4 Pareto points for tabulation:

```bash
python scripts/select_solutions.py --pareto outputs/pareto_front.csv --topk 4 --out outputs/pareto_selected.csv
```

## Expected Outputs

Generated under `outputs/`:

- `all_trials.csv`
- `all_trials_with_pareto_flag.csv`
- `pareto_front.csv`
- `pareto_selected.csv`
- `metrics_summary.json`
- `figures/pareto_2d.png`
- `figures/pareto_3d.png`
- `figures/parallel_coordinates.png`

## Notes for Report

- Hypervolume is approximated with Monte Carlo (`approximate_hypervolume` in [src/metrics.py](src/metrics.py)).
- Spacing metric is included.
- A template is provided in [reports/report_template.md](reports/report_template.md).
