#!/usr/bin/env bash
set -euo pipefail

# One-command pipeline:
# 1) Run MOO optimization
# 2) Analyze trials and build Pareto plots/metrics
# 3) Select representative Pareto solutions

CONFIG_PATH="${1:-configs/default.yaml}"
OUTDIR="${2:-outputs}"
TOPK="${3:-4}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: config file not found at '$CONFIG_PATH'"
  echo "Usage: bash scripts/run_pipeline.sh [config_path] [outdir] [topk]"
  exit 1
fi

echo "[1/3] Running optimization with config: $CONFIG_PATH"
python scripts/run_optimization.py --config "$CONFIG_PATH"

echo "[2/3] Analyzing results and generating Pareto artifacts in: $OUTDIR"
python scripts/analyze_results.py --trials "$OUTDIR/all_trials.csv" --outdir "$OUTDIR"

echo "[3/3] Selecting top $TOPK representative Pareto solutions"
python scripts/select_solutions.py --pareto "$OUTDIR/pareto_front.csv" --topk "$TOPK" --out "$OUTDIR/pareto_selected.csv"

echo "Pipeline completed successfully."
echo "Outputs available in: $OUTDIR"
