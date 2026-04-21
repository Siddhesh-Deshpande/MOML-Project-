from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.moo import run_optimization
from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-objective optimization for Fashion-MNIST")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config).data
    ensure_dir(cfg["output_dir"])

    df = run_optimization(cfg)
    print(f"Optimization finished. Trials saved to {Path(cfg['output_dir']) / 'all_trials.csv'}")
    print(f"Total evaluated trials: {len(df)}")


if __name__ == "__main__":
    main()
