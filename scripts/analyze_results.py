from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import approximate_hypervolume, spacing_metric
from src.pareto import extract_pareto_dataframe, plot_parallel_coordinates, plot_pareto_2d, plot_pareto_3d
from src.utils import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Pareto front and export plots/metrics")
    parser.add_argument("--trials", type=str, default="outputs/all_trials.csv", help="CSV file from optimization")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    figures_dir = ensure_dir(Path(outdir) / "figures")

    df = pd.read_csv(args.trials)
    pareto_df = extract_pareto_dataframe(df)

    pareto_trial_numbers = set(pareto_df["trial_number"].tolist())
    df["is_pareto"] = df["trial_number"].isin(pareto_trial_numbers)

    df.to_csv(Path(outdir) / "all_trials_with_pareto_flag.csv", index=False)
    pareto_df.to_csv(Path(outdir) / "pareto_front.csv", index=False)

    plot_pareto_2d(df, Path(figures_dir) / "pareto_2d.png")
    plot_pareto_3d(df, Path(figures_dir) / "pareto_3d.png")
    plot_parallel_coordinates(df, Path(figures_dir) / "parallel_coordinates.png")

    minimization_points = pareto_df[["obj_accuracy_min", "obj_inference_ms", "obj_model_params"]].to_numpy()
    if len(minimization_points) > 0:
        reference_point = np.max(minimization_points, axis=0) * 1.1
        hv = approximate_hypervolume(minimization_points, reference_point)
        spacing = spacing_metric(minimization_points)
    else:
        hv = 0.0
        spacing = 0.0

    metrics_payload = {
        "num_trials": int(len(df)),
        "num_pareto_points": int(len(pareto_df)),
        "hypervolume_approx": float(hv),
        "spacing": float(spacing),
    }

    write_json(Path(outdir) / "metrics_summary.json", metrics_payload)
    print("Analysis complete.")
    print(f"Pareto points: {len(pareto_df)}")
    print(f"Saved metrics to {Path(outdir) / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
