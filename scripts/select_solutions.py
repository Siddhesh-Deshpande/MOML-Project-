from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Select representative Pareto-optimal solutions")
    parser.add_argument("--pareto", type=str, default="outputs/pareto_front.csv", help="Pareto CSV path")
    parser.add_argument("--topk", type=int, default=4, help="Minimum number of points to export")
    parser.add_argument("--out", type=str, default="outputs/pareto_selected.csv", help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.pareto)
    if df.empty:
        raise ValueError("Pareto front is empty. Run optimization and analysis first.")

    # Representative picks: best accuracy, fastest, smallest, and balanced score.
    best_acc = df.sort_values("val_accuracy", ascending=False).head(1)
    fastest = df.sort_values("inference_ms", ascending=True).head(1)
    smallest = df.sort_values("model_params", ascending=True).head(1)

    norm = df[["val_accuracy", "inference_ms", "model_params"]].copy()
    norm["val_accuracy"] = 1.0 - (norm["val_accuracy"] - norm["val_accuracy"].min()) / (norm["val_accuracy"].max() - norm["val_accuracy"].min() + 1e-12)
    norm["inference_ms"] = (norm["inference_ms"] - norm["inference_ms"].min()) / (norm["inference_ms"].max() - norm["inference_ms"].min() + 1e-12)
    norm["model_params"] = (norm["model_params"] - norm["model_params"].min()) / (norm["model_params"].max() - norm["model_params"].min() + 1e-12)
    balance_score = norm.mean(axis=1)

    balanced = df.iloc[[int(balance_score.argmin())]]

    selected = pd.concat([best_acc, fastest, smallest, balanced], ignore_index=True)
    selected = selected.drop_duplicates(subset=["trial_number"])

    if len(selected) < args.topk:
        filler = df[~df["trial_number"].isin(selected["trial_number"])].head(args.topk - len(selected))
        selected = pd.concat([selected, filler], ignore_index=True)

    selected.to_csv(Path(args.out), index=False)
    print(f"Saved {len(selected)} selected Pareto solutions to {args.out}")


if __name__ == "__main__":
    main()
