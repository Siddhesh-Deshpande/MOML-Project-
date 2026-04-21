from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import is_pareto_efficient


def extract_pareto_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    objective_matrix = df[["obj_accuracy_min", "obj_inference_ms", "obj_model_params"]].to_numpy()
    mask = is_pareto_efficient(objective_matrix)
    pareto_df = df.loc[mask].copy()
    pareto_df = pareto_df.sort_values(by="accuracy", ascending=False)
    return pareto_df


def plot_pareto_2d(df: pd.DataFrame, output_path: str | Path) -> None:
    plt.figure(figsize=(9, 6))
    plt.scatter(df["inference_ms"], df["model_params"], alpha=0.4, label="All Trials")
    plt.scatter(
        df[df["is_pareto"]]["inference_ms"],
        df[df["is_pareto"]]["model_params"],
        c="red",
        s=60,
        label="Pareto Points",
    )
    plt.xlabel("Inference Time (ms/sample)")
    plt.ylabel("Model Parameters")
    plt.title("Pareto Front Projection: Efficiency vs Complexity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pareto_3d(df: pd.DataFrame, output_path: str | Path) -> None:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        df["accuracy"],
        df["inference_ms"],
        df["model_params"],
        alpha=0.3,
        label="All Trials",
    )

    pareto = df[df["is_pareto"]]
    ax.scatter(
        pareto["accuracy"],
        pareto["inference_ms"],
        pareto["model_params"],
        c="red",
        s=60,
        label="Pareto",
    )

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Inference ms/sample")
    ax.set_zlabel("Model Parameters")
    ax.set_title("3D Pareto Front")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_parallel_coordinates(df: pd.DataFrame, output_path: str | Path) -> None:
    pareto = df[df["is_pareto"]].copy()
    if pareto.empty:
        return

    features = ["accuracy", "inference_ms", "model_params", "noisy_accuracy"]
    normalized = pareto[features].copy()

    for column in features:
        col_min = normalized[column].min()
        col_max = normalized[column].max()
        if col_max - col_min < 1e-12:
            normalized[column] = 0.5
        else:
            normalized[column] = (normalized[column] - col_min) / (col_max - col_min)

    plt.figure(figsize=(10, 6))
    for _, row in normalized.iterrows():
        plt.plot(features, row.values, alpha=0.6)

    plt.title("Parallel Coordinates (Normalized) for Pareto Solutions")
    plt.ylabel("Normalized Value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
