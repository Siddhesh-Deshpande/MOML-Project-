from __future__ import annotations

from typing import Iterable

import numpy as np


def is_pareto_efficient(points: np.ndarray) -> np.ndarray:
    """Return a boolean mask where True means nondominated (for minimization)."""
    n_points = points.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if not is_efficient[i]:
            continue
        dominates = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
        is_efficient[dominates] = False

    return is_efficient


def spacing_metric(points: np.ndarray) -> float:
    """Spacing metric for Pareto set diversity (lower is better)."""
    if len(points) < 2:
        return 0.0

    distances = []
    for i in range(len(points)):
        others = np.delete(points, i, axis=0)
        dists = np.linalg.norm(others - points[i], axis=1)
        distances.append(np.min(dists))

    distances = np.array(distances)
    mean_d = np.mean(distances)
    spacing = float(np.sqrt(np.sum((distances - mean_d) ** 2) / (len(points) - 1)))
    return spacing


def hypervolume_2d(points: np.ndarray, reference_point: np.ndarray) -> float:
    """Simple 2D hypervolume for minimization problems."""
    if points.shape[1] != 2:
        raise ValueError("hypervolume_2d expects exactly 2 objectives")

    sorted_points = points[np.argsort(points[:, 0])]

    hv = 0.0
    prev_x = reference_point[0]

    for x, y in sorted_points[::-1]:
        width = prev_x - x
        height = reference_point[1] - y
        if width > 0 and height > 0:
            hv += width * height
        prev_x = x

    return float(hv)


def approximate_hypervolume(points: np.ndarray, reference_point: np.ndarray) -> float:
    """Monte Carlo approximation of hypervolume for minimization."""
    if len(points) == 0:
        return 0.0

    mins = np.min(points, axis=0)
    maxs = reference_point
    if np.any(maxs <= mins):
        return 0.0

    n_samples = 50_000
    random_samples = np.random.uniform(low=mins, high=maxs, size=(n_samples, points.shape[1]))

    dominated = np.zeros(n_samples, dtype=bool)
    for p in points:
        dominated |= np.all(random_samples >= p, axis=1)

    box_volume = np.prod(maxs - mins)
    return float(box_volume * np.mean(dominated))


def generational_distance(approx_front: np.ndarray, reference_front: np.ndarray) -> float:
    """Compute GD from approximate front to reference front (lower is better)."""
    if len(approx_front) == 0 or len(reference_front) == 0:
        return float("inf")

    distances = []
    for p in approx_front:
        d = np.min(np.linalg.norm(reference_front - p, axis=1))
        distances.append(d)

    return float(np.mean(distances))
