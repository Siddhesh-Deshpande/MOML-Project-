from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class EvalResults:
    clean_accuracy: float
    noisy_accuracy: float
    inference_ms_per_sample: float


def _accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / max(labels.size(0), 1)


def train_one_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer_name: str,
    learning_rate: float,
    epochs: int,
    device: torch.device,
) -> float:
    criterion = nn.CrossEntropyLoss()

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    model.to(device)
    best_val_acc = 0.0

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        val_acc = evaluate_accuracy(model, val_loader, device)
        best_val_acc = max(best_val_acc, val_acc)

    return best_val_acc


def evaluate_accuracy(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return float(correct / max(total, 1))


def evaluate_with_noise(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    noise_std: float,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            noisy_images = images + torch.randn_like(images) * noise_std
            noisy_images = torch.clamp(noisy_images, -1.0, 1.0)
            logits = model(noisy_images)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return float(correct / max(total, 1))


def measure_inference_time_ms(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    warmup_batches: int = 3,
    max_batches: int = 20,
) -> float:
    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_samples = 0
    start_time = None

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            images = images.to(device)

            if batch_idx < warmup_batches:
                _ = model(images)
                continue

            if start_time is None:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

            _ = model(images)
            total_samples += images.size(0)

            if batch_idx >= warmup_batches + max_batches - 1:
                break

    if device.type == "cuda":
        torch.cuda.synchronize()

    if start_time is None or total_samples == 0:
        return float("inf")

    elapsed = time.perf_counter() - start_time
    ms_per_sample = (elapsed / total_samples) * 1000.0
    return float(ms_per_sample)


def full_evaluation(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    noise_std: float,
    device: torch.device,
) -> EvalResults:
    clean_acc = evaluate_accuracy(model, test_loader, device)
    noisy_acc = evaluate_with_noise(model, test_loader, noise_std, device)
    inference_ms = measure_inference_time_ms(model, test_loader, device)

    return EvalResults(
        clean_accuracy=clean_acc,
        noisy_accuracy=noisy_acc,
        inference_ms_per_sample=inference_ms,
    )
