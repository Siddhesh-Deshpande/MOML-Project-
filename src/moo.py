from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
import pandas as pd

from src.data import build_dataloaders
from src.model import DynamicCNN
from src.train_eval import full_evaluation, train_one_model
from src.utils import count_parameters, ensure_dir, make_trial_id, select_device, set_seed


def _suggest_architecture(trial: optuna.trial.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    n_conv_layers = trial.suggest_int(
        "n_conv_layers",
        int(search_space["n_conv_layers"][0]),
        int(search_space["n_conv_layers"][1]),
    )

    conv_channels = []
    for i in range(n_conv_layers):
        conv_channels.append(
            trial.suggest_categorical(
                f"conv_channels_l{i+1}",
                search_space["channels_options"],
            )
        )

    n_fc_layers = trial.suggest_int(
        "n_fc_layers",
        int(search_space["n_fc_layers"][0]),
        int(search_space["n_fc_layers"][1]),
    )

    hidden_units = []
    for i in range(n_fc_layers):
        hidden_units.append(
            trial.suggest_categorical(
                f"hidden_units_l{i+1}",
                search_space["hidden_units_options"],
            )
        )

    architecture = {
        "n_conv_layers": n_conv_layers,
        "conv_channels": conv_channels,
        "n_fc_layers": n_fc_layers,
        "hidden_units": hidden_units,
        "input_resolution": trial.suggest_categorical("input_resolution", search_space["input_resolution_options"]),
        "dropout": trial.suggest_float("dropout", search_space["dropout"][0], search_space["dropout"][1]),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            search_space["learning_rate"][0],
            search_space["learning_rate"][1],
            log=True,
        ),
        "batch_size": trial.suggest_categorical("batch_size", search_space["batch_size_options"]),
        "epochs": trial.suggest_int("epochs", int(search_space["epochs"][0]), int(search_space["epochs"][1])),
        "optimizer": trial.suggest_categorical("optimizer", search_space["optimizer_options"]),
    }

    return architecture


def run_optimization(config: dict[str, Any]) -> pd.DataFrame:
    set_seed(int(config["seed"]))

    output_dir = ensure_dir(config["output_dir"])
    data_dir = config["dataset"]["data_dir"]
    num_workers = int(config["dataset"]["num_workers"])
    noise_std = float(config["objectives"]["gaussian_noise_std"])

    opt_cfg = config["optimization"]
    search_space = config["search_space"]

    directions = [
        "minimize",  # minimize 1 - accuracy
        "minimize",  # minimize inference time
        "minimize",  # minimize model size
    ]

    sampler = optuna.samplers.NSGAIISampler(seed=int(config["seed"]))

    study = optuna.create_study(
        directions=directions,
        sampler=sampler,
        study_name=opt_cfg["study_name"],
        storage=opt_cfg["storage"],
        load_if_exists=True,
    )

    device = select_device()

    def objective(trial: optuna.trial.Trial) -> tuple[float, float, float]:
        arch = _suggest_architecture(trial, search_space)

        loaders = build_dataloaders(
            data_dir=data_dir,
            input_resolution=int(arch["input_resolution"]),
            batch_size=int(arch["batch_size"]),
            train_subset=int(config["dataset"]["train_subset"]),
            val_subset=int(config["dataset"]["val_subset"]),
            test_subset=int(config["dataset"]["test_subset"]),
            num_workers=num_workers,
            seed=int(config["seed"]) + trial.number,
        )

        model = DynamicCNN(
            input_resolution=int(arch["input_resolution"]),
            n_conv_layers=int(arch["n_conv_layers"]),
            conv_channels=[int(x) for x in arch["conv_channels"]],
            n_fc_layers=int(arch["n_fc_layers"]),
            hidden_units=[int(x) for x in arch["hidden_units"]],
            dropout=float(arch["dropout"]),
            num_classes=10,
        )

        _ = train_one_model(
            model=model,
            train_loader=loaders.train_loader,
            val_loader=loaders.val_loader,
            optimizer_name=str(arch["optimizer"]),
            learning_rate=float(arch["learning_rate"]),
            epochs=int(arch["epochs"]),
            device=device,
        )

        eval_results = full_evaluation(model, loaders.test_loader, noise_std=noise_std, device=device)
        model_params = count_parameters(model)

        trial.set_user_attr("trial_id", make_trial_id(trial.number))
        trial.set_user_attr("accuracy", eval_results.clean_accuracy)
        trial.set_user_attr("noisy_accuracy", eval_results.noisy_accuracy)
        trial.set_user_attr("inference_ms", eval_results.inference_ms_per_sample)
        trial.set_user_attr("model_params", model_params)

        # We convert maximize-accuracy to minimization by optimizing 1 - accuracy.
        return (1.0 - eval_results.clean_accuracy, eval_results.inference_ms_per_sample, float(model_params))

    study.optimize(
        objective,
        n_trials=int(opt_cfg["n_trials"]),
        timeout=int(opt_cfg["timeout_seconds"]),
        gc_after_trial=True,
    )

    rows = []
    for trial in study.trials:
        if trial.values is None:
            continue

        rows.append(
            {
                "trial_number": trial.number,
                "trial_id": trial.user_attrs.get("trial_id", f"trial_{trial.number}"),
                "state": str(trial.state),
                "obj_accuracy_min": float(trial.values[0]),
                "obj_inference_ms": float(trial.values[1]),
                "obj_model_params": float(trial.values[2]),
                "accuracy": float(trial.user_attrs.get("accuracy", 1.0 - float(trial.values[0]))),
                "noisy_accuracy": float(trial.user_attrs.get("noisy_accuracy", 0.0)),
                "inference_ms": float(trial.user_attrs.get("inference_ms", trial.values[1])),
                "model_params": int(trial.user_attrs.get("model_params", trial.values[2])),
                **trial.params,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = Path(output_dir) / "all_trials.csv"
    df.to_csv(csv_path, index=False)

    return df
