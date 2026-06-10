"""Optuna hyper-parameter tuning script for 2024-GGAD.

Place this file in the repository root, next to run.py/model.py/utils.py.
Example:
    uv pip install optuna
    python tune_optuna.py --dataset Reddit --data_dir ~/datasets/GAD/mat \
        --n_trials 50 --num_trials 3 --metric auc

Notes:
- --n_trials is the number of Optuna search trials.
- --num_trials is inherited from run.py and means repeated training seeds per Optuna trial.
"""

from __future__ import annotations

import os
import argparse
import copy
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from run import apply_default_args, build_arg_parser, train_one_trial


def add_optuna_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add Optuna-specific arguments without changing run.py."""
    group = parser.add_argument_group("Optuna tuning")
    group.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials.")
    group.add_argument("--timeout", type=int, default=None, help="Optuna timeout in seconds.")
    group.add_argument("--study_name", type=str, default=None, help="Optuna study name.")
    group.add_argument("--storage", type=str, default=None, help="Optuna DB URL, e.g. sqlite:///results/ggad_optuna.db")
    group.add_argument("--metric", type=str, default="auc", choices=["auc", "auprc"], help="Metric to maximize.")
    group.add_argument("--sampler_seed", type=int, default=42, help="Seed for Optuna sampler.")
    group.add_argument("--n_jobs", type=int, default=1, help="Parallel Optuna jobs. Use >1 only if memory allows.")
    group.add_argument("--save_dir", type=str, default="results")
    return parser


def parse_args() -> argparse.Namespace:
    parser = add_optuna_args(build_arg_parser())
    args = parser.parse_args()
    return apply_default_args(args)


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Search space matched to arguments currently used by run.py/model.py."""
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256, 300, 512])
    readout = trial.suggest_categorical("readout", ["avg", "max", "min"])

    return {
        "lr": trial.suggest_float("lr", 1e-5, 5e-2, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3]),
        "embedding_dim": embedding_dim,
        "readout": readout,
        "negsamp_ratio": trial.suggest_int("negsamp_ratio", 1, 5),
        "confidence_margin": trial.suggest_float("confidence_margin", 0.1, 2.0),
        "normal_rate": trial.suggest_float("normal_rate", 0.1, 1.0),
        "outlier_rate": trial.suggest_float("outlier_rate", 0.01, 0.30),
        "mean": trial.suggest_float("mean", 0.0, 0.10),
        "var": trial.suggest_float("var", 0.0, 0.10),
        "num_epoch": trial.suggest_categorical("num_epoch", [50, 100, 200, 400])
    }


def write_trial_csv(path: str, row: dict[str, Any]) -> None:
    csv_path = Path(path).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time",
        "dataset",
        "optuna_trial",
        "metric",
        "value",
        "auc_mean",
        "auc_std",
        "auprc_mean",
        "auprc_std",
        "train_repeats",
        "params_json",
    ]
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def objective_factory(base_args: argparse.Namespace):
    def objective(trial: optuna.Trial) -> float:
        args = copy.deepcopy(base_args)
        params = suggest_params(trial)
        for key, value in params.items():
            setattr(args, key, value)

        auc_values: list[float] = []
        auprc_values: list[float] = []
        repeats = max(1, int(args.num_trials))

        for repeat in range(repeats):
            result = train_one_trial(args, seed=args.seed + repeat)
            auc_values.append(float(result["auc"]))
            auprc_values.append(float(result["auprc"]))

            # Report intermediate value so MedianPruner can stop weak configurations.
            current_values = auc_values if args.metric == "auc" else auprc_values
            trial.report(float(np.mean(current_values)), step=repeat)
            if trial.should_prune():
                raise optuna.TrialPruned()

        auc_mean = float(np.mean(auc_values))
        auprc_mean = float(np.mean(auprc_values))
        value = auc_mean if args.metric == "auc" else auprc_mean

        trial.set_user_attr("auc_mean", auc_mean)
        trial.set_user_attr("auc_std", float(np.std(auc_values)))
        trial.set_user_attr("auprc_mean", auprc_mean)
        trial.set_user_attr("auprc_std", float(np.std(auprc_values)))

        write_trial_csv(
            args.tune_csv,
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": args.dataset,
                "optuna_trial": trial.number,
                "metric": args.metric,
                "value": value,
                "auc_mean": auc_mean,
                "auc_std": float(np.std(auc_values)),
                "auprc_mean": auprc_mean,
                "auprc_std": float(np.std(auprc_values)),
                "train_repeats": repeats,
                "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
            },
        )
        return value

    return objective


def main() -> None:
    args = parse_args()
    args.study_name = args.study_name or f"ggad_{args.dataset}_{args.metric}"
    args.tune_csv = os.path.join(args.save_dir, args.study_name+".csv")
    args.best_cmd = os.path.join(args.save_dir, args.study_name+".sh")
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective_factory(args), n_trials=args.n_trials, timeout=args.timeout, n_jobs=args.n_jobs, show_progress_bar=True)
    params = study.best_trial.params
    best_cmd = f"python run.py --dataset {args.dataset} --num_trials 10"
    for k, v in params.items():
        best_cmd += f" \\\n\t--{k} {v}"
    with open(args.best_cmd, "w") as f:
        f.write(best_cmd)

    print("Optuna tuning finished.")
    print(f"Study: {study.study_name}")
    print(f"Best {args.metric}: {study.best_value:.4f}")
    print(f"Trial log saved to: {Path(args.tune_csv).expanduser()}")
    print(f"Trial cmd saved to: {Path(args.best_cmd).expanduser()}")


if __name__ == "__main__":
    main()
