import argparse
import csv
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import trange

from model import Model
from utils import load_mat, normalize_adj, preprocess_features


DEFAULT_DATA_DIR = "~/datasets/GAD/mat"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_default_args(args: argparse.Namespace) -> argparse.Namespace:
    dataset_key = args.dataset.lower()

    if args.lr is None:
        args.lr = 1e-3

    if args.num_epoch is None:
        default_epochs = {
            "photo": 100,
            "elliptic": 150,
            "reddit": 300,
            "t_finance": 500,
            "t-finance": 500,
            "amazon": 800,
        }
        args.num_epoch = default_epochs.get(dataset_key, 300)

    if args.mean is None or args.var is None:
        if dataset_key in {"reddit", "photo"}:
            args.mean = 0.02 if args.mean is None else args.mean
            args.var = 0.01 if args.var is None else args.var
        else:
            args.mean = 0.0 if args.mean is None else args.mean
            args.var = 0.0 if args.var is None else args.var

    return args


def prepare_data(args: argparse.Namespace):
    (
        adj,
        features,
        labels,
        all_idx,
        idx_train,
        idx_val,
        idx_test,
        ano_label,
        str_ano_label,
        attr_ano_label,
        normal_label_idx,
        abnormal_label_idx,
    ) = load_mat(
        args.dataset,
        data_dir=args.data_dir,
        train_rate=args.train_rate,
        val_rate=args.val_rate,
        normal_rate=args.normal_rate,
        outlier_rate=args.outlier_rate,
        verbose=False,
    )

    dataset_key = args.dataset.lower()
    if dataset_key in {"amazon", "tf_finace", "t_finance", "reddit", "elliptic"}:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()

    raw_adj = adj
    adj = normalize_adj(adj)

    raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    raw_adj = torch.FloatTensor(raw_adj[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])

    return features, adj, raw_adj, labels, idx_test, ano_label, normal_label_idx, abnormal_label_idx


def train_one_trial(args: argparse.Namespace, seed: int) -> dict:
    set_seed(seed)
    trial_start = time.time()

    features, adj, raw_adj, labels, idx_test, ano_label, normal_label_idx, abnormal_label_idx = prepare_data(args)

    ft_size = features.shape[2]
    model = Model(ft_size, args.embedding_dim, "prelu", args.negsamp_ratio, args.readout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    b_xent = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor([args.negsamp_ratio], dtype=torch.float32)
    )

    final_auc = np.nan
    final_auprc = np.nan

    for epoch in trange(args.num_epoch, desc="Epoch", position=1, leave=False):
        model.train()
        optimiser.zero_grad()

        train_flag = True
        emb, emb_combine, logits, emb_con, emb_abnormal = model(
            features, adj, abnormal_label_idx, normal_label_idx, train_flag, args
        )

        lbl = torch.unsqueeze(
            torch.cat((torch.zeros(len(normal_label_idx)), torch.ones(len(emb_con)))), 1
        ).unsqueeze(0)

        loss_bce = torch.mean(b_xent(logits, lbl))

        emb = torch.squeeze(emb)
        emb_inf = torch.pow(torch.norm(emb, dim=-1, keepdim=True), -1)
        emb_inf[torch.isinf(emb_inf)] = 0.0
        emb_norm = emb * emb_inf
        sim_matrix = torch.mm(emb_norm, emb_norm.T)

        raw_adj_squeezed = torch.squeeze(raw_adj)
        similar_matrix = sim_matrix * raw_adj_squeezed
        r_inv = torch.pow(torch.sum(raw_adj_squeezed, 0), -1)
        r_inv[torch.isinf(r_inv)] = 0.0
        affinity = torch.sum(similar_matrix, 0) * r_inv

        affinity_normal_mean = torch.mean(affinity[normal_label_idx])
        affinity_abnormal_mean = torch.mean(affinity[abnormal_label_idx])
        loss_margin = (args.confidence_margin - (affinity_normal_mean - affinity_abnormal_mean)).clamp_min(min=0)

        diff_attribute = torch.pow(emb_con - emb_abnormal, 2)
        loss_rec = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1)))

        loss = loss_margin + loss_bce + loss_rec
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        train_flag = False
        _emb, _emb_combine, logits, _emb_con, _emb_abnormal = model(
            features, adj, abnormal_label_idx, normal_label_idx, train_flag, args
        )
        scores = np.squeeze(logits[:, idx_test, :].cpu().numpy())
        y_true = ano_label[idx_test]
        final_auc = roc_auc_score(y_true, scores) * 100.0
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        final_auprc = auc(recall, precision) * 100.0

    return {
        "auc": final_auc,
        "auprc": final_auprc,
        "train_time_min": (time.time() - trial_start) / 60.0,
    }


def metric_summary(values: list[float]) -> str:
    arr = np.array(values, dtype=float)
    return f"{arr.mean():.2f}±{arr.std(ddof=0):.2f}({arr.max():.2f})"


def append_results_csv(csv_path: str, summary_row: dict) -> None:
    path = Path(csv_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "time",
        "dataset",
        "trial",
        "auc",
        "auprc",
        "train_time_min",
    ]

    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow({k: summary_row.get(k, "") for k in fieldnames})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GGAD multi-trial runner")
    parser.add_argument("--dataset", type=str, default="Reddit")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result_csv", type=str, default="results/ggad_results.csv")

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--drop_prob", type=float, default=0.0)
    parser.add_argument("--readout", type=str, default="avg", choices=["avg", "max", "min", "weighted_sum"])
    parser.add_argument("--auc_test_rounds", type=int, default=256)
    parser.add_argument("--negsamp_ratio", type=int, default=1)
    parser.add_argument("--mean", type=float, default=None)
    parser.add_argument("--var", type=float, default=None)
    parser.add_argument("--confidence_margin", type=float, default=0.7)

    parser.add_argument("--train_rate", type=float, default=0.3)
    parser.add_argument("--val_rate", type=float, default=0.1)
    parser.add_argument("--normal_rate", type=float, default=0.5)
    parser.add_argument("--outlier_rate", type=float, default=None)

    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = apply_default_args(build_arg_parser().parse_args())

    trial_rows = []
    for trial in trange(args.num_trials, desc="Trial", position=0, leave=False):
        trial_rows.append(train_one_trial(args, seed=args.seed+trial))

    auc_values = [r["auc"] for r in trial_rows]
    auprc_values = [r["auprc"] for r in trial_rows]
    total_time_min = sum(r["train_time_min"] for r in trial_rows)

    summary_row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": args.dataset,
        "epochs": args.num_epoch,
        "trial": args.num_trials,
        "auc": metric_summary(auc_values),
        "auprc": metric_summary(auprc_values),
        "train_time_min": f"{total_time_min:.2f}",
    }

    append_results_csv(args.result_csv, summary_row)

    print("Training finished.")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.num_epoch}")
    print(f"Trials: {args.num_trials}")
    print(f"AUC: {summary_row['auc']}")
    print(f"AUPRC: {summary_row['auprc']}")
    print(f"Results saved to: {Path(args.result_csv).expanduser()}")


if __name__ == "__main__":
    main()
