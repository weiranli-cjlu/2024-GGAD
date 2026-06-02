from collections import Counter
from pathlib import Path
import random

import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            shape = mx.shape
        return coords, mx.data, shape

    if isinstance(sparse_mx, list):
        return [to_tuple(mx) for mx in sparse_mx]
    return to_tuple(sparse_mx)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def _resolve_mat_path(dataset: str, data_dir: str) -> Path:
    base = Path(data_dir).expanduser()
    candidates = [
        base / f"{dataset}.mat",
        base / f"{dataset.lower()}.mat",
        base / f"{dataset.upper()}.mat",
        base / f"{dataset.capitalize()}.mat",
    ]
    for path in candidates:
        if path.exists():
            return path
    available = sorted(p.name for p in base.glob("*.mat")) if base.exists() else []
    raise FileNotFoundError(
        f"Cannot find dataset '{dataset}' under {base}. Tried: "
        + ", ".join(str(p) for p in candidates)
        + (f". Available .mat files: {available}" if available else "")
    )


def load_mat(
    dataset,
    data_dir="~/datasets/GAD/mat",
    train_rate=0.3,
    val_rate=0.1,
    normal_rate=0.5,
    outlier_rate=None,
    verbose=False,
):
    """Load .mat dataset and sample labeled normal/outlier nodes."""
    data = sio.loadmat(_resolve_mat_path(dataset, data_dir))

    label = data["Label"] if "Label" in data else data["gnd"]
    attr = data["Attributes"] if "Attributes" in data else data["X"]
    network = data["Network"] if "Network" in data else data["A"]

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)
    ano_labels = np.squeeze(np.array(label))

    if "str_anomaly_label" in data:
        str_ano_labels = np.squeeze(np.array(data["str_anomaly_label"]))
        attr_ano_labels = np.squeeze(np.array(data["attr_anomaly_label"]))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)

    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[:num_train]
    idx_val = all_idx[num_train : num_train + num_val]
    idx_test = all_idx[num_train + num_val :]

    all_normal_label_idx = [i for i in idx_train if ano_labels[i] == 0]
    normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * normal_rate)]
    random.shuffle(normal_label_idx)

    if outlier_rate is None:
        outlier_rate = 0.05 if dataset.lower() == "amazon" else 0.15
    abnormal_label_idx = normal_label_idx[: int(len(normal_label_idx) * outlier_rate)]

    if verbose:
        print("Training", Counter(np.squeeze(ano_labels[idx_train])))
        print("Test", Counter(np.squeeze(ano_labels[idx_test])))
        print("Normal label rate", normal_rate)
        print("Generated outlier rate", outlier_rate)

    return (
        adj,
        feat,
        ano_labels,
        all_idx,
        idx_train,
        idx_val,
        idx_test,
        ano_labels,
        str_ano_labels,
        attr_ano_labels,
        normal_label_idx,
        abnormal_label_idx,
    )