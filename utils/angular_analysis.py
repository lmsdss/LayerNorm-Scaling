#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 Angular Distance Layer Analysis

Analyzes layer-wise angular distance for Qwen3 and similar Transformer models:
evaluates each layer's importance to the n-th subsequent layer on the C4 validation set
and plots a heatmap.

Dependencies: torch, transformers, datasets, matplotlib, numpy, tqdm
Requires the project's short_hf module.
"""

import os
import numpy as np
import torch
import datasets
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator, Trainer, TrainingArguments
from short_hf import ShortHFModel
from itertools import islice
import random

# ==================== Configurable paths (edit for your environment) ====================
# HuggingFace mirror (optional; comment out if not needed)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Model path: local directory or HuggingFace model ID
MODEL_NAME_OR_PATH = "Qwen/Qwen3-8B"  # e.g. "Qwen/Qwen3-8B" or "/path/to/local/Qwen3-8B"

# Dataset: local path or HuggingFace dataset name (C4 en validation, streaming)
DATASET_PATH = "allenai/c4"  # or "/path/to/local/c4"
DATASET_CONFIG = "en"
DATASET_SPLIT = "validation"

# Output directory for figures and intermediate results
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== Load model ====================
short_model = ShortHFModel(
    model_name=MODEL_NAME_OR_PATH,
    layers_path="model.layers",
    n_prune_layers=1,
)
_ = short_model.model  # trigger model loading

# ==================== Load data ====================
val_data = datasets.load_dataset(
    DATASET_PATH,
    DATASET_CONFIG,
    split=DATASET_SPLIT,
    streaming=True,
    trust_remote_code=True,
)

# ==================== Collect per-layer importance (Angular) ====================
# Evaluate importance for the n-th subsequent layer (n=1..N-1); each sample evaluated test_times
num_subsequent_layers = 27   # number of subsequent layers to consider; adjust for model depth
test_times = 2
max_samples = 1000
alldata = []

for nidx in range(1, num_subsequent_layers + 1):
    short_model.importances = [0 for _ in range(12)]
    idx = 0
    for i, cur in enumerate(tqdm(val_data, desc=f"n={nidx}")):
        if i >= max_samples:
            break
        prompts = cur["text"]
        _ = short_model.eval_importance(
            prompts=prompts,
            max_seq_len=256,
            stride=256,
            max_gen_len=0,
            angular=True,
            n=nidx,
        )
        idx += 1
        if idx > test_times:
            break
    alldata.append(short_model.importances)

alldata = np.array(alldata) / 3

# ==================== Normalization ====================
def normalize_rows_robustly(data, p_low=1, p_high=99):
    """Per-row percentile clip then min-max normalize to [0, 1]."""
    normalized_data = np.zeros_like(data, dtype=float)
    for i in range(data.shape[0]):
        row = data[i, :]
        val_low = np.nanpercentile(row, p_low)
        val_high = np.nanpercentile(row, p_high)
        if val_low == val_high:
            normalized_data[i, :] = 0.0
            continue
        row_clipped = np.clip(row, val_low, val_high)
        min_clipped = np.min(row_clipped)
        max_clipped = np.max(row_clipped)
        if max_clipped == min_clipped:
            normalized_data[i, :] = 0.0
        else:
            normalized_data[i, :] = (row_clipped - min_clipped) / (max_clipped - min_clipped)
    return normalized_data

# Optional: save/load intermediate results
# np.save(os.path.join(OUTPUT_DIR, "angular_importance.npy"), alldata)
# alldata = np.load(os.path.join(OUTPUT_DIR, "angular_importance.npy"))

alldata_normed = normalize_rows_robustly(alldata, p_low=1, p_high=99)

# ==================== Visualization ====================
def plot_angular_distance_heatmap(data_array, L_total, vmin=0.0, vmax=1.0, save_path=None):
    """
    Plot layer-wise angular distance heatmap.
    data_array: shape (N, M), N = subsequent layer index, M = layer index
    L_total: total number of layers; used to mask invalid region
    """
    N, M = data_array.shape
    l_indices = np.arange(M)
    n_indices = np.arange(N)
    mask = (l_indices[np.newaxis, :] + n_indices[:, np.newaxis] + 1) >= L_total
    masked_data = np.where(mask, np.nan, data_array)

    fig, ax = plt.subplots(figsize=(6, 4))
    x_coords = np.arange(M + 1)
    y_coords = np.arange(N + 1)
    mesh = ax.pcolormesh(
        x_coords, y_coords, masked_data,
        cmap="viridis_r",
        edgecolors="white",
        linewidth=0.2,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("equal")
    ax.set_xlabel(r"Layer Index $\ell$", fontsize=10)
    ax.set_ylabel(r"Subsequent $n^{th}$ Layer", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(bottom=False, left=False)

    x_tick_interval = 5 if M > 5 else max(1, M // 10)
    y_tick_interval = 4 if N > 4 else max(1, N // 7)
    ax.set_xticks(np.arange(0, M, x_tick_interval))
    ax.set_yticks(np.arange(0, N, y_tick_interval))

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.05)
    cbar.set_ticks(np.arange(round(vmin, 1), vmax + 0.01, 0.1))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.show()

# Total number of layers (e.g. Qwen3-8B has 36;  adjust for your model)
L_TOTAL_LAYERS = 36
heatmap_path = os.path.join(OUTPUT_DIR, "qwen3_angular_distance.pdf")
plot_angular_distance_heatmap(alldata_normed, L_TOTAL_LAYERS, save_path=heatmap_path)

# Restore matplotlib defaults
mpl.rcParams.update(mpl.rcParamsDefault)
