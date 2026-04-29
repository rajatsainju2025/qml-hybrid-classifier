"""Data loading, preprocessing, and visualisation utilities.

Provides a single ``load_dataset`` function that returns stratified
train/val/test splits as numpy arrays, and helper functions for
generating publication-ready plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(
    name: str = "breast_cancer",
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[np.ndarray, ...]:
    """Load and split a classification dataset into train/val/test arrays.

    Features are standardised (zero mean, unit variance) using statistics
    computed *only* from the training split to prevent data leakage.

    Supported dataset names:
        - ``"breast_cancer"``: Wisconsin Breast Cancer (569 samples, 30 features,
          binary classification).

    Args:
        name: Dataset identifier string.
        seed: Random seed for reproducible splits.
        val_size: Fraction of full data for validation.
        test_size: Fraction of full data for test.

    Returns:
        Tuple ``(X_train, X_val, X_test, y_train, y_val, y_test)``
        all as ``np.ndarray``.

    Raises:
        ValueError: For unsupported dataset names.
    """
    if name == "breast_cancer":
        X_raw, y_raw = load_breast_cancer(return_X_y=True)
        X = np.asarray(X_raw, dtype=np.float32)  # type: ignore[arg-type]
        y = np.asarray(y_raw, dtype=np.int64)  # type: ignore[arg-type]
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Currently supported: 'breast_cancer'."
        )

    # First split: hold out test set
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # Second split: carve out validation from remaining data
    val_fraction_of_tmp = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_fraction_of_tmp, random_state=seed, stratify=y_tmp
    )

    # Standardise using training statistics only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    title: str = "Training Curves",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training and validation loss curves.

    Args:
        train_losses: Per-epoch training loss.
        val_losses: Per-epoch validation loss.
        title: Plot title.
        save_path: If provided, saves the figure to this path (PNG, 300 dpi).

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val loss", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL Loss")
    ax.set_title(title)
    ax.legend()
    sns.despine(ax=ax)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_results_heatmap(
    results_df: pd.DataFrame,
    title: str = "Model Comparison",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Visualise a model-comparison DataFrame as a colour-coded heatmap.

    Args:
        results_df: DataFrame with models as index and metrics as columns.
        title: Plot title.
        save_path: Optional path for saving the figure.

    Returns:
        Matplotlib Figure object.
    """
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    fig, ax = plt.subplots(figsize=(len(numeric_cols) * 1.4 + 1, len(results_df) * 0.8 + 1))
    sns.heatmap(
        results_df[numeric_cols].astype(float),
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_expressibility(
    expressibility_data: dict[str, Any],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot Meyer–Wallach expressibility vs. circuit depth for multiple ansatze.

    Args:
        expressibility_data: Dict mapping ``(ansatz_name, n_layers)`` tuples
            to Meyer–Wallach Q values; e.g.,
            ``{("strongly_entangling", 1): 0.42, ...}``.
        save_path: Optional path for saving the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    # Group by ansatz
    grouped: dict[str, dict[int, float]] = {}
    for (ansatz, depth), q_val in expressibility_data.items():
        grouped.setdefault(ansatz, {})[depth] = q_val

    for ansatz_name, depth_map in grouped.items():
        depths = sorted(depth_map.keys())
        q_vals = [depth_map[d] for d in depths]
        ax.plot(depths, q_vals, marker="o", label=ansatz_name, linewidth=1.5)

    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Meyer–Wallach Q")
    ax.set_title("Circuit Expressibility vs. Depth")
    ax.legend()
    ax.set_ylim(0, 1)
    sns.despine(ax=ax)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
