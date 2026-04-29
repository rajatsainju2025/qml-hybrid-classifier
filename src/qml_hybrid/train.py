"""Training loop with MLflow tracking, early stopping, and checkpointing.

All randomness is seeded through ``run_training(config)`` for full
reproducibility.  The only public entry point is ``run_training``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from .model import HybridQClassifier
from .utils import load_dataset

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Monitors validation loss and signals when training should stop.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change in monitored value to qualify as improvement.
        checkpoint_path: Path to save the best model weights.
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        checkpoint_path: str = "best_model.pt",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module) -> None:
        """Update state given the current validation loss.

        Saves a checkpoint if the loss improved; increments the patience
        counter otherwise.

        Args:
            val_loss: Validation loss for the current epoch.
            model: Model whose state dict to checkpoint.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def _make_dataloaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """Load, split, and wrap the dataset in DataLoaders.

    Args:
        config: Experiment config dict; must contain ``dataset``, ``seed``,
            and ``batch_size`` keys.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, y_test_array).
    """
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(
        name=config["dataset"],
        seed=config["seed"],
        val_size=config.get("val_size", 0.15),
        test_size=config.get("test_size", 0.15),
    )

    def _to_loader(X, y, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=config["batch_size"], shuffle=shuffle)

    return (
        _to_loader(X_train, y_train, shuffle=True),
        _to_loader(X_val, y_val, shuffle=False),
        _to_loader(X_test, y_test, shuffle=False),
        y_test,
    )


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """Train a HybridQClassifier and return results.

    Sets global seeds for full reproducibility.  Logs all hyperparameters,
    per-epoch metrics, and final test metrics to MLflow.

    Args:
        config: Flat dictionary with the following required keys:

            - ``n_qubits`` (int)
            - ``n_layers`` (int)
            - ``ansatz`` (str)
            - ``embedding`` (str)
            - ``lr`` (float)
            - ``batch_size`` (int)
            - ``max_epochs`` (int)
            - ``patience`` (int)
            - ``seed`` (int)
            - ``dataset`` (str)

    Returns:
        Dictionary with keys ``"train_losses"``, ``"val_losses"``,
        ``"test_preds"``, ``"test_probs"``, ``"model_path"``, ``"train_time_s"``.
    """
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, test_loader, _ = _make_dataloaders(config)

    # Infer input dimension from first batch
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[1]

    model = HybridQClassifier(
        input_dim=input_dim,
        n_qubits=config["n_qubits"],
        n_layers=config["n_layers"],
        n_classes=config.get("n_classes", 2),
        ansatz=config["ansatz"],
        embedding=config["embedding"],
    )
    logger.info("Model: %s", model)

    optimizer = Adam(model.parameters(), lr=config["lr"])
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.NLLLoss()

    results_dir = Path(config.get("results_dir", "results"))
    checkpoint_path = str(results_dir / "best_model.pt")
    os.makedirs(results_dir, exist_ok=True)

    early_stopping = EarlyStopping(
        patience=config["patience"],
        checkpoint_path=checkpoint_path,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    t0 = time.time()

    mlflow.set_experiment(config.get("experiment_name", "qml-hybrid-classifier"))
    with mlflow.start_run(run_name=config.get("run_name", None)):
        mlflow.log_params(
            {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))}
        )
        mlflow.log_param("n_quantum_params", model.n_quantum_params)
        mlflow.log_param("n_classical_params", model.n_classical_params)

        for epoch in range(1, config["max_epochs"] + 1):
            # --- Train ---
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                log_probs = model(X_batch)
                loss = criterion(log_probs, y_batch)
                # Gradient clipping guards against exploding gradients in PQC
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(X_batch)
            train_loss /= len(train_loader.dataset)  # type: ignore[arg-type]

            # --- Validate ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    log_probs = model(X_batch)
                    val_loss += criterion(log_probs, y_batch).item() * len(X_batch)
            val_loss /= len(val_loader.dataset)  # type: ignore[arg-type]

            scheduler.step(val_loss)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
            )

            if epoch % 10 == 0:
                logger.info(
                    "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f",
                    epoch,
                    config["max_epochs"],
                    train_loss,
                    val_loss,
                )

            early_stopping.step(val_loss, model)
            if early_stopping.should_stop:
                logger.info("Early stopping at epoch %d.", epoch)
                break

        # --- Test ---
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        model.eval()
        all_preds, all_probs = [], []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                log_probs = model(X_batch)
                probs = torch.exp(log_probs)
                all_probs.append(probs.numpy())
                all_preds.append(log_probs.argmax(dim=1).numpy())

        test_preds = np.concatenate(all_preds)
        test_probs = np.concatenate(all_probs)
        train_time = time.time() - t0

        mlflow.log_metric("train_time_s", train_time)
        mlflow.log_artifact(checkpoint_path)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_preds": test_preds,
        "test_probs": test_probs,
        "model_path": checkpoint_path,
        "train_time_s": train_time,
    }
