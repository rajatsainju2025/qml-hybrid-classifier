"""Main experiment runner: VQC vs classical baselines.

Usage:
    python experiments/run_experiment.py --config experiments/configs/baseline_vqc.yaml

Outputs:
    results/tables/main_results.csv
    results/figures/training_curves.png
    results/figures/results_heatmap.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from qml_hybrid.evaluate import generate_results_table
from qml_hybrid.train import run_training
from qml_hybrid.utils import (
    load_dataset,
    plot_results_heatmap,
    plot_training_curves,
)


def train_classical_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    baseline_names: list[str],
) -> dict[str, dict]:
    """Train classical baseline models and return predictions.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels (unused here, returned for caller).
        baseline_names: Subset of ``["svm", "logistic_regression", "mlp"]``.

    Returns:
        Mapping from model name to ``{"preds": ..., "probs": ...}`` dicts.
    """
    results: dict[str, dict] = {}

    if "svm" in baseline_names:
        clf = SVC(kernel="rbf", probability=True, random_state=42)
        clf.fit(X_train, y_train)
        results["SVM (RBF)"] = {
            "preds": clf.predict(X_test),
            "probs": clf.predict_proba(X_test),
        }

    if "logistic_regression" in baseline_names:
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        results["Logistic Regression"] = {
            "preds": clf.predict(X_test),
            "probs": clf.predict_proba(X_test),
        }

    if "mlp" in baseline_names:
        clf = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=500,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        results["MLP (32-16)"] = {
            "preds": clf.predict(X_test),
            "probs": clf.predict_proba(X_test),
        }

    return results


def main() -> None:
    """Run the main experiment: VQC vs. classical baselines."""
    parser = argparse.ArgumentParser(description="Run VQC vs baseline experiment.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment config file.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = Path(config.get("results_dir", "results"))
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data for classical baselines (VQC training does its own split) ---
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(
        name=config["dataset"],
        seed=config["seed"],
        val_size=config.get("val_size", 0.15),
        test_size=config.get("test_size", 0.15),
    )
    # Combine train+val for classical baselines (they don't need early stopping)
    X_cl_train = np.concatenate([X_train, X_val])
    y_cl_train = np.concatenate([y_train, y_val])

    # --- Train VQC ---
    print("\n[1/3] Training Hybrid VQC …")
    vqc_result = run_training(config)
    print(f"      Done in {vqc_result['train_time_s']:.1f}s | "
          f"model saved to {vqc_result['model_path']}")

    # --- Train classical baselines ---
    print("\n[2/3] Training classical baselines …")
    baseline_names = config.get("baseline_models", ["svm", "logistic_regression", "mlp"])
    classical_results = train_classical_baselines(
        X_cl_train, y_cl_train, X_test, y_test, baseline_names
    )

    # --- Collect all results ---
    all_results = {
        "VQC (ours)": {
            "preds": vqc_result["test_preds"],
            "probs": vqc_result["test_probs"],
        },
        **classical_results,
    }

    print("\n[3/3] Evaluating and saving results …")
    df = generate_results_table(y_test, all_results, reference_model="VQC (ours)")
    csv_path = tables_dir / "main_results.csv"
    df.to_csv(csv_path)

    # --- Figures ---
    plot_training_curves(
        vqc_result["train_losses"],
        vqc_result["val_losses"],
        title=f"VQC Training Curves ({config['ansatz']}, {config['n_layers']} layers)",
        save_path=str(figures_dir / "training_curves.png"),
    )
    plot_results_heatmap(
        df.drop(columns=["p_bonferroni", "significant"], errors="ignore"),
        title="Model Comparison — Breast Cancer",
        save_path=str(figures_dir / "results_heatmap.png"),
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(df.to_string())
    print("\nSaved:")
    print(f"  Table  → {csv_path}")
    print(f"  Curves → {figures_dir / 'training_curves.png'}")
    print(f"  Heatmap→ {figures_dir / 'results_heatmap.png'}")


if __name__ == "__main__":
    main()
