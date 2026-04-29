"""Ablation runner: sweeps circuit depth and logs Meyer–Wallach expressibility.

Usage:
    python experiments/run_ablation.py --config experiments/configs/ablation_depth.yaml

Outputs:
    results/tables/ablation_results.csv
    results/figures/ablation_depth.png
    results/figures/expressibility.png
"""

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qml_hybrid.circuits import meyer_wallach_expressibility
from qml_hybrid.evaluate import compute_metrics
from qml_hybrid.train import run_training
from qml_hybrid.utils import load_dataset, plot_expressibility


def main() -> None:
    """Run the depth ablation study."""
    parser = argparse.ArgumentParser(description="Run circuit depth ablation.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML ablation config file.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = Path(config.get("results_dir", "results"))
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    ablation_param: str = config["ablation_param"]
    ablation_values: list = config["ablation_values"]
    ansatz: str = config["ansatz"]
    n_qubits: int = config["n_qubits"]

    rows = []
    expressibility_data = {}

    # Load test labels once (same split for all ablation runs)
    _, _, _, _, _, y_test = load_dataset(
        name=config["dataset"],
        seed=config["seed"],
        val_size=config.get("val_size", 0.15),
        test_size=config.get("test_size", 0.15),
    )

    for val in ablation_values:
        run_config = copy.deepcopy(config)
        run_config[ablation_param] = val
        run_config["run_name"] = f"ablation-{ablation_param}-{val}"

        n_layers = run_config["n_layers"]
        print(f"\n--- {ablation_param}={val} (n_layers={n_layers}) ---")

        result = run_training(run_config)
        metrics = compute_metrics(
            y_test,
            result["test_preds"],
            result["test_probs"],
        )

        # Meyer–Wallach measure for this depth
        mw = meyer_wallach_expressibility(
            n_qubits=n_qubits,
            n_layers=n_layers,
            ansatz=ansatz,
            n_samples=100,
            seed=config["seed"],
        )
        expressibility_data[(ansatz, n_layers)] = mw

        from qml_hybrid.circuits import weight_shape as _ws

        n_params = 1
        for dim in _ws(n_qubits, n_layers, ansatz):
            n_params *= dim

        rows.append(
            {
                ablation_param: val,
                "n_params": n_params,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "roc_auc": metrics.get("roc_auc", float("nan")),
                "meyer_wallach": mw,
                "train_time_s": result["train_time_s"],
            }
        )
        print(
            f"    acc={metrics['accuracy']:.3f} | "
            f"roc_auc={metrics.get('roc_auc', float('nan')):.3f} | "
            f"MW={mw:.3f} | "
            f"time={result['train_time_s']:.1f}s"
        )

    df = pd.DataFrame(rows).set_index(ablation_param)
    csv_path = tables_dir / "ablation_results.csv"
    df.to_csv(csv_path)

    plot_expressibility(
        expressibility_data,
        save_path=str(figures_dir / "expressibility.png"),
    )

    print("\n" + "=" * 60)
    print("ABLATION RESULTS")
    print("=" * 60)
    print(df.to_string())
    print(f"\nSaved table → {csv_path}")


if __name__ == "__main__":
    main()
