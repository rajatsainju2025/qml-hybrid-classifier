"""Evaluation utilities: metrics, baseline comparison, statistical testing.

All functions are pure (no side effects) and accept numpy arrays so they can
be used independently of any framework.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute a comprehensive suite of classification metrics.

    Args:
        y_true: Ground-truth class labels, shape ``(n_samples,)``.
        y_pred: Predicted class labels, shape ``(n_samples,)``.
        y_prob: Predicted class probabilities, shape ``(n_samples, n_classes)``.
            Required for ROC-AUC and PR-AUC; pass ``None`` to skip those metrics.

    Returns:
        Dictionary with keys: ``accuracy``, ``f1_macro``, ``f1_weighted``,
        and (if ``y_prob`` provided) ``roc_auc``, ``pr_auc``.
    """
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }
    if y_prob is not None:
        n_classes = y_prob.shape[1] if y_prob.ndim == 2 else 2
        if n_classes == 2:
            scores = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
            metrics["pr_auc"] = float(average_precision_score(y_true, scores))
        else:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
    return metrics


def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> tuple[float, float]:
    """Perform McNemar's test to compare two paired classifiers.

    Tests the null hypothesis that classifiers A and B have the same error rate
    on the same test set.  Appropriate for paired, non-parametric comparison
    of binary classifiers (McNemar, 1947).

    The contingency table is::

              B correct  B wrong
        A correct   n00       n01
        A wrong     n10       n11

    Args:
        y_true: Ground-truth labels.
        preds_a: Predictions from classifier A.
        preds_b: Predictions from classifier B.

    Returns:
        Tuple of ``(chi2_statistic, p_value)``.  Reject H₀ if p < 0.05.
    """
    correct_a = preds_a == y_true
    correct_b = preds_b == y_true
    n00 = int(np.sum(correct_a & correct_b))
    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))
    n11 = int(np.sum(~correct_a & ~correct_b))
    table = np.array([[n00, n01], [n10, n11]])
    chi2_stat, p_val, _, _ = chi2_contingency(table, correction=True)  # type: ignore[misc]
    return float(chi2_stat), float(p_val)  # type: ignore[arg-type]


def compare_to_baseline(
    y_true: np.ndarray,
    model_results: dict[str, dict],
) -> pd.DataFrame:
    """Aggregate metrics across multiple models into a comparison DataFrame.

    Args:
        y_true: Ground-truth labels.
        model_results: Mapping from model name to a dict containing
            ``"preds"`` (np.ndarray) and optionally ``"probs"`` (np.ndarray).

    Returns:
        DataFrame with one row per model and columns for each metric.
    """
    rows: list[dict] = []
    for name, res in model_results.items():
        m: dict = compute_metrics(y_true, res["preds"], res.get("probs"))
        m["model"] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values("roc_auc", ascending=False) if "roc_auc" in df.columns else df


def run_significance_test(
    y_true: np.ndarray,
    model_results: dict[str, dict],
    reference_model: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run pairwise McNemar tests vs. a reference model with Bonferroni correction.

    Args:
        y_true: Ground-truth labels.
        model_results: Same format as ``compare_to_baseline``.
        reference_model: Key in ``model_results`` to treat as the reference.
        alpha: Family-wise significance level (Bonferroni corrected).

    Returns:
        DataFrame with columns ``chi2``, ``p_value``, ``p_bonferroni``,
        ``significant`` for each non-reference model.
    """
    ref_preds = model_results[reference_model]["preds"]
    others = {k: v for k, v in model_results.items() if k != reference_model}
    n_comparisons = len(others)
    rows = []
    for name, res in others.items():
        chi2, p_val = mcnemar_test(y_true, ref_preds, res["preds"])
        p_bonf = min(p_val * n_comparisons, 1.0)
        rows.append(
            {
                "model": name,
                "chi2": chi2,
                "p_value": p_val,
                "p_bonferroni": p_bonf,
                "significant": p_bonf < alpha,
            }
        )
    return pd.DataFrame(rows).set_index("model")


def generate_results_table(
    y_true: np.ndarray,
    model_results: dict[str, dict],
    reference_model: str | None = None,
) -> pd.DataFrame:
    """Generate a publication-ready results table.

    Combines metric comparison and (optionally) statistical significance into
    a single DataFrame ready for saving as CSV or rendering in a notebook.

    Args:
        y_true: Ground-truth labels.
        model_results: Mapping from model name to ``{"preds": ..., "probs": ...}``.
        reference_model: If provided, append McNemar p-values vs. this model.

    Returns:
        Formatted DataFrame with all metrics and optional significance columns.
    """
    df = compare_to_baseline(y_true, model_results)
    if reference_model is not None and reference_model in model_results:
        sig_df = run_significance_test(y_true, model_results, reference_model)
        df = df.join(sig_df[["p_bonferroni", "significant"]], how="left")
    return df
