"""Unit tests for evaluation utilities in qml_hybrid.evaluate."""

import numpy as np
import pytest

from qml_hybrid.evaluate import (
    compute_metrics,
    generate_results_table,
    mcnemar_test,
    run_significance_test,
)


class TestComputeMetrics:
    """Tests for compute_metrics."""

    def test_perfect_classifier(self):
        y = np.array([0, 1, 0, 1, 1])
        m = compute_metrics(y_true=y, y_pred=y)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1_macro"] == pytest.approx(1.0)
        assert m["f1_weighted"] == pytest.approx(1.0)

    def test_with_probabilities(self):
        y = np.array([0, 1, 0, 1])
        preds = np.array([0, 1, 0, 1])
        probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        m = compute_metrics(y, preds, probs)
        assert "roc_auc" in m
        assert "pr_auc" in m
        assert m["roc_auc"] == pytest.approx(1.0)

    def test_without_probabilities_no_roc(self):
        y = np.array([0, 1, 0, 1])
        preds = np.array([0, 1, 1, 0])
        m = compute_metrics(y, preds, None)
        assert "roc_auc" not in m
        assert "accuracy" in m

    def test_metric_ranges(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=50)
        preds = rng.integers(0, 2, size=50)
        m = compute_metrics(y, preds)
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["f1_macro"] <= 1.0


class TestMcnemarTest:
    """Tests for McNemar's test."""

    def test_identical_classifiers_not_significant(self):
        """Two identical classifiers should not be significantly different."""
        y = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        preds = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        chi2, p = mcnemar_test(y, preds, preds)
        # Identical predictions → off-diagonal = 0 → p = 1.0
        assert p == pytest.approx(1.0, abs=0.01)

    def test_returns_two_floats(self):
        y = np.array([0, 1, 0, 1])
        a = np.array([0, 1, 0, 1])
        b = np.array([0, 0, 0, 1])
        chi2, p = mcnemar_test(y, a, b)
        assert isinstance(chi2, float)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0


class TestGenerateResultsTable:
    """Integration test for generate_results_table."""

    def test_output_is_dataframe_with_correct_index(self):
        import pandas as pd

        y = np.array([0, 1, 0, 1, 0, 1] * 5)
        model_results = {
            "ModelA": {"preds": y, "probs": np.column_stack([1 - y * 0.9, y * 0.9])},
            "ModelB": {"preds": np.zeros_like(y), "probs": None},
        }
        df = generate_results_table(y, model_results, reference_model="ModelA")
        assert isinstance(df, pd.DataFrame)
        assert "ModelA" in df.index or "ModelB" in df.index
        assert "accuracy" in df.columns

    def test_significance_columns_present_when_reference_given(self):
        y = np.array([0, 1] * 10)
        preds_a = np.array([0, 1] * 10)
        preds_b = np.array([1, 0] * 10)
        model_results = {
            "A": {"preds": preds_a},
            "B": {"preds": preds_b},
        }
        df = generate_results_table(y, model_results, reference_model="A")
        assert "p_bonferroni" in df.columns
        assert "significant" in df.columns
