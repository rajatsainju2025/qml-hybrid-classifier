"""Unit tests for HybridQClassifier model in qml_hybrid.model."""

import pytest
import torch
import numpy as np

from qml_hybrid.model import HybridQClassifier


class TestHybridQClassifier:
    """Tests for model instantiation and forward pass."""

    @pytest.fixture
    def small_model(self):
        """A minimal model for fast CPU tests."""
        return HybridQClassifier(input_dim=8, n_qubits=4, n_layers=1, n_classes=2)

    def test_output_shape(self, small_model):
        """Forward pass must return (batch_size, n_classes)."""
        x = torch.randn(5, 8)
        out = small_model(x)
        assert out.shape == (5, 2)

    def test_output_is_log_prob(self, small_model):
        """Output should be log-probabilities: exp sums to ~1 per sample."""
        x = torch.randn(4, 8)
        log_probs = small_model(x)
        probs = torch.exp(log_probs)
        sums = probs.sum(dim=-1)
        np.testing.assert_allclose(sums.detach().numpy(), np.ones(4), atol=1e-5)

    def test_parameter_counts(self, small_model):
        """Quantum and classical parameter counts must be positive integers."""
        assert small_model.n_quantum_params > 0
        assert small_model.n_classical_params > 0

    def test_strongly_entangling_param_count(self):
        """StronglyEntangling: n_layers * n_qubits * 3 quantum params."""
        model = HybridQClassifier(input_dim=10, n_qubits=4, n_layers=2, ansatz="strongly_entangling")
        assert model.n_quantum_params == 2 * 4 * 3  # 24

    def test_basic_entangling_param_count(self):
        """BasicEntangling: n_layers * n_qubits quantum params."""
        model = HybridQClassifier(input_dim=10, n_qubits=4, n_layers=2, ansatz="basic_entangling")
        assert model.n_quantum_params == 2 * 4  # 8

    def test_repr_contains_key_info(self, small_model):
        r = repr(small_model)
        assert "n_qubits=4" in r
        assert "n_layers=1" in r
        assert "quantum_params=" in r

    def test_gradients_flow(self, small_model):
        """Loss.backward() must populate gradients in all parameters."""
        x = torch.randn(3, 8)
        y = torch.tensor([0, 1, 0])
        criterion = torch.nn.NLLLoss()
        out = small_model(x)
        loss = criterion(out, y)
        loss.backward()
        for name, param in small_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    @pytest.mark.parametrize("ansatz", ["strongly_entangling", "basic_entangling"])
    @pytest.mark.parametrize("embedding", ["angle", "amplitude"])
    def test_all_configs(self, ansatz, embedding):
        """All (ansatz, embedding) combinations must produce valid output."""
        n_qubits = 4
        input_dim = n_qubits if embedding == "angle" else 2**n_qubits
        model = HybridQClassifier(
            input_dim=input_dim,
            n_qubits=n_qubits,
            n_layers=1,
            ansatz=ansatz,
            embedding=embedding,
        )
        x = torch.randn(2, input_dim)
        out = model(x)
        assert out.shape == (2, 2)
        assert not torch.any(torch.isnan(out))
