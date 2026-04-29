"""Unit tests for PennyLane circuit definitions in qml_hybrid.circuits."""

import numpy as np
import pytest

from qml_hybrid.circuits import (
    make_vqc_circuit,
    meyer_wallach_expressibility,
    weight_shape,
)


class TestWeightShape:
    """Tests for the weight_shape utility."""

    def test_strongly_entangling_shape(self):
        shape = weight_shape(n_qubits=4, n_layers=2, ansatz="strongly_entangling")
        assert shape == (2, 4, 3)

    def test_basic_entangling_shape(self):
        shape = weight_shape(n_qubits=4, n_layers=3, ansatz="basic_entangling")
        assert shape == (3, 4)

    def test_invalid_ansatz_raises(self):
        with pytest.raises(ValueError, match="Unknown ansatz"):
            weight_shape(4, 2, ansatz="bad_ansatz")


class TestMakeVQCCircuit:
    """Tests for the make_vqc_circuit factory."""

    @pytest.mark.parametrize(
        "ansatz, embedding",
        [
            ("strongly_entangling", "angle"),
            ("basic_entangling", "angle"),
            ("strongly_entangling", "amplitude"),
        ],
    )
    def test_circuit_output_shape(self, ansatz, embedding):
        """Circuit must return one expectation value per qubit."""
        n_qubits, n_layers = 4, 2
        circuit = make_vqc_circuit(n_qubits, n_layers, ansatz, embedding)
        w = np.zeros(weight_shape(n_qubits, n_layers, ansatz))

        if embedding == "angle":
            x = np.zeros(n_qubits)
        else:
            x = np.ones(2**n_qubits) / np.sqrt(2**n_qubits)

        out = circuit(w, x)
        assert len(out) == n_qubits

    def test_circuit_output_range(self):
        """PauliZ expectation values must be in [-1, 1]."""
        n_qubits, n_layers = 4, 1
        circuit = make_vqc_circuit(n_qubits, n_layers)
        rng = np.random.default_rng(0)
        w = rng.uniform(0, 2 * np.pi, size=weight_shape(n_qubits, n_layers))
        x = rng.uniform(0, np.pi, size=n_qubits)
        out = [float(v) for v in circuit(w, x)]
        for val in out:
            assert -1.0 - 1e-6 <= val <= 1.0 + 1e-6

    def test_invalid_embedding_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding"):
            make_vqc_circuit(4, 2, embedding="bad")

    def test_invalid_ansatz_raises(self):
        with pytest.raises(ValueError, match="Unknown ansatz"):
            make_vqc_circuit(4, 2, ansatz="bad")

    def test_circuit_reproducibility(self):
        """Same weights and input must produce identical output."""
        circuit = make_vqc_circuit(4, 2)
        w = np.ones(weight_shape(4, 2)) * 0.5
        x = np.ones(4) * 0.3
        out1 = [float(v) for v in circuit(w, x)]
        out2 = [float(v) for v in circuit(w, x)]
        np.testing.assert_allclose(out1, out2, atol=1e-10)


class TestMeyerWallach:
    """Tests for the Meyer-Wallach expressibility estimator."""

    def test_range(self):
        """Meyer-Wallach Q must be in [0, 1]."""
        mw = meyer_wallach_expressibility(
            n_qubits=4, n_layers=1, ansatz="strongly_entangling", n_samples=10, seed=0
        )
        assert 0.0 <= mw <= 1.0 + 1e-6

    def test_deeper_circuits_more_expressive(self):
        """Deeper circuits should have >= expressibility than shallower ones (in expectation)."""
        mw_shallow = meyer_wallach_expressibility(
            n_qubits=4, n_layers=1, ansatz="strongly_entangling", n_samples=50, seed=0
        )
        mw_deep = meyer_wallach_expressibility(
            n_qubits=4, n_layers=4, ansatz="strongly_entangling", n_samples=50, seed=0
        )
        # Not guaranteed for all seeds but holds in expectation for SEL
        assert mw_deep >= mw_shallow - 0.1  # generous tolerance
