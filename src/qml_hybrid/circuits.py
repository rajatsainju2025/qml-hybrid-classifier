"""Parameterised quantum circuit definitions for the hybrid VQC classifier.

Supports two ansatz families (StronglyEntangling, BasicEntangling) and two
data-embedding strategies (AngleEmbedding, AmplitudeEmbedding), all composable
as PennyLane QNodes.

References:
    Cerezo, M. et al. (2021). Variational quantum algorithms.
        Nature Reviews Physics. arXiv:2012.09265
    Sim, S. et al. (2019). Expressibility and entangling capability of PQCs.
        Advanced Quantum Technologies. arXiv:1905.10876
    Holmes, Z. et al. (2022). Connecting ansatz expressibility to gradient
        magnitudes. PRX Quantum. arXiv:2101.02138
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pennylane as qml


def _build_device(n_qubits: int) -> qml.Device:
    """Instantiate a noiseless ``default.qubit`` PennyLane simulator.

    Args:
        n_qubits: Number of qubits in the circuit.

    Returns:
        Configured PennyLane device.
    """
    return qml.device("default.qubit", wires=n_qubits)


def weight_shape(
    n_qubits: int,
    n_layers: int,
    ansatz: Literal["strongly_entangling", "basic_entangling"] = "strongly_entangling",
) -> tuple[int, ...]:
    """Return the weight tensor shape for the chosen ansatz.

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of ansatz repetitions.
        ansatz: Ansatz family name.

    Returns:
        Shape tuple compatible with ``torch.nn.Parameter`` initialisation.

    Raises:
        ValueError: For unrecognised ansatz names.
    """
    if ansatz == "strongly_entangling":
        return (n_layers, n_qubits, 3)
    if ansatz == "basic_entangling":
        return (n_layers, n_qubits)
    raise ValueError(f"Unknown ansatz '{ansatz}'. Use 'strongly_entangling' or 'basic_entangling'.")


def make_vqc_circuit(
    n_qubits: int,
    n_layers: int,
    ansatz: Literal["strongly_entangling", "basic_entangling"] = "strongly_entangling",
    embedding: Literal["angle", "amplitude"] = "angle",
) -> qml.QNode:
    """Factory returning a QNode implementing the full VQC forward pass.

    Architecture:
        1. Data-embedding layer  (AngleEmbedding or AmplitudeEmbedding)
        2. Variational ansatz    (StronglyEntangling or BasicEntangling)
        3. Measurement           (PauliZ expectation on every qubit)

    The returned QNode has signature ``(weights, x) -> list[float]`` and is
    wrapped with the ``"torch"`` interface so gradients flow through
    ``torch.autograd``.

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of ansatz repetitions (circuit depth).
        ansatz: Variational layer family to use.
        embedding: Data-encoding strategy.

    Returns:
        PennyLane QNode with PyTorch interface.

    Raises:
        ValueError: For unrecognised ``ansatz`` or ``embedding`` strings.
    """
    if embedding not in ("angle", "amplitude"):
        raise ValueError(f"Unknown embedding '{embedding}'. Use 'angle' or 'amplitude'.")
    if ansatz not in ("strongly_entangling", "basic_entangling"):
        raise ValueError(
            f"Unknown ansatz '{ansatz}'. Use 'strongly_entangling' or 'basic_entangling'."
        )

    dev = _build_device(n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(dev, interface="torch")
    def circuit(weights, x):
        # --- Data embedding ---
        if embedding == "angle":
            qml.AngleEmbedding(x, wires=wires, rotation="X")
        else:
            qml.AmplitudeEmbedding(x, wires=wires, normalize=True)

        # --- Variational ansatz ---
        if ansatz == "strongly_entangling":
            qml.StronglyEntanglingLayers(weights, wires=wires)
        else:
            qml.BasicEntanglingLayers(weights, wires=wires)

        # --- Measurement ---
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    return circuit


def _reduced_dm_purity(state: np.ndarray, qubit: int, n_qubits: int) -> float:
    """Compute purity Tr(ρ_k²) of the single-qubit reduced density matrix.

    Args:
        state: Full statevector of length ``2**n_qubits``.
        qubit: Index of the qubit to retain after partial trace.
        n_qubits: Total number of qubits.

    Returns:
        Purity value in [0.25, 1.0]; 0.5 indicates maximally mixed single qubit.
    """
    psi = state.reshape([2] * n_qubits)
    # Move target qubit to axis 0, then collapse remaining axes
    psi_t = np.moveaxis(psi, qubit, 0).reshape(2, -1)  # (2, 2^{n-1})
    rho = psi_t @ psi_t.conj().T  # (2, 2) reduced density matrix
    return float(np.real(np.trace(rho @ rho)))


def meyer_wallach_expressibility(
    n_qubits: int,
    n_layers: int,
    ansatz: Literal["strongly_entangling", "basic_entangling"] = "strongly_entangling",
    n_samples: int = 200,
    seed: int = 42,
) -> float:
    """Estimate the Meyer–Wallach global entanglement measure Q.

    Q ∈ [0, 1]: higher values indicate greater entangling capability and
    expressibility.  Uses a Monte Carlo estimator over random parameter draws
    following Sim et al. (2019) arXiv:1905.10876.

    The measure is defined as:
        Q = (2/n_qubits) * E_{θ}[Σ_k (1 - Tr(ρ_k²))]
    where ρ_k is the single-qubit reduced density matrix for qubit k.

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of variational layers.
        ansatz: Ansatz family to evaluate.
        n_samples: Number of random parameter draws for Monte Carlo estimation.
        seed: Random seed for reproducibility.

    Returns:
        Estimated Meyer–Wallach measure in [0, 1].
    """
    rng = np.random.default_rng(seed)
    dev = _build_device(n_qubits)
    wires = list(range(n_qubits))
    shape = weight_shape(n_qubits, n_layers, ansatz)

    @qml.qnode(dev, interface="numpy")
    def _state_circuit(weights, x):
        qml.AngleEmbedding(x, wires=wires, rotation="X")
        if ansatz == "strongly_entangling":
            qml.StronglyEntanglingLayers(weights, wires=wires)
        else:
            qml.BasicEntanglingLayers(weights, wires=wires)
        return qml.state()

    entanglement_sum = 0.0
    for _ in range(n_samples):
        w = rng.uniform(0, 2 * np.pi, size=shape)
        x = rng.uniform(0, np.pi, size=n_qubits)
        state = np.array(_state_circuit(w, x))
        linear_entropy_sum = sum(
            1.0 - _reduced_dm_purity(state, k, n_qubits) for k in range(n_qubits)
        )
        entanglement_sum += (2.0 / n_qubits) * linear_entropy_sum

    return entanglement_sum / n_samples
