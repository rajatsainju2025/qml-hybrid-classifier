"""Hybrid quantum-classical classifier built as a ``torch.nn.Module``.

The architecture is:
    1. Classical linear pre-processing: input_dim → n_qubits
    2. Quantum circuit layer:           n_qubits  → n_qubits  (expectation values)
    3. Classical post-processing:       n_qubits  → n_classes (softmax)

The quantum layer is wrapped via ``qml.qnn.TorchLayer`` so that its parameters
participate in standard PyTorch autograd differentiation.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import pennylane as qml

from .circuits import make_vqc_circuit, weight_shape


class HybridQClassifier(nn.Module):
    """Hybrid quantum-classical binary/multi-class classifier.

    The model projects classical features to a qubit-sized representation,
    passes them through a parameterised quantum circuit layer, and applies a
    final linear + softmax head.

    Args:
        input_dim: Dimensionality of input features.
        n_qubits: Number of qubits in the PQC.
        n_layers: Number of variational ansatz repetitions.
        n_classes: Number of output classes.
        ansatz: Ansatz family — ``"strongly_entangling"`` or ``"basic_entangling"``.
        embedding: Encoding strategy — ``"angle"`` or ``"amplitude"``.

    Attributes:
        pre: Classical linear projection (input_dim → n_qubits).
        qlayer: PennyLane TorchLayer wrapping the QNode.
        post: Classical linear head (n_qubits → n_classes).

    Example:
        >>> model = HybridQClassifier(input_dim=30, n_qubits=8, n_layers=3)
        >>> x = torch.randn(4, 30)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([4, 2])
    """

    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 8,
        n_layers: int = 3,
        n_classes: int = 2,
        ansatz: Literal["strongly_entangling", "basic_entangling"] = "strongly_entangling",
        embedding: Literal["angle", "amplitude"] = "angle",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.ansatz = ansatz
        self.embedding = embedding

        # 1 — Classical pre-processing
        self.pre = nn.Linear(input_dim, n_qubits)

        # 2 — Quantum circuit layer
        circuit = make_vqc_circuit(n_qubits, n_layers, ansatz, embedding)
        w_shape = {"weights": weight_shape(n_qubits, n_layers, ansatz)}
        self.qlayer = qml.qnn.TorchLayer(circuit, w_shape)

        # 3 — Classical post-processing
        self.post = nn.Linear(n_qubits, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class log-probabilities for a batch of inputs.

        Args:
            x: Float tensor of shape ``(batch_size, input_dim)``.

        Returns:
            Log-softmax tensor of shape ``(batch_size, n_classes)``.
        """
        h = torch.tanh(self.pre(x))  # (B, n_qubits) — scale to [-1, 1] ≈ angle range
        # TorchLayer expects (batch_size, n_qubits); returns (batch_size, n_qubits)
        q_out = self.qlayer(h)
        return torch.log_softmax(self.post(q_out), dim=-1)

    @property
    def n_quantum_params(self) -> int:
        """Number of trainable parameters inside the quantum circuit."""
        return sum(p.numel() for p in self.qlayer.parameters())

    @property
    def n_classical_params(self) -> int:
        """Number of trainable parameters in the classical pre/post layers."""
        return sum(p.numel() for p in self.pre.parameters()) + sum(
            p.numel() for p in self.post.parameters()
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"HybridQClassifier("
            f"ansatz={self.ansatz}, "
            f"embedding={self.embedding}, "
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"quantum_params={self.n_quantum_params}, "
            f"classical_params={self.n_classical_params})"
        )
