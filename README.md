# qml-hybrid-classifier

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.38-brightgreen.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/rsainju/qml-hybrid-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/rsainju/qml-hybrid-classifier/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> **Hybrid quantum-classical classifier using parameterized quantum circuits (PQC)
> with Bayesian-optimized circuit architecture search.**

---

## Motivation

Variational quantum algorithms (VQAs) represent a leading candidate for near-term
quantum advantage, combining the expressibility of parameterized quantum circuits
with classical gradient-based optimization [Cerezo et al., 2021; arXiv:2012.09265].
For classification tasks, hybrid quantum-classical models have been shown to achieve
competitive accuracy on structured datasets while using fewer parameters exponentially
than classical neural networks in certain regimes [Biamonte et al., 2017;
arXiv:1611.09347]. This project implements a rigorous benchmark of VQC variants
against classical baselines on the Wisconsin Breast Cancer dataset, with circuit
architecture search framed as a Bayesian optimization problem.

**Key references:**
- Cerezo, M. et al. (2021). *Variational quantum algorithms.* Nature Reviews Physics 3, 625–644. [arXiv:2012.09265](https://arxiv.org/abs/2012.09265)
- Biamonte, J. et al. (2017). *Quantum machine learning.* Nature 549, 195–202. [arXiv:1611.09347](https://arxiv.org/abs/1611.09347)
- Sim, S. et al. (2019). *Expressibility and entangling capability of parameterized quantum circuits.* Advanced Quantum Technologies. [arXiv:1905.10876](https://arxiv.org/abs/1905.10876)
- Holmes, Z. et al. (2022). *Connecting ansatz expressibility to gradient magnitudes.* PRX Quantum 3, 010313. [arXiv:2101.02138](https://arxiv.org/abs/2101.02138)

---

## Architecture

```
Classical input (30 features)
         │
         ▼
┌─────────────────────┐
│  Linear projection  │  30 → n_qubits   (trainable)
└─────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              Quantum Circuit Layer (QNode)              │
│                                                         │
│  ┌───────────────────────────────────────┐              │
│  │  Data Embedding                        │             │
│  │  AngleEmbedding (Rx)  or              │              │
│  │  AmplitudeEmbedding                   │              │
│  └───────────────────────────────────────┘              │
│                      │                                  │
│                      ▼                                  │
│  ┌───────────────────────────────────────┐              │
│  │  Variational Ansatz (L layers)         │             │
│  │  StronglyEntanglingLayers  or          │             │
│  │  BasicEntanglingLayers                 │             │
│  └───────────────────────────────────────┘              │
│                      │                                  │
│          ⟨Z₀⟩  ⟨Z₁⟩  …  ⟨Zₙ⟩  (expectation values)      │ 
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Linear + Softmax   │  n_qubits → 2   (trainable)
└─────────────────────┘
         │
         ▼
  Class probabilities
```

The circuit architecture (n_qubits, n_layers, ansatz, embedding) is treated as a
discrete hyperparameter space over which Bayesian optimisation (BO) is applied using
`scikit-optimize` to find the Pareto-optimal frontier between accuracy and circuit depth.

---

## Quickstart

```bash
git clone https://github.com/rsainju/qml-hybrid-classifier.git
cd qml-hybrid-classifier
conda env create -f environment.yml && conda activate qml-hybrid
pip install -e .
python experiments/run_experiment.py --config experiments/configs/baseline_vqc.yaml
```

Results are saved to `results/tables/` and `results/figures/`.

---

## Experiments

Results will be populated after running the full experiment suite:

```bash
python experiments/run_experiment.py --config experiments/configs/baseline_vqc.yaml
python experiments/run_ablation.py --config experiments/configs/ablation_depth.yaml
```

Output tables are saved to `results/tables/` and figures to `results/figures/`.

---

## Statistical Significance

Classifier comparison uses **McNemar's test** on matched per-sample predictions
(paired, non-parametric), appropriate for comparing two binary classifiers on the
same test set. Multi-model comparisons use Bonferroni correction. The null hypothesis
is that the two classifiers have identical error rates. Reported p-values are
two-sided; significance threshold α = 0.05.

---

## Limitations & Future Work

**Known limitations:**
- **Barren plateaus:** Gradient variance decays exponentially with qubit count for
  global cost functions [McClean et al., 2018; arXiv:1803.11173]. This implementation
  uses local cost (per-qubit PauliZ) which partially mitigates but does not eliminate
  this issue. Circuits beyond ~12 qubits on this architecture will likely fail to train.
- **Shot noise:** Simulation uses exact expectation values (`default.qubit`); real
  hardware requires finite shot estimation, introducing variance proportional to 1/√shots.
- **Hardware noise:** NISQ devices introduce gate errors, decoherence, and readout
  errors not modelled here. Noise-aware training (e.g., via `qml.device("default.mixed")`)
  is a direct extension.
- **Encoding bottleneck:** AngleEmbedding requires classical pre-processing to reduce
  features to n_qubits; information loss is not quantified.

**Future work:**
- Data re-uploading circuits [Pérez-Salinas et al., 2020] for richer feature encoding
- Quantum kernel methods as an alternative to VQC for this dataset size
- Hardware execution on IBM Quantum or QuEra Aquila via PennyLane device plugins
- Quantum reinforcement learning extension using the same PQC backbone as policy network

---

## Citation

```bibtex
@software{sainju2024qmlhybrid,
  author       = {Sainju, Rajat},
  title        = {{qml-hybrid-classifier: Hybrid Quantum-Classical Variational
                   Circuit Classifier with Bayesian Architecture Search}},
  year         = {2024},
  publisher    = {GitHub},
  url          = {https://github.com/rsainju/qml-hybrid-classifier},
  license      = {MIT}
}
```

---

## License

MIT © Rajat Sainju — see [LICENSE](LICENSE).
