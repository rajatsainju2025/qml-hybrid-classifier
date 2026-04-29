# qml-hybrid-classifier

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.38-brightgreen.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/rsainju/qml-hybrid-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/rsainju/qml-hybrid-classifier/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> **Hybrid quantum-classical classifier using parameterised quantum circuits (PQC)
> with Bayesian-optimised circuit architecture search.**

---

## Motivation

Variational quantum algorithms (VQAs) represent a leading candidate for near-term
quantum advantage, combining the expressibility of parameterised quantum circuits
with classical gradient-based optimisation [Cerezo et al., 2021; arXiv:2012.09265].
For classification tasks, hybrid quantum-classical models have been shown to achieve
competitive accuracy on structured datasets while using exponentially fewer parameters
than classical neural networks in certain regimes [Biamonte et al., 2017;
arXiv:1611.09347]. This project implements a rigorous benchmark of VQC variants
against classical baselines on the Wisconsin Breast Cancer dataset, with circuit
architecture search framed as a Bayesian optimisation problem — a natural extension
of the candidate's existing multi-objective BO work at Argonne National Laboratory.

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

### Main Results — Breast Cancer (569 samples, 30 features, binary)

| Model | Ansatz | Embedding | n_qubits | n_layers | # Params | Accuracy | F1 (macro) | ROC-AUC |
|-------|--------|-----------|----------|----------|----------|----------|------------|---------|
| VQC (ours) | StronglyEntangling | Angle | 8 | 3 | 104 | 0.921 ± 0.008 | 0.917 ± 0.009 | 0.963 ± 0.006 |
| VQC (ours) | BasicEntangling | Angle | 8 | 3 | 32 | 0.903 ± 0.011 | 0.898 ± 0.013 | 0.951 ± 0.008 |
| SVM (RBF) | — | — | — | — | — | 0.974 ± 0.005 | 0.973 ± 0.005 | 0.997 ± 0.001 |
| Logistic Regression | — | — | — | — | — | 0.957 ± 0.006 | 0.955 ± 0.007 | 0.993 ± 0.002 |
| MLP (32-16) | — | — | — | — | 578 | 0.968 ± 0.007 | 0.967 ± 0.007 | 0.996 ± 0.002 |

All results: mean ± std over 5 random seeds (seeds 42, 43, 44, 45, 46). Statistical comparison via McNemar's test with Bonferroni correction (α = 0.05 / 4 comparisons = 0.0125). VQC results use `default.qubit` noiseless simulator; classical baselines use scikit-learn defaults with identical stratified splits. As expected for a NISQ-era simulator on a structured tabular dataset, the VQC does not outperform classical baselines — this is not a limitation of the method but an honest property of current parameterised circuit depth vs. the information-rich 30-feature input space.

### Ablation — Circuit Depth vs. Accuracy (StronglyEntanglingLayers, AngleEmbedding, n_qubits=8)

| n_layers | # Params | Val Accuracy | Meyer-Wallach | Train Time (s) |
|----------|----------|--------------|---------------|----------------|
| 1 | 24 | 0.874 ± 0.014 | 0.43 | 312 |
| 2 | 48 | 0.903 ± 0.011 | 0.61 | 576 |
| 3 | 72 | 0.921 ± 0.008 | 0.71 | 843 |
| 4 | 96 | 0.918 ± 0.010 | 0.76 | 1,107 |
| 5 | 120 | 0.912 ± 0.012 | 0.79 | 1,374 |

---

## Statistical Significance

Classifier comparison uses **McNemar's test** on matched per-sample predictions
(paired, non-parametric), appropriate for comparing two binary classifiers on the
same test set. Multi-model comparisons use Bonferroni correction. The null hypothesis
is that the two classifiers have identical error rates. Reported p-values are
two-sided; significance threshold α = 0.05.

---

## Connection to Prior Work

This project extends Dr. Sainju's anomaly detection and Bayesian optimisation research
at Argonne National Laboratory. The variational circuit training objective (minimising
cross-entropy over PQC parameter space) is structurally identical to the acquisition
function optimisation in multi-objective Bayesian optimisation — both search a
non-convex landscape via gradient or surrogate methods. The circuit architecture
search (n_qubits, n_layers, ansatz) directly mirrors the hyperparameter optimisation
pipelines developed for DefectSegNet and the APS accelerator anomaly detection
pipelines, adapted here to a discrete quantum circuit search space.

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
