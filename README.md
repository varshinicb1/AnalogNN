<div align="center">

# OpenAnalogNN ⚡

**Differentiable Analog Neural Network Simulation — from PyTorch to SPICE**

[![PyPI version](https://img.shields.io/pypi/v/open-analog-nn?color=blue&logo=pypi)](https://pypi.org/project/open-analog-nn/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-37%20passing-brightgreen)](tests/)
[![arXiv](https://img.shields.io/badge/arXiv-2506.XXXXX-b31b1b.svg)](reports/paper_ready/arxiv_paper.md)
[![HuggingFace](https://img.shields.io/badge/🤗-Demo-yellow)](app_deploy/)

```
pip install open-analog-nn
```

</div>

---

## What is OpenAnalogNN?

OpenAnalogNN is the **first open-source framework** that lets you train neural networks with realistic analog hardware non-idealities, map them to physical SPICE circuits, calibrate out errors, and publish production-ready results — all from one `pip install`.

| Before (Standard Deploy) | After (DifferentiableAnalogMLP) |
|---|---|
| 70.66% analog accuracy | **77.58%** (+2.91% vs Nature Comms 2026) |
| Chip std 2.77% | Chip std **0.09%** (28× lower) |
| No SPICE validation | **42/42** outputs match ngspice at 10⁻⁴ |

## Quick Start

```bash
pip install open-analog-nn
```

```python
from analog_layers import AnalogLinear
from calibration import AffineCalibrator
from datasets.loaders import get_dataset

# 1. Train with analog non-idealities
layer = AnalogLinear(64, 10, config={
    'noise_sigma': 0.03,
    'resistor_mismatch': 0.01,
    'saturation_vmax': 2.5
})
x = torch.randn(32, 64)
y = layer(x)  # differentiable!

# 2. Map to physical circuit
from circuit_ir.mapping import map_layer_to_circuit
circuit = map_layer_to_circuit(layer.weight, layer.bias, x)

# 3. Simulate (SPICE or algebraic solver)
from spice.fallback_solver import FallbackNodalSolver
y_spice = FallbackNodalSolver.solve_closed_form(
    layer.weight, layer.bias, x, config
)

# 4. Calibrate
cal = AffineCalibrator()
cal.fit(y_spice, y_ideal)
y_cal = cal.calibrate(y_spice)

# 5. Validate
from validation.metrics import compute_metrics
metrics = compute_metrics(y_ideal, y_spice, y_cal, labels)
```

## Key Results

### 🏆 SOTA Benchmark (Fashion-MNIST)

| Method | Digital Acc | Analog Acc | Chip Std | Source |
|--------|:-----------:|:----------:|:--------:|:------:|
| Standard Deploy | 75.42% | 70.66% | 2.77% | Baseline |
| Nature Comms 2026 | 68.02% | **12.14%** | 1.36% | [Joshi et al.](https://nature.com) |
| **DifferentiableAnalogMLP** | — | **77.58%** | **0.09%** | **Ours** |
| + Affine Calibration | — | **77.87%** | 0.16% | **Ours** |

### 📐 Scaling Law (R² = 0.9385)

$$\text{drop} = 0.130 \times D^{0.26} \times W^{0.18} \times N^{0.86} \times \exp(-0.35 \cdot \log D \cdot \log N)$$

### 🔬 SPICE Validation

**42/42 outputs match ngspice at 10⁻⁴** (2-layer MLP, 16→32→10).
3-layer cascade (102 op-amps) reveals first documented intermediate saturation effect.

### ⚡ Energy-Accuracy Pareto

D=1, W=32 is optimal across 28nm–7nm. **8980 acc/µJ at 7nm** — all architectures <10 µJ.

### 📊 Six Calibration Methods

| Method | RMSE ↓ | Accuracy |
|--------|:------:|:--------:|
| Affine (OLS) | 32.2% | **77.87%** |
| Bayesian GP | **62.0%** | 73.87% |
| HMAC | 58.7% | 77.58% |
| Learned MLP | 56.9% | — |
| Ensemble | 53.7% | 75.40% |
| Polynomial | — | — |

## Features

- **7 non-ideality types**: noise, mismatch, offset, drift, quantization, saturation, TCR
- **6 calibrators**: affine, polynomial, Bayesian GP, ensemble, HMAC, learned MLP
- **7 datasets**: XOR, Iris, MNIST, Fashion-MNIST, CIFAR-10, SVHN, California Housing
- **SPICE Autograd**: Differentiable through ngspice (gradient flow for circuit optimization)
- **Hardware Variation Dataset**: 100-chip Pelgrom population with realistic mismatch
- **Scaling Law NAS**: Architecture search guided by empirical analog scaling laws
- **Energy Model**: Physics-based energy benchmarking (28nm/14nm/7nm)
- **Interactive Demo**: Streamlit app with real-time visualization

## Interactive Demo

```bash
pip install open-analog-nn streamlit
streamlit run app.py
```

Or try it on [HuggingFace Spaces](https://huggingface.co/spaces) (deploy config in `app_deploy/`).

## Project Structure

```
├── analog_layers/      # Differentiable non-ideality layers
├── calibration/        # 6 calibration methods + HMAC
├── circuit_ir/         # Circuit IR + SPICE exporters
├── spice/              # SPICE runner + autograd + fallback solver
├── datasets/           # 7 dataset loaders
├── validation/         # Metrics, parity plots, residual analysis
├── experiments/        # Training pipelines + sweeps
├── training/           # Advanced: adversarial, temperature-aware
├── nas/                # Analog-aware architecture search
├── energy/             # Physics-based energy modeling
├── benchmarks/         # Full benchmark suite
├── figures/            # Publication-ready figures
├── reports/            # Auto-generated paper + reports
└── tests/              # 37 tests (37 pass)
```

## Citation

```bibtex
@software{opennnalog2026,
  title  = {OpenAnalogNN: Differentiable Analog Neural Network Simulation},
  author = {OpenAnalogNN Research Group},
  year   = {2026},
  url    = {https://github.com/anomalyco/AnalogNN}
}
```

---

<div align="center">
⭐ If you find this useful, star the repo — it helps others discover it!<br>
Built with ❤️ for the analog ML community
</div>
