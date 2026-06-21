# OpenAnalogNN — Executive Summary

**OpenAnalogNN** is an open-source framework for differentiable analog neural network simulation, calibration, and hardware validation. It bridges the gap between software-trained weights and physical resistor-op-amp circuits by providing end-to-end differentiable non-ideality simulation, automatic SPICE netlist generation, closed-form analytical solvers (verified 42/42 outputs match ngspice at 1e-4), and six calibration methods — all in a single `pip install open-analog-nn` package.

## Key Results

- **DifferentiableAnalogMLP**: 75.04% analog accuracy on Fashion-MNIST (only 1.17% digital drop), beating Nature Comms 2026 by **+2.91%** with **28× lower chip variance**
- **Scaling law** (R²=0.9385): accuracy drop = 0.130 × D^0.26 × W^0.18 × N^0.86 × exp(-0.35·log(D)·log(N)) — depth primarily hurts through noise amplification, not directly
- **SPICE validation**: closed-form solver matches ngspice across **42/42 outputs** at 1e-4 tolerance, running **1000× faster**
- **Best calibration accuracy**: Affine calibration achieves **77.87%** (+0.29% over uncalibrated) on Fashion-MNIST
- **Best RMSE reduction**: Bayesian GP calibration achieves **62.0%** RMSE reduction
- **Energy-accuracy Pareto**: D=1, W=32 is optimal across 28nm–7nm nodes, consuming **0.1 nJ/inference at 7nm**
- **Test suite**: **39 tests** (37 pass, 2 skipped) covering core layers, non-idealities, calibrators, datasets, and scaling law NAS

## Component Summary

| Component | Key Takeaway |
|:---|---|
| **Datasets** | 7 built-in datasets (XOR, Iris, MNIST, Fashion-MNIST, CIFAR-10, SVHN, California Housing) with procedural generators for offline reliability |
| **Non-idealities** | 5 types (mismatch, offset, drift, quantization, saturation) implemented as differentiable PyTorch operations, enabling gradient-based training under hardware constraints |
| **Calibration** | 6 methods (affine, polynomial, learned MLP, HMAC, Bayesian GP, ensemble) — affine gives best accuracy, Bayesian GP gives best RMSE reduction |
| **Scaling law** | Empirical model from 504 sweeps; depth exponent (0.26) > width exponent (0.18) confirming width is the preferred scaling axis |
| **SPICE** | Automated ngspice batch execution + closed-form fallback solver; 42/42 outputs match at 1e-4; 1000× speedup over real SPICE |
| **NAS / Energy** | Pareto analysis across 28nm–7nm nodes finds 1-layer, 32-wide is energy-optimal; all architectures <10 µJ per inference |
| **Temperature** | Johnson-Nyquist noise negligible (0.04%); on-chip polysilicon TCR causes 4.8% drift over 60°C; temperature-aware training mitigates without extra hardware |

## Quick Start

```bash
pip install open-analog-nn && streamlit run app.py
```

---

*OpenAnalogNN v0.2.0 — https://github.com/opencode/AnalogNN*
