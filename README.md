# OpenAnalogNN: Research-Grade Autonomous Analog Inference Platform

OpenAnalogNN is a scientific infrastructure platform for modeling, simulating, calibrating, and validating analog neural network inference across PyTorch-based neural abstractions and SPICE-level physical circuit simulations.

This is **NOT** a toy simulator or a generic demo; it is a reproducible, publication-grade research framework built for automated hardware-software co-design and neuromorphic circuit experimentation.

---

## 🛠 Mandatory System Architecture

The platform supports a robust, linear-sequential hardware mapping and validation pipeline:

```
Dataset 
  ↓
Digital baseline training (MLP)
  ↓
Analog-aware abstraction (quantization, noise, drift, mismatch, amp offsets)
  ↓
Circuit Intermediate Representation (IR component graph)
  ↓
SPICE Backend Orchestration (or robust Mathematical Nodal Solver fallback)
  ↓
Waveform / DC extraction
  ↓
Cross-layer Validation (RMSE, R², accuracy degradation, parity plots)
  ↓
Calibration (Affine, Polynomial, Deep Learned MLP calibration)
  ↓
Statistical Benchmarking (repeated sweeps, std error/confidence intervals)
  ↓
Automated Report Generation (Markdown files, high-res publication figures, LaTeX tables)
```

---

## 📁 Repository Structure

```
OpenAnalogNN/
├── analog_layers/           # PyTorch layers simulating non-ideal analog hardware behaviors
│   ├── analog_linear.py     # Main layer implementing non-ideal linear matrix multiplication
│   ├── noise_models.py      # Gaussian weight and activation noise
│   ├── drift_models.py      # Device conductance/weight exponential drift over time
│   ├── quantization.py      # DAC/ADC resolution constraints (symmetric/asymmetric)
│   ├── saturation.py        # Op-amp supply rails output voltage limits
│   └── mismatch.py          # Device-to-device static variations (resistor tolerances)
│
├── circuit_ir/              # Backend-agnostic circuit graph representation
│   ├── components.py        # Resistor, Capacitor, Op-Amp, Source component models
│   ├── circuit.py           # Circuit class containing and managing schematic connections
│   ├── exporters/           # Exporters mapping IR graphs to SPICE decks
│   │   ├── ngspice_exporter.py
│   │   └── ltspice_exporter.py
│   └── mapping.py           # [TO IMPLEMENT] Maps weights/biases to differential resistor summer networks
│
├── spice/                   # Automated simulation orchestrator
│   ├── netlist_generator.py # [TO IMPLEMENT] Converts weights to SPICE netlists using Exporters
│   ├── spice_runner.py      # [TO IMPLEMENT] Runs ngspice batch jobs
│   ├── fallback_solver.py   # [TO IMPLEMENT] SciPy-based mathematical nodal equations solver (offline backup)
│   ├── waveform_parser.py   # [TO IMPLEMENT] Parses raw SPICE outputs
│   └── convergence.py       # [TO IMPLEMENT] Detects & handles convergence errors
│
├── calibration/             # Post-simulation correction methods
│   ├── affine.py            # [TO IMPLEMENT] Linear regression (y = ax + b)
│   ├── polynomial.py        # [TO IMPLEMENT] Degree-d polynomial regression
│   ├── learned.py           # [TO IMPLEMENT] Tiny MLP neural calibrator trained in PyTorch
│   └── metrics.py           # [TO IMPLEMENT] Calibration error reduction metrics
│
├── validation/              # Analytical comparisons
│   ├── metrics.py           # [TO IMPLEMENT] RMSE, R², and accuracy degradation
│   ├── parity.py            # [TO IMPLEMENT] Parity plot creators
│   └── statistical.py       # [TO IMPLEMENT] Standard error, confidence intervals, LaTeX tables
│
├── datasets/                # Custom data splits and dataloaders
│   └── loaders.py           # XOR, Iris, and MNIST/Fashion subset loaders with robust offline generator
│
├── experiments/             # High-level sweep and network configurations
│   ├── models.py            # Parameterizable PyTorch Digital MLP class & training utilities
│   └── runner.py            # [TO IMPLEMENT] Runs configurable parametric sweeps
│
├── reports/                 # [TO GENERATE] Directory for auto-generated LaTeX tables & Markdown reports
├── figures/                 # [TO GENERATE] Directory for high-res publication-grade figures
├── reproduce/               # Entrypoint scripts for automated verification
│   └── run_all.py           # [TO IMPLEMENT] Runs baseline -> simulation -> calibration -> report end-to-end
│
└── tests/                   # Code base test cases
```

---

## 🔬 Mathematical Formulations

### Noise
$$w_{eff} = w + \mathcal{N}(0, \sigma^2)$$

### Quantization (DAC/ADC)
$$Q(x) = \frac{\text{round}(x \cdot (2^n - 1))}{2^n - 1}$$

### Drift
$$G(t) = G_0 \exp\left(-\frac{t}{\tau}\right)$$

### Saturation
$$\text{clamp}(x, -V_{max}, V_{max})$$

### Mismatch (Resistor Tolerance)
$$R_{actual} = R_{nominal} \cdot (1 + \delta), \quad \delta \sim \mathcal{N}(0, \sigma_R^2)$$
$$w_{eff} = \frac{w}{1 + \delta}$$

### Op-Amp Input Offset
$$V_{out, offset} = \left(1 + \sum_{j} |w_j|\right) V_{os}$$
