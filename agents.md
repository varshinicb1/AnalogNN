# AI Agent Handover & Architecture Specification

Welcome to the development of **OpenAnalogNN**! This document provides the exact technical specification, mathematics, design decisions, and concrete next steps to complete the system. Follow this guide to continue building the codebase systematically.

---

## 🏁 Current Project State

The core dataset loading, baseline training, and PyTorch mathematical non-ideality simulation layers have been successfully implemented:

| Module / Component | Implemented File | Description / Functionality |
| :--- | :--- | :--- |
| **Configuration** | [config.yaml](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/configs/config.yaml) | YAML parameters for training, noise, drift, mismatch, circuit ref, calibration methods. |
| **Datasets** | [loaders.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/datasets/loaders.py) | Dataloaders for XOR, Iris, and downsampled MNIST/Fashion, featuring a robust procedural digit generator for offline reliability. |
| **Digital Baseline** | [models.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/experiments/models.py) | Highly configurable PyTorch MLP, loss tracking, train/test evaluator, and publication-ready plotter. |
| **Analog Non-Idealities** | [noise_models.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/analog_layers/noise_models.py) | Temporal additive Gaussian weight and read noise. |
| | [drift_models.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/analog_layers/drift_models.py) | Conductance physical decay: $G(t) = G_0 \exp(-t/\tau)$. |
| | [quantization.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/analog_layers/quantization.py) | Uniform symmetric/asymmetric quantization modeling finite resolution DACs/ADCs. |
| | [saturation.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/analog_layers/saturation.py) | Voltage clamping modeling op-amp supply rail boundaries. |
| | [mismatch.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/analog_layers/mismatch.py) | Static resistor tolerance mismatch: $w_{eff} = w / (1+\delta)$. |
| | [analog_linear.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/analog_layers/analog_linear.py) | PyTorch `nn.Module` linear layer wrapper running the non-ideality cascade, incorporating closed-loop op-amp input offsets. |
| **Circuit IR** | [components.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/circuit_ir/components.py) | Components definition: `Resistor`, `Capacitor`, `OpAmp`, `VoltageSource`, `CurrentSource`. |
| | [circuit.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/circuit_ir/circuit.py) | Unified circuit graph tracking nodes and component connections. |
| | [ngspice_exporter.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/circuit_ir/exporters/ngspice_exporter.py) | Serializer creating convergence-stable `ngspice` deck files with behavioral clamping. |
| | [ltspice_exporter.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/circuit_ir/exporters/ltspice_exporter.py) | Serializer creating `LTspice` compatible deck files. |

---

## 🛠 Complete Architectural Specifications for Modules to Build

### 1. Circuit Mapping Engine (`circuit_ir/mapping.py`)
This module maps weights $W \in \mathbb{R}^{M \times N}$ and biases $b \in \mathbb{R}^{M}$ of an `AnalogLinear` layer to physical circuit topologies inside our `Circuit` IR.

*   **Hardware Topology**: Op-Amp Differential Summing Amplifier.
    For each output neuron $i \in \{1, \dots, M\}$:
    We split weights into positive weights ($w_{ij} > 0$) and negative weights ($w_{ij} < 0$).
    We map these to a differential architecture:
    *   **Positive Summer Node** ($V^-_{i, pos}$): Accumulates negative contributions (since inverting):
        $$V_{out, pos} = - \sum_{w_{ij} > 0} \frac{R_f}{R_{ij}} V_{in, j} - \frac{R_f}{R_{bias, i}^+} V_{bias}$$
    *   **Negative Summer Node** ($V^-_{i, neg}$): Accumulates positive contributions:
        $$V_{out, neg} = - \sum_{w_{ij} < 0} \frac{R_f}{R_{ij}} V_{in, j} - \frac{R_f}{R_{bias, i}^-} V_{bias}$$
    *   **Differential Subtractor Stage**: An op-amp subtracting $V_{out, neg}$ from $V_{out, pos}$:
        $$V_{out, i} = V_{out, pos} - V_{out, neg} = \sum_{w_{ij} > 0} \frac{R_f}{R_{ij}} V_{in, j} - \sum_{w_{ij} < 0} \frac{R_f}{R_{ij}} V_{in, j} + \text{bias}$$
        *By setting $R_f = R_{ref}$ and $R_{ij} = R_{ref} / |w_{ij}|$, the voltage output is mathematically equivalent to the linear matrix multiplication!*
*   **Design Details**:
    *   Map activations $x_j$ to voltage sources: $V_{in, j} = x_j \cdot V_{ref}$.
    *   Map weights: $R_{ij} = R_{ref} / \max(|w_{ij}|, \epsilon)$ to avoid division-by-zero. If a weight $|w_{ij}| < 10^{-6}$, omit the resistor or set it to an open circuit (extremely high value like $10^{12} \ \Omega$).
    *   Treat bias $b_i$ as an extra input from $V_{bias} = V_{ref}$. If $b_i > 0$, connect a resistor $R_{bias, i} = R_{ref}/b_i$ to the positive summing node. If $b_i < 0$, connect it to the negative summing node.
    *   Output of the differential stage represents the prediction logits. Saturation limits are modeled by setting the op-amp model supply rails to $\pm V_{max}$.

---

### 2. SPICE Orchestration & Fallback Solver (`spice/`)
This module automates the execution of simulations and provides a mathematical nodal solver fallback for machines that do not have `ngspice` installed.

#### SPICE Orchestrator (`spice/netlist_generator.py` & `spice/spice_runner.py`)
*   Takes the mapped `Circuit` IR and writes `.cir` netlist files.
*   Calls `ngspice` in batch mode: `ngspice -b -r raw_file netlist_file`.
*   Parses the resulting `.raw` binary or ASCII output files (`spice/waveform_parser.py`) to extract nodal voltages of output op-amps.

#### Fallback Nodal Solver (`spice/fallback_solver.py`)
To ensure that OpenAnalogNN is 100% stable, offline-compatible, and zero-install, we build a mathematical **Nodal Equation Solver** using standard linear algebra (`scipy.linalg` or `scipy.sparse.linalg`).
*   **Physics Formulation**: For a resistor-opamp network, we solve Kirchhoff's Current Law (KCL) at each node.
    $$G \cdot V = I$$
    where $G$ is the conductance matrix, $V$ is the vector of node voltages, and $I$ is the vector of independent current/voltage feeds.
*   **Simplified Algebraic Equivalence**: Since the summing circuit is an ideal op-amp network, the voltage output $V_{out, i}$ of the subtractor for neuron $i$ is analytically:
    $$V_{out, i} = \text{clamp}\left( \sum_{j} w_{ij} x_j + b_i, -V_{max}, V_{max} \right)$$
    With resistor mismatch $\delta_{R, ij}$, the actual node voltage solves:
    $$V_{out, i} = \text{clamp}\left( \sum_{j} \frac{w_{ij}}{1 + \delta_{ij}} x_j + \frac{b_i}{1 + \delta_{b, i}} + V_{os, i} \cdot \left(1 + \sum_{j} \frac{|w_{ij}|}{1 + \delta_{ij}}\right), -V_{max}, V_{max} \right)$$
    *The Fallback Solver must implement this closed-form algebraic network simulation. This yields identical outputs to SPICE for linear resistor networks and runs thousands of times faster, which is perfect for fast parameter sweeps!*

---

### 3. Calibration Engine (`calibration/`)
Corrects systemic circuit non-idealities (gain errors from mismatch, DC amplifier offsets) by finding a mapping from non-ideal simulated voltages $y_{spice}$ to ideal mathematical activations $y_{ideal}$.

*   **Affine Calibration (`calibration/affine.py`)**:
    Fits a linear regressor for each output class:
    $$y_{cal} = a \cdot y_{spice} + b$$
    Solved using ordinary least squares (OLS).
*   **Polynomial Calibration (`calibration/polynomial.py`)**:
    Fits a degree-$d$ polynomial (default $d=3$):
    $$y_{cal} = \sum_{k=0}^d a_k y_{spice}^k$$
    Solved using numpy polynomial fitting.
*   **Learned Calibration (`calibration/learned.py`)**:
    A tiny MLP network built in PyTorch (e.g. Input (10) -> Hidden (16) -> Output (10)) trained using SGD/Adam to minimize the mean squared error (MSE) loss:
    $$\mathcal{L}_{cal} = \|y_{cal} - y_{ideal}\|^2$$
    This is highly effective at resolving non-linear offsets caused by saturation or cascading mismatch!

---

### 4. Cross-Layer Validation Engine (`validation/`)
Calculates rigorous metrics comparing three layers:
1.  **Ideal Layer**: Digital floating-point weights without non-idealities.
2.  **Abstract Analog Layer**: Mathematical PyTorch simulation of noise, quantization, etc.
3.  **Circuit Sim (SPICE / Solver)**: The mapped circuit netlist nodal voltage outputs.

*   **Metrics (`validation/metrics.py`)**: RMSE, Pearson correlation coefficient ($R$), and classification accuracy drop.
*   **Plotting (`validation/parity.py`)**: Generates parity scatter plots ($y_{ideal}$ on x-axis vs $y_{spice}$ on y-axis) comparing pre-calibration and post-calibration states.

---

### 5. Automated Experiment & Sweeps (`experiments/runner.py`)
Automates sweeps across multi-dimensional parameters to evaluate hardware robustness:
*   Sweep ranges: Noise standard deviation ($\sigma \in [0.0, 0.25]$), Resistor mismatch tolerance ($\delta_R \in [0\%, 10\%]$), Quantization bits ($n \in [4, 8]$), and Drift time constant.
*   Runs repeated runs with different random seeds to calculate **Mean, Standard Deviation, Standard Error, and 95% Confidence Intervals (CI)**.
*   Outputs structured LaTeX summary tables ready for academic publication.

---

## 🚀 Concrete Step-by-Step Next Actions

When you start implementing, follow this exact sequence:

1.  **Task 4: Implement Circuit Mapping Engine**
    *   Create [mapping.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/circuit_ir/mapping.py).
    *   Write a function `map_layer_to_circuit(analog_linear: AnalogLinear, x: torch.Tensor) -> Circuit` which constructs the components (`VoltageSource` for inputs, `Resistor` for positive/negative weights and bias, and `OpAmp` for summing and subtractor blocks) and links them in a `Circuit` IR object.
2.  **Task 5: Implement Fallback Nodal Solver & SPICE Runner**
    *   Create [fallback_solver.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/spice/fallback_solver.py) implementing the closed-form physical nodal voltage equations.
    *   Create [spice_runner.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/spice/spice_runner.py) and [waveform_parser.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/spice/waveform_parser.py) to manage ngspice executables. Make sure the runner degrades gracefully to the fallback solver if `ngspice` is not found in the environment PATH.
3.  **Task 6: Implement Calibration Engine**
    *   Create [affine.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/calibration/affine.py), [polynomial.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/calibration/polynomial.py), and [learned.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/calibration/learned.py).
4.  **Task 7: Implement Validation metrics & Parity plotting**
    *   Create [metrics.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/validation/metrics.py) and [parity.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/validation/parity.py).
5.  **Task 8: Implement Sweeps and LaTeX formatting**
    *   Create [runner.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/experiments/runner.py) to run parametric sweeps, aggregate confidence intervals, and write out reports.
6.  **Task 9: Implement Entrypoint Script**
    *   Create [run_all.py](file:///c:/Users/varsh/OneDrive/Documents/6THSEM/AnalogNN/reproduce/run_all.py). This will load MNIST downsampled data, train the baseline, run the mapping, run the SPICE/Nodal solver, execute calibration, perform sweeps, and write a publication Markdown report in `reports/` and figures in `figures/`.
7.  **Task 10: Run and Verify**
    *   Verify the system using standard tests: `pytest tests/`.
