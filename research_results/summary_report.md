# Research Findings Summary

Generated: 2026-06-20 03:18:16

## 1. Phase Transitions

Critical mismatch thresholds where accuracy collapses:

- **IRIS**: 30.0% mismatch (max drop: -0.6000)
- **MNIST**: 24.0% mismatch (max drop: -0.7240)
- **FASHION**: 28.0% mismatch (max drop: -0.6000)

## 2. Scaling Laws

Architecture robustness patterns:


### IRIS
- D1-W64: Robustness = 1.0353
- D2-W64: Robustness = 1.0042
- D3-W64: Robustness = 0.9878
- D4-W64: Robustness = 0.9959
- D2-W32: Robustness = 0.9795
- D2-W64: Robustness = 1.0000
- D2-W128: Robustness = 1.0000
- D2-W256: Robustness = 1.0042

### MNIST
- D1-W64: Robustness = 0.9798
- D2-W64: Robustness = 0.9776
- D3-W64: Robustness = 0.9753
- D4-W64: Robustness = 0.8711
- D2-W32: Robustness = 0.9871
- D2-W64: Robustness = 0.9663
- D2-W128: Robustness = 0.9358
- D2-W256: Robustness = 0.8922

### FASHION
- D1-W64: Robustness = 0.8843
- D2-W64: Robustness = 0.7646
- D3-W64: Robustness = 0.5995
- D4-W64: Robustness = 0.5447
- D2-W32: Robustness = 0.7692
- D2-W64: Robustness = 0.8209
- D2-W128: Robustness = 0.8208
- D2-W256: Robustness = 0.7839

## 3. Calibration Effectiveness

Best calibration methods by architecture:


### IRIS
- **Shallow**: Polynomial (error: 0.0061)
- **Deep**: Polynomial (error: 0.5558)
- **Wide**: Polynomial (error: 1.0985)

### MNIST
- **Shallow**: Learned (error: 0.6777)
- **Deep**: Learned (error: 1.2279)
- **Wide**: Learned (error: 1.5465)

### FASHION
- **Shallow**: Learned (error: 0.9734)
- **Deep**: Learned (error: 1.6216)
- **Wide**: Learned (error: 1.6991)

## 4. Energy-Accuracy Tradeoffs

Most efficient architectures:


### IRIS
- [32]: Acc=0.6400, Energy=7.88e-08J, Efficiency=8.12e+06
- [64]: Acc=0.6800, Energy=1.45e-07J, Efficiency=4.68e+06
- [128]: Acc=0.8200, Energy=2.82e-07J, Efficiency=2.91e+06

### MNIST
- [32]: Acc=0.7920, Energy=3.22e-07J, Efficiency=2.46e+06
- [64]: Acc=0.7980, Energy=5.65e-07J, Efficiency=1.41e+06
- [64, 64]: Acc=0.8300, Energy=7.98e-07J, Efficiency=1.04e+06

### FASHION
- [32]: Acc=0.7420, Energy=2.65e-07J, Efficiency=2.80e+06
- [64]: Acc=0.7700, Energy=4.45e-07J, Efficiency=1.73e+06
- [64, 64]: Acc=0.7760, Energy=7.01e-07J, Efficiency=1.11e+06
