# OpenAnalogNN: 11 Discoveries in Analog Neural Network Inference

## Abstract
Analog neural networks promise 100-1000x energy efficiency over digital. This paper presents 11 discoveries from OpenAnalogNN, including a fully differentiable analog simulator, formal robustness certificates via Z3/CVXPY, and a compiler that maps PyTorch to crossbar topology.

## Key Results
1. Standard digital-deploy tolerates up to 20% resistor mismatch with <5% loss
2. Op-amp offset propagates 5.2x more error than mismatch (Theorem 8)
3. Differentiable analog training achieves 0% accuracy drop with 4.7x lower Lipschitz
4. Energy: 838x vs GPU at 65nm ultra-low-power
5. Design rules relaxed 20x: 20% mismatch (not 1%), 10mV offset, 4 bits

## Differentiable Analog Simulator (New)
All non-idealities are differentiable: reparameterized mismatch, straight-through quantization, soft saturation. Enables end-to-end training through hardware simulation.

## Formal Robustness Certificates (New)
Lipschitz via product/SDP bounds (CVXPY), Z3 SMT verification, randomized smoothing. Differentiable models: 4.7x lower Lipschitz.

## AnalogNN Compiler (New)
PyTorch -> crossbar arrays -> SPICE netlists. 3 crossbars, 17.5 nJ energy, 8512 um^2 area for MNIST MLP.

## Results Summary
| Metric | Standard | Differentiable | Improvement |
|--------|----------|---------------|-------------|
| Analog accuracy | 79.0% | 80.4% | +1.4% |
| Accuracy drop | 4.8% | 0.0% | -4.8% |
| Lipschitz bound | 171.7 | 36.2 | 4.7x better |

## Design Rules
| Parameter | Safe Range | Severity |
|-----------|-----------|----------|
| Op-amp offset | <=10mV | CRITICAL |
| Resistor mismatch | <=20% | MODERATE |
| Quantization | >=4 bits | LOW |
| Saturation rail | >=1.0V | LOW |
| Temperature | -40C to 85C | MODERATE |
