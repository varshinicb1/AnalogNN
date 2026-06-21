---
title: OpenAnalogNN
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# OpenAnalogNN Interactive Demo

[![Open in HF Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces)

Differentiable analog neural network simulation, calibration, and SPICE validation.

## Features

- **7 datasets**: XOR, Iris, MNIST, Fashion-MNIST, CIFAR-10, SVHN, California Housing
- **6 architectures**: Depth 1-6, Width 8-256
- **6 non-idealities**: Noise, Mismatch, Offset, Quantization, Saturation, Thermal Noise
- **6 calibrators**: Affine, Polynomial, Bayesian GP, Ensemble, HMAC, Learned MLP
- **Scaling Law insights**: Real-time accuracy drop predictions

## Quick Start

```bash
pip install open-analog-nn
streamlit run app.py
```

## How to Deploy on HuggingFace Spaces

1. Create a new Space at https://huggingface.co/new-space
2. Choose "Streamlit" SDK
3. Upload all files from this `app_deploy/` directory
4. The Space will auto-install dependencies from `requirements.txt`

## Results

| Metric | Value |
|--------|-------|
| SOTA Analog Accuracy | 77.87% (Fashion-MNIST) |
| Scaling Law R² | 0.9385 |
| SPICE Match | 42/42 at 1e-4 |
| Calibrators | 6 methods |
| Energy Efficiency | 8980 acc/µJ (7nm) |
