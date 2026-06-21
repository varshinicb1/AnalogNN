"""
Test the SPICE autograd Function.
Verifies that gradients flow correctly through SPICE simulation.
"""

import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spice.spice_autograd import SPICEFunction


def test_spice_autograd_gradients():
    """Verify gradients flow through SPICEFunction."""
    print("=" * 60)
    print("SPICE Autograd: Gradient Flow Test")
    print("=" * 60)

    in_features, out_features = 4, 2
    batch_size = 3

    weight = torch.randn(out_features, in_features, requires_grad=True)
    bias = torch.randn(out_features, requires_grad=True)
    x = torch.randn(batch_size, in_features)

    config = {
        'use_spice': False,
        'resistor_mismatch': 0.0,
        'opamp_offset': 0.0,
        'saturation_vmax': 5.0,
        'enable_mismatch': False,
        'enable_offset': False,
        'enable_saturation': True,
    }

    y = SPICEFunction.apply(weight, bias, x, config)

    loss = y.sum()

    loss.backward()

    print(f"\nInput shape: {x.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Loss: {loss.item():.4f}")

    assert weight.grad is not None, "Weight gradient is None!"
    assert bias.grad is not None, "Bias gradient is None!"

    print(f"Weight grad shape: {weight.grad.shape}")
    print(f"Weight grad: {weight.grad}")
    print(f"Bias grad shape: {bias.grad.shape}")
    print(f"Bias grad: {bias.grad}")

    print("\n==> Gradients flow correctly through SPICEFunction!")
    return True


def test_spice_autograd_training_step():
    """Verify that SPICEFunction can be used in a training loop."""
    print("\n" + "=" * 60)
    print("SPICE Autograd: Training Step Test")
    print("=" * 60)

    in_features, out_features = 8, 3
    batch_size = 16

    torch.manual_seed(42)
    weight_true = torch.randn(out_features, in_features)
    bias_true = torch.randn(out_features)

    x = torch.randn(batch_size, in_features)
    y_true = x @ weight_true.t() + bias_true

    weight = torch.randn(out_features, in_features, requires_grad=True)
    bias = torch.randn(out_features, requires_grad=True)

    config = {
        'use_spice': False,
        'resistor_mismatch': 0.01,
        'opamp_offset': 0.002,
        'saturation_vmax': 5.0,
        'enable_mismatch': True,
        'enable_offset': True,
        'enable_saturation': True,
    }

    optimizer = torch.optim.SGD([weight, bias], lr=0.01)

    y_pred = SPICEFunction.apply(weight, bias, x, config)
    loss = torch.nn.functional.mse_loss(y_pred, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss before: {loss.item():.4f}")

    assert weight.grad is not None
    assert bias.grad is not None
    print(f"Weight grad norm: {weight.grad.norm():.4f}")
    print(f"Bias grad norm: {bias.grad.norm():.4f}")

    print("\n==> SPICEFunction works in training loop!")
    return True


def test_spice_autograd_saturation():
    """Verify gradients are masked for saturated outputs."""
    print("\n" + "=" * 60)
    print("SPICE Autograd: Saturation Gradient Test")
    print("=" * 60)

    in_features, out_features = 2, 2
    batch_size = 4

    weight = torch.ones(out_features, in_features, requires_grad=True)
    bias = torch.zeros(out_features, requires_grad=True)

    x = torch.tensor([[10.0, 10.0], [0.5, 0.5], [-10.0, -10.0], [0.1, 0.1]])

    config = {
        'use_spice': False,
        'resistor_mismatch': 0.0,
        'opamp_offset': 0.0,
        'saturation_vmax': 2.5,
        'enable_mismatch': False,
        'enable_offset': False,
        'enable_saturation': True,
    }

    y = SPICEFunction.apply(weight, bias, x, config)
    loss = y.sum()
    loss.backward()

    print(f"Output: {y}")
    print(f"Weight grad: {weight.grad}")

    expected_grad = torch.tensor([[0.6, 0.6], [0.6, 0.6]])

    assert weight.grad is not None
    assert torch.allclose(weight.grad, expected_grad, atol=1e-5), \
        f"Expected {expected_grad}, got {weight.grad}"

    print(f"Expected grad: {expected_grad}")
    print(f"Actual grad: {weight.grad}")
    print("==> Saturation gradients are correctly masked!")
    return True


if __name__ == '__main__':
    test_spice_autograd_gradients()
    test_spice_autograd_training_step()
    test_spice_autograd_saturation()
    print("\n" + "=" * 60)
    print("ALL SPICE AUTOGRAD TESTS PASSED!")
    print("=" * 60)

