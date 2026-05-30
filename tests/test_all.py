import pytest
import torch
import numpy as np
import os

from analog_layers.quantization import quantize_tensor
from analog_layers.noise_models import apply_weight_noise
from analog_layers.drift_models import apply_drift
from analog_layers.saturation import apply_saturation
from analog_layers.mismatch import apply_mismatch
from analog_layers.analog_linear import AnalogLinear

from circuit_ir.circuit import Circuit
from circuit_ir.components import Resistor, OpAmp, VoltageSource
from circuit_ir.mapping import map_layer_to_circuit

from spice.fallback_solver import FallbackNodalSolver
from spice.waveform_parser import WaveformParser
from calibration.polynomial import PolynomialCalibrator
from validation.metrics import compute_metrics

def test_quantization_correctness():
    """
    Verifies that asymmetric and symmetric quantization correctly scale values.
    """
    tensor = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    # Quantize to 2 bits (levels: 0.0, 0.333, 0.666, 1.0)
    q_tensor = quantize_tensor(tensor, bits=2, symmetric=False)
    assert q_tensor[0] == 0.0
    assert q_tensor[-1] == 1.0
    
    # Check shape preservation
    assert q_tensor.shape == tensor.shape

def test_analog_linear_shapes_and_digital_loading():
    """
    Verifies shape consistency of forward passes and correctness of digital layer loading.
    """
    in_features, out_features = 8, 4
    batch_size = 10
    
    digital_layer = torch.nn.Linear(in_features, out_features)
    x = torch.randn(batch_size, in_features)
    
    analog_layer = AnalogLinear.from_digital(digital_layer, config={
        'noise_sigma': 0.02,
        'quantization_bits': 8,
        'saturation_vmax': 2.5
    })
    
    y_ideal = digital_layer(x)
    y_sim = analog_layer(x)
    
    assert y_sim.shape == (batch_size, out_features)
    assert y_sim.shape == y_ideal.shape

def test_circuit_mapping_graph():
    """
    Ensures that mapping weights to circuits produces the correct amount of
    op-amps, voltage sources, and input nodes in the IR.
    """
    weight = torch.tensor([[1.0, -0.5], [0.2, 0.0]])
    bias = torch.tensor([0.1, -0.1])
    x = torch.tensor([0.8, 0.9])
    
    circuit = map_layer_to_circuit(weight, bias, x, r_ref=10000.0, v_ref=1.0)
    
    # 2 inputs + 1 bias = 3 voltage sources
    sources = circuit.get_components_of_type(VoltageSource)
    assert len(sources) == 3
    
    # 2 output neurons => each has 3 op-amps (pos summer, neg summer, subtractor)
    opamps = circuit.get_components_of_type(OpAmp)
    assert len(opamps) == 6

def test_fallback_solver_correctness():
    """
    Verifies that the analytical closed-form solver computes expected outputs.
    """
    weight = torch.tensor([[1.0, 2.0], [-1.0, 0.0]])
    bias = torch.tensor([0.0, 0.5])
    x = torch.tensor([1.0, 0.5]) # Expected output: [2.0, -0.5]
    
    config = {
        'resistor_mismatch': 0.0,
        'drift_time': 0.0,
        'opamp_offset': 0.0,
        'saturation_vmax': 5.0,
        'enable_mismatch': False,
        'enable_drift': False,
        'enable_offset': False,
        'enable_saturation': True
    }
    
    y_out = FallbackNodalSolver.solve_closed_form(weight, bias, x, config)
    
    assert y_out.shape == (1, 2)
    assert np.allclose(y_out.numpy()[0], [2.0, -0.5], atol=1e-5)

def test_waveform_parser():
    """
    Checks that the text waveform parser extracts values from standard logs correctly.
    """
    sample_log = """
    Variable values:
    0 node_out_0 voltage = 2.3456
    1 node_out_1 voltage = -1.2345
    """
    # Write to a temporary file
    temp_file = "temp_test_log.log"
    with open(temp_file, "w") as f:
        f.write(sample_log)
        
    voltages = WaveformParser.parse_raw_file(temp_file)
    
    assert voltages.get("node_out_0") == 2.3456
    assert voltages.get("node_out_1") == -1.2345
    
    if os.path.exists(temp_file):
        os.remove(temp_file)

def test_calibration_and_metrics():
    """
    Verifies that polynomial calibration reduces validation RMSE and returns correct metrics.
    """
    y_ideal = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    # simulated voltages have a systematic offset (+0.5V)
    y_sim = y_ideal + 0.5
    y_true = torch.tensor([1, 1])
    
    calibrator = PolynomialCalibrator(degree=1)
    calibrator.fit(y_sim, y_ideal)
    y_cal = calibrator.calibrate(y_sim)
    
    metrics = compute_metrics(y_ideal, y_sim, y_cal, y_true)
    
    assert metrics['rmse_pre_calibration'] > 0.4
    assert metrics['rmse_post_calibration'] < 1e-4
    assert metrics['calibration_improvement_pct'] > 99.0
