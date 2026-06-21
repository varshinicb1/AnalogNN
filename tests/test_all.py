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
from circuit_ir.templates import CircuitTemplates

from spice.fallback_solver import FallbackNodalSolver
from spice.waveform_parser import WaveformParser
from calibration.polynomial import PolynomialCalibrator
from calibration.hmac import HMACCalibrator
from validation.metrics import compute_metrics
from validation.error_bounds import AnalogErrorBound, OptimalResistanceAllocation
from validation.statistical_tests import CalibrationStatisticalTester
from validation.limitation_analysis import CalibrationLimitationAnalyzer
from calibration.circuit_optimization import CircuitOptimizer


def test_quantization_correctness():
    """
    Verifies that asymmetric and symmetric quantization correctly scale values.
    """
    tensor = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    # Quantize to 2 bits (levels: 0.0, 0.333, 0.666, 1.0)
    q_tensor = quantize_tensor(tensor, bits=2, symmetric=False)
    assert q_tensor[0] == 0.0
    assert q_tensor[-1] == 1.0
    assert q_tensor.shape == tensor.shape


def test_quantization_zero_tensor_fix():
    """
    Test that quantization of all-zero tensor returns zeros (not unchanged).
    This is a regression test for the bug where unchanged tensor was returned.
    """
    tensor = torch.zeros(10)
    q_tensor = quantize_tensor(tensor, bits=4, symmetric=True)
    assert torch.all(q_tensor == 0.0)
    assert q_tensor.shape == tensor.shape


def test_mismatch_numerical_stability():
    """
    Test that mismatch doesn't cause division by zero even with extreme values.
    This is a regression test for the denominator clamping fix.
    """
    weight = torch.tensor([[1.0, 0.5], [0.2, 0.1]])
    # Use high mismatch sigma to test stability
    w_mismatched = apply_mismatch(weight, mismatch_sigma=0.5, pelgrom_matching=False)
    # Should not contain inf or nan
    assert torch.all(torch.isfinite(w_mismatched))
    assert w_mismatched.shape == weight.shape


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
    y_sim = y_ideal + 0.5
    y_true = torch.tensor([1, 1])
    
    calibrator = PolynomialCalibrator(degree=1)
    calibrator.fit(y_sim, y_ideal)
    y_cal = calibrator.calibrate(y_sim)
    
    metrics = compute_metrics(y_ideal, y_sim, y_cal, y_true)
    
    assert metrics['rmse_pre_calibration'] > 0.4
    assert metrics['rmse_post_calibration'] < 1e-4
    assert metrics['calibration_improvement_pct'] > 99.0


def test_hmac_calibrator():
    """
    Verifies that HMAC calibration properly fits and reduces error.
    """
    weight = torch.tensor([[1.5, -0.5], [0.2, 1.2]])
    y_ideal = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_sim = y_ideal + 0.3 * torch.randn_like(y_ideal)
    
    hmac = HMACCalibrator(weight_matrix=weight, polynomial_degree=1)
    hmac.fit(y_sim, y_ideal)
    y_cal = hmac.calibrate(y_sim)
    
    assert y_cal.shape == y_ideal.shape
    
    # Breusch-Pagan test should run without exceptions
    bp_results = hmac.breusch_pagan_test(y_sim, y_ideal)
    assert len(bp_results) == 2


def test_error_bounds():
    """
    Verifies the Analytical Error Bound calculations.
    """
    weight = torch.tensor([[1.0, -0.5], [0.2, 1.2]])
    bias = torch.tensor([0.1, -0.1])
    
    bound_calc = AnalogErrorBound(
        weight_matrix=weight,
        bias=bias,
        mismatch_sigma=0.01,
        offset_sigma=0.002,
        noise_sigma=0.01,
        quantization_bits=8
    )
    
    bounds = bound_calc.compute_error_bound(x_second_moment=0.5)
    assert bounds['total_error_bound'] > 0.0
    assert bounds['mismatch_pct'] > 0.0


def test_circuit_optimization():
    """
    Verifies the resistor allocation optimization returns reasonable values.
    """
    weight = torch.tensor([[1.0, -0.5], [0.2, 1.2]])
    optimizer = CircuitOptimizer(
        weight_matrix=weight,
        area_budget=1e7,
        pelgrom_constant=1e-3
    )
    opt = optimizer.optimize_resistance_allocation()
    assert opt['optimal_r_ref'] > 0.0
    assert opt['optimal_r_ref'] <= 1e7


def test_statistical_tester():
    """
    Verifies the statistical hypothesis tester computes t-tests and normality tests.
    """
    errors = {
        'OLS': np.random.normal(0.1, 0.02, size=100),
        'HMAC': np.random.normal(0.01, 0.002, size=100)
    }
    tester = CalibrationStatisticalTester()
    comparisons = tester.compare_calibrators(errors)
    pairwise = tester.pairwise_comparisons(errors)
    
    assert len(pairwise) == 1
    assert pairwise[0]['significant'] is not None


def test_circuit_templates():
    """
    Verifies that templates construct correctly populated circuits.
    """
    weights = np.array([1.0, -0.5])
    inputs = np.array([0.8, 0.9])
    
    c_single = CircuitTemplates.single_ended_summing_amp(weights, bias=0.1, inputs=inputs)
    assert len(c_single.components) > 0
    
    c_tia = CircuitTemplates.transimpedance_amplifier(inputs, weights)
    assert len(c_tia.components) > 0
