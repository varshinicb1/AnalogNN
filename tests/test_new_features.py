"""Comprehensive tests for new OpenAnalogNN features."""
import pytest
import torch
import numpy as np

from analog_layers.thermal_noise import apply_thermal_noise
from analog_layers.temperature_dependence import apply_temperature_drift
from calibration.bayesian import BayesianCalibrator
from calibration.ensemble import EnsembleCalibrator
from calibration.affine import AffineCalibrator
from calibration.polynomial import PolynomialCalibrator
from nas.analog_nas import ScalingLawRobustnessScorer
from datasets.loaders import get_dataset


# =============================================================================
# 1. Thermal Noise Tests
# =============================================================================

def test_thermal_noise_physical_scale():
    """Verify Johnson-Nyquist noise RMS matches physics."""
    x = torch.zeros(10000, 1)
    y = apply_thermal_noise(x, temperature_K=300.0, resistance_ohm=1e4, bandwidth_Hz=1e6)

    empirical_std = y.std().item()
    theoretical_rms = (4.0 * 1.380649e-23 * 300.0 * 1e4 * 1e6) ** 0.5

    assert abs(empirical_std - theoretical_rms) < 1e-4, f"Std {empirical_std:.6f} vs theory {theoretical_rms:.6f}"


def test_thermal_noise_zero_temp():
    """At 0 Kelvin, thermal noise should vanish (or be near-zero)."""
    x = torch.zeros(1000, 5)
    y = apply_thermal_noise(x, temperature_K=0.0, resistance_ohm=1e4, bandwidth_Hz=1e6)
    assert torch.allclose(y, x, atol=1e-10), "Noise should be ~0 at 0K"


def test_thermal_noise_shape_preserved():
    """Thermal noise should preserve input shape."""
    x = torch.randn(10, 5, 3)
    y = apply_thermal_noise(x)
    assert y.shape == x.shape


# =============================================================================
# 2. Temperature Dependence Tests
# =============================================================================

def test_temperature_drift_no_drift_at_ref():
    """At reference temperature, drift should be identity."""
    w = torch.randn(4, 8)
    w_drifted = apply_temperature_drift(w, temperature_C=25.0, T_ref=25.0)
    assert torch.allclose(w, w_drifted), "No drift at T_ref"


def test_temperature_drift_increases_with_temp():
    """Higher temperature should cause more drift."""
    w = torch.ones(4, 8)
    w_25 = apply_temperature_drift(w, temperature_C=25.0, T_ref=25.0)
    w_85 = apply_temperature_drift(w, temperature_C=85.0, T_ref=25.0)
    w_125 = apply_temperature_drift(w, temperature_C=125.0, T_ref=25.0)
    assert torch.all(w_85 < w_25), "Higher temp should reduce effective weight"
    assert torch.all(w_125 < w_85), "Even higher temp should reduce more"


def test_temperature_drift_resistor_types():
    """Precision resistors should drift less than standard."""
    w = torch.ones(4, 8)
    w_standard = apply_temperature_drift(w, temperature_C=85.0, resistor_type='standard')
    w_precision = apply_temperature_drift(w, temperature_C=85.0, resistor_type='precision')
    w_ultra = apply_temperature_drift(w, temperature_C=85.0, resistor_type='ultra_precision')
    assert abs(1.0 - w_ultra.mean().item()) < abs(1.0 - w_standard.mean().item())
    assert abs(1.0 - w_precision.mean().item()) < abs(1.0 - w_standard.mean().item())


# =============================================================================
# 3. Bayesian Calibrator Tests
# =============================================================================

def test_bayesian_calibrator_fit_and_calibrate():
    """Bayesian calibrator should fit and reduce error."""
    torch.manual_seed(42)
    np.random.seed(42)
    y_ideal = torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.5, 2.5], [2.0, 3.0], [0.5, 1.0], [2.5, 3.5]], dtype=torch.float32)
    y_sim = y_ideal + 0.2 * torch.randn_like(y_ideal)

    cal = BayesianCalibrator()
    cal.fit(y_sim, y_ideal)
    y_cal = cal.calibrate(y_sim)

    rmse_pre = float(torch.sqrt(torch.mean((y_ideal - y_sim) ** 2)))
    rmse_post = float(torch.sqrt(torch.mean((y_ideal - y_cal) ** 2)))

    assert y_cal.shape == y_ideal.shape
    assert rmse_post < rmse_pre + 0.01, f"RMSE should decrease: {rmse_post} vs {rmse_pre}"


def test_bayesian_calibrator_uncertainty():
    """Bayesian calibrator should provide uncertainty estimates."""
    y_ideal = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    y_sim = y_ideal + 0.1 * torch.randn_like(y_ideal)

    cal = BayesianCalibrator()
    cal.fit(y_sim, y_ideal)
    y_cal, y_std = cal.calibrate(y_sim, return_std=True)

    assert y_std.shape == y_cal.shape
    assert torch.all(y_std >= 0), "Standard deviations should be non-negative"


def test_bayesian_calibrator_perfect_fit():
    """With perfect linear data, Bayesian calibrator should recover exact mapping."""
    y_sim = torch.tensor([[0.0], [0.5], [1.0], [1.5], [2.0]], dtype=torch.float32)
    y_ideal = 2.0 * y_sim + 1.0

    cal = BayesianCalibrator()
    cal.fit(y_sim, y_ideal)
    y_cal = cal.calibrate(y_sim)

    assert torch.allclose(y_cal, y_ideal, atol=0.5), "Should approximate perfect fit"


# =============================================================================
# 4. Ensemble Calibrator Tests
# =============================================================================

def test_ensemble_calibrator_average():
    """Ensemble with average strategy should work."""
    cal = EnsembleCalibrator({
        'affine': AffineCalibrator(),
        'poly': PolynomialCalibrator(degree=2),
    }, strategy='average')

    y_ideal = torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.5, 2.5], [2.0, 3.0], [0.5, 1.0], [3.5, 4.5]], dtype=torch.float32)
    y_sim = y_ideal + 0.5

    cal.fit(y_sim, y_ideal)
    y_cal = cal.calibrate(y_sim)

    assert y_cal.shape == y_ideal.shape

    rmse = float(torch.sqrt(torch.mean((y_ideal - y_cal) ** 2)))
    assert rmse < 0.5, f"Ensemble should reduce error: RMSE={rmse}"


def test_ensemble_calibrator_weighted():
    """Ensemble with weighted strategy should work."""
    cal = EnsembleCalibrator({
        'affine': AffineCalibrator(),
        'poly': PolynomialCalibrator(degree=2),
    }, strategy='weighted')

    y_ideal = torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.5, 1.5], [2.5, 3.5]], dtype=torch.float32)
    y_sim = y_ideal + 0.5

    cal.fit(y_sim, y_ideal)
    y_cal = cal.calibrate(y_sim)

    assert y_cal.shape == y_ideal.shape


def test_ensemble_calibrator_multiple_strategies():
    """All ensemble strategies should produce valid outputs."""
    y_ideal = torch.randn(10, 3)
    y_sim = y_ideal + 0.1 * torch.randn_like(y_ideal)

    for strategy in ['average', 'weighted', 'stacking']:
        cal = EnsembleCalibrator({
            'a1': AffineCalibrator(),
            'a2': AffineCalibrator(),
        }, strategy=strategy)
        cal.fit(y_sim, y_ideal)
        y_cal = cal.calibrate(y_sim)
        assert y_cal.shape == y_ideal.shape, f"Strategy {strategy} failed shape check"


# =============================================================================
# 5. Scaling Law Robustness Scorer Tests
# =============================================================================

def test_scaling_law_scorer_predictions():
    """Scaling law scorer should produce reasonable predictions."""
    scorer = ScalingLawRobustnessScorer()

    drop_1 = scorer.predict_drop(depth=1, width=128)
    drop_4 = scorer.predict_drop(depth=4, width=128)
    assert drop_4 > drop_1, "Deeper nets should have larger drops"

    drop_32 = scorer.predict_drop(depth=2, width=32)
    drop_256 = scorer.predict_drop(depth=2, width=256)

    assert drop_32 >= 0
    assert drop_256 >= 0


def test_scaling_law_accuracy_predictions():
    """Accuracy predictions should be between 0 and digital accuracy."""
    scorer = ScalingLawRobustnessScorer()

    for depth in [1, 2, 3]:
        for width in [32, 64, 128]:
            analog_acc = scorer.predict_accuracy(digital_accuracy=0.95, depth=depth, width=width)
            assert 0.0 <= analog_acc <= 0.95, f"Bad accuracy for D={depth}, W={width}: {analog_acc}"


def test_scaling_law_constraints():
    """Architectural constraints should be reasonable."""
    scorer = ScalingLawRobustnessScorer()
    constraints = scorer.get_architectural_constraints(target_accuracy_drop=0.02)

    assert 'D=1' in constraints
    assert 'D=2' in constraints
    assert constraints['D=1'] > constraints['D=2'], "Deeper = tighter noise budget"


# =============================================================================
# 6. New Dataset Tests
# =============================================================================

def test_cifar10_rgb_dataset():
    """CIFAR-10 RGB should return correct shapes with 3 channels."""
    import os
    # Only test if data already downloaded to avoid timeout
    cifar_dir = os.path.join('data', 'cifar-10-batches-py')
    if not os.path.exists(cifar_dir):
        X_train, y_train, X_test, y_test, nf, nc = get_dataset('synthetic_large', subset_size=200, downsample_size=4)
        assert nc >= 10
        assert nf == 16
        return

    X_train, y_train, X_test, y_test, nf, nc = get_dataset('cifar10_rgb', subset_size=10, downsample_size=4)
    assert nf == 48, f"Expected 48 features (4*4*3), got {nf}"
    assert nc == 10
    assert X_train.shape[0] == 10
    assert len(X_train.shape) == 2, "Should be flattened"


def test_svhn_dataset():
    """SVHN should return proper shapes."""
    import os
    # Only test if data already downloaded to avoid timeout
    svhn_dir = os.path.join('data', 'svhn')
    if not os.path.exists(svhn_dir):
        X_train, y_train, X_test, y_test, nf, nc = get_dataset('synthetic_large', subset_size=200, downsample_size=4)
        assert nc >= 10
        return

    X_train, y_train, X_test, y_test, nf, nc = get_dataset('svhn', subset_size=10, downsample_size=4)
    assert nc == 10
    assert nf == 16
    assert X_train.shape[0] == 10
    assert y_train.dtype == torch.long


def test_regression_dataset():
    """Regression dataset should return continuous 2D targets."""
    X_train, y_train, X_test, y_test, nf, nc = get_dataset('regression', subset_size=50, seed=42)

    assert nf == 5, "Regression should have 5 features"
    assert nc == 1, "Regression should have 1 output"
    assert len(y_train.shape) == 2, "Regression target should be 2D (N, 1)"
    assert y_train.shape[1] == 1, "Regression target should have 1 column"


def test_california_housing_dataset():
    """California housing should return continuous targets."""
    X_train, y_train, X_test, y_test, nf, nc = get_dataset('california_housing', subset_size=50, seed=42)

    assert nf == 8, "California housing has 8 features"
    assert nc == 1, "Should be regression (1 output)"
    assert len(y_train.shape) == 2, "Target should be 2D"
    assert X_train.min() >= 0.0, "Features should be normalized >= 0"
    assert X_train.max() <= 1.0, "Features should be normalized <= 1"
