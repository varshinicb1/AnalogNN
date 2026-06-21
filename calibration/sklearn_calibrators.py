"""
Scikit-learn Based Calibrators
==============================

Replaces manual calibration implementations with scikit-learn's
calibration modules (CalibratedClassifierCV, IsotonicRegression).

Also includes the novel Physics-Informed Neural Calibration (PINC) method
that incorporates circuit physics into the calibration process.
"""

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from typing import Optional, Dict, Tuple
from .physics_informed import PhysicsInformedCalibrator, PhysicsInformedCalibrationTrainer


class SklearnAffineCalibrator:
    """
    Affine calibration using scikit-learn LinearRegression.
    Equivalent to y_cal = a * y_sim + b
    """

    def __init__(self):
        self.model = LinearRegression(fit_intercept=True)

    def fit(self, y_sim: torch.Tensor, y_ideal: torch.Tensor) -> None:
        """Fit affine transformation."""
        y_sim_np = y_sim.detach().cpu().numpy().reshape(-1, 1)
        y_ideal_np = y_ideal.detach().cpu().numpy().reshape(-1, 1)
        self.model.fit(y_sim_np, y_ideal_np)

    def calibrate(self, y_sim: torch.Tensor) -> torch.Tensor:
        """Apply affine calibration."""
        y_sim_np = y_sim.detach().cpu().numpy().reshape(-1, 1)
        y_cal_np = self.model.predict(y_sim_np)
        return torch.tensor(y_cal_np, dtype=torch.float32, device=y_sim.device)


class SklearnPolynomialCalibrator:
    """
    Polynomial calibration using scikit-learn Pipeline.
    Equivalent to y_cal = sum_{k=0}^d a_k * y_sim^k
    """

    def __init__(self, degree: int = 3):
        self.degree = degree
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression(fit_intercept=True))
        ])

    def fit(self, y_sim: torch.Tensor, y_ideal: torch.Tensor) -> None:
        """Fit polynomial transformation."""
        y_sim_np = y_sim.detach().cpu().numpy().reshape(-1, 1)
        y_ideal_np = y_ideal.detach().cpu().numpy().reshape(-1, 1)
        self.model.fit(y_sim_np, y_ideal_np)

    def calibrate(self, y_sim: torch.Tensor) -> torch.Tensor:
        """Apply polynomial calibration."""
        y_sim_np = y_sim.detach().cpu().numpy().reshape(-1, 1)
        y_cal_np = self.model.predict(y_sim_np)
        return torch.tensor(y_cal_np, dtype=torch.float32, device=y_sim.device)


class SklearnIsotonicCalibrator:
    """
    Isotonic calibration using scikit-learn IsotonicRegression.
    Monotonic non-parametric calibration.
    """

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')

    def fit(self, y_sim: torch.Tensor, y_ideal: torch.Tensor) -> None:
        """Fit isotonic transformation."""
        y_sim_np = y_sim.detach().cpu().numpy().flatten()
        y_ideal_np = y_ideal.detach().cpu().numpy().flatten()
        self.model.fit(y_sim_np, y_ideal_np)

    def calibrate(self, y_sim: torch.Tensor) -> torch.Tensor:
        """Apply isotonic calibration."""
        y_sim_np = y_sim.detach().cpu().numpy().flatten()
        y_cal_np = self.model.predict(y_sim_np)
        return torch.tensor(y_cal_np, dtype=torch.float32, device=y_sim.device)


class SklearnWeightedCalibrator:
    """
    Weighted Least Squares calibration using scikit-learn.
    Similar to HMAC but using sklearn's sample_weight parameter.
    """

    def __init__(self, degree: int = 1):
        self.degree = degree
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression(fit_intercept=True))
        ])

    def fit(self, y_sim: torch.Tensor, y_ideal: torch.Tensor,
            sample_weights: Optional[np.ndarray] = None) -> None:
        """
        Fit weighted polynomial transformation.
        
        Args:
            y_sim: Simulated outputs
            y_ideal: Ideal outputs
            sample_weights: Weights for each sample (e.g., inverse variance)
        """
        y_sim_np = y_sim.detach().cpu().numpy().reshape(-1, 1)
        y_ideal_np = y_ideal.detach().cpu().numpy().reshape(-1, 1)
        
        if sample_weights is None:
            sample_weights = np.ones(len(y_sim_np))
        
        self.model.fit(y_sim_np, y_ideal_np, linear__sample_weight=sample_weights)

    def calibrate(self, y_sim: torch.Tensor) -> torch.Tensor:
        """Apply weighted calibration."""
        y_sim_np = y_sim.detach().cpu().numpy().reshape(-1, 1)
        y_cal_np = self.model.predict(y_sim_np)
        return torch.tensor(y_cal_np, dtype=torch.float32, device=y_sim.device)
