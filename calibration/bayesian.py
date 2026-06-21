import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


class BayesianCalibrator:
    """
    Bayesian calibration using Gaussian Process Regression.

    Unlike deterministic calibrators (affine, polynomial), Bayesian calibration:
    1. Provides uncertainty estimates for each calibration prediction
    2. Naturally handles non-linearities through the kernel function
    3. Is sample-efficient (GP prior provides regularization)

    The kernel combines:
    - ConstantKernel: overall scale
    - RBF: smooth non-linear mapping
    - WhiteKernel: observation noise

    For a SPICE output y_spice, the calibrated output is:
        y_cal = GP_mean(y_spice) +/- 2 * GP_std(y_spice)

    The uncertainty is useful for:
    - Detection of out-of-distribution operating points
    - Confidence-weighted ensemble calibration
    - Adaptive analog-to-digital conversion resolution
    """

    def __init__(self, kernel=None, alpha=1e-10, n_restarts_optimizer=5, normalize_y=True):
        if kernel is None:
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)

        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y
        )
        self._fitted = False

    def fit(self, y_spice: torch.Tensor, y_ideal: torch.Tensor):
        spice_np = y_spice.detach().cpu().numpy()
        ideal_np = y_ideal.detach().cpu().numpy()

        num_classes = spice_np.shape[1]
        self.models = []

        for i in range(num_classes):
            X = spice_np[:, i].reshape(-1, 1)
            y = ideal_np[:, i]
            gpr = GaussianProcessRegressor(
                kernel=self.gpr.kernel,
                alpha=self.gpr.alpha,
                n_restarts_optimizer=self.gpr.n_restarts_optimizer,
                normalize_y=self.gpr.normalize_y,
                random_state=42 + i
            )
            gpr.fit(X, y)
            self.models.append(gpr)

        self._fitted = True

    def calibrate(self, y_spice: torch.Tensor, return_std: bool = False) -> torch.Tensor:
        if not self._fitted:
            raise ValueError("Calibrator has not been fitted!")

        spice_np = y_spice.detach().cpu().numpy()
        num_classes = spice_np.shape[1]
        calibrated = np.zeros_like(spice_np)
        uncertainties = np.zeros_like(spice_np)

        for i in range(num_classes):
            X = spice_np[:, i].reshape(-1, 1)
            y_mean, y_std = self.models[i].predict(X, return_std=True)
            calibrated[:, i] = y_mean
            uncertainties[:, i] = y_std

        result = torch.tensor(calibrated, dtype=torch.float32)

        if return_std:
            return result, torch.tensor(uncertainties, dtype=torch.float32)
        return result

    def get_uncertainty(self, y_spice: torch.Tensor) -> torch.Tensor:
        return self.calibrate(y_spice, return_std=True)[1]
