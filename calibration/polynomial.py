import numpy as np
import torch

class PolynomialCalibrator:
    def __init__(self, degree: int = 3):
        self.degree = degree
        self.coefficients = {} # Maps class index to polynomial coefficients

    def fit(self, y_spice: torch.Tensor, y_ideal: torch.Tensor):
        """
        Fits a degree-d polynomial for each output neuron:
        y_ideal = sum_k c_k * y_spice^k
        
        Parameters:
        - y_spice: (N, num_classes)
        - y_ideal: (N, num_classes)
        """
        spice_np = y_spice.detach().cpu().numpy()
        ideal_np = y_ideal.detach().cpu().numpy()
        
        num_classes = spice_np.shape[1]
        for i in range(num_classes):
            x = spice_np[:, i]
            y = ideal_np[:, i]
            
            # Fit polynomial coefficients
            coeffs = np.polyfit(x, y, self.degree)
            self.coefficients[i] = coeffs
            
    def calibrate(self, y_spice: torch.Tensor) -> torch.Tensor:
        """
        Applies polynomial mapping to simulated outputs.
        """
        if not self.coefficients:
            raise ValueError("Calibrator has not been fitted yet!")
            
        spice_np = y_spice.detach().cpu().numpy()
        calibrated = np.zeros_like(spice_np)
        
        num_classes = spice_np.shape[1]
        for i in range(num_classes):
            coeffs = self.coefficients[i]
            calibrated[:, i] = np.polyval(coeffs, spice_np[:, i])
            
        return torch.tensor(calibrated, dtype=torch.float32)
