import numpy as np
import torch

class AffineCalibrator:
    def __init__(self):
        self.gains = None
        self.offsets = None

    def fit(self, y_spice: torch.Tensor, y_ideal: torch.Tensor):
        """
        Fits an affine mapping for each output neuron: y_ideal = gain * y_spice + offset.
        Uses Ordinary Least Squares.
        
        Parameters:
        - y_spice: tensor of shape (N, num_classes) containing simulated voltages
        - y_ideal: tensor of shape (N, num_classes) containing ideal mathematical logits
        """
        spice_np = y_spice.detach().cpu().numpy()
        ideal_np = y_ideal.detach().cpu().numpy()
        
        num_classes = spice_np.shape[1]
        self.gains = np.zeros(num_classes)
        self.offsets = np.zeros(num_classes)
        
        for i in range(num_classes):
            # Fit y_ideal[:, i] = a * y_spice[:, i] + b
            x = spice_np[:, i]
            y = ideal_np[:, i]
            
            # Use polyfit degree 1
            gain, offset = np.polyfit(x, y, 1)
            self.gains[i] = gain
            self.offsets[i] = offset
            
    def calibrate(self, y_spice: torch.Tensor) -> torch.Tensor:
        """
        Applies the fitted affine calibration to simulated outputs.
        """
        if self.gains is None or self.offsets is None:
            raise ValueError("Calibrator has not been fitted yet!")
            
        spice_np = y_spice.detach().cpu().numpy()
        calibrated = spice_np * self.gains + self.offsets
        return torch.tensor(calibrated, dtype=torch.float32)
