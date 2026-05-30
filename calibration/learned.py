import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class LearnedCalibrationNet(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 16):
        super(LearnedCalibrationNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

class LearnedCalibrator:
    def __init__(self, hidden_dim: int = 16, epochs: int = 100, lr: float = 0.01):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def fit(self, y_spice: torch.Tensor, y_ideal: torch.Tensor):
        """
        Trains the neural network to map simulated logits back to ideal logits.
        
        Parameters:
        - y_spice: (N, num_classes)
        - y_ideal: (N, num_classes)
        """
        num_classes = y_spice.shape[1]
        self.model = LearnedCalibrationNet(num_classes, self.hidden_dim)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        dataset = TensorDataset(y_spice, y_ideal)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
    def calibrate(self, y_spice: torch.Tensor) -> torch.Tensor:
        """
        Applies learned calibration network to simulated outputs.
        """
        if self.model is None:
            raise ValueError("Calibrator has not been fitted yet!")
            
        self.model.eval()
        with torch.no_grad():
            calibrated = self.model(y_spice)
            
        return calibrated
