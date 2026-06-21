"""
Physics-Informed Neural Calibration (PINC)
=========================================

A novel calibration method that incorporates circuit physics into the neural network
calibration process. Unlike black-box calibration, PINC uses knowledge of the circuit
topology, non-ideality models, and physical constraints to guide the calibration network.

Key Innovation:
- The calibration network receives both the SPICE output AND the circuit parameters
  (mismatch sigma, offset voltage, quantization bits, drift time constant)
- Physics-based loss terms penalize violations of known circuit constraints
- The network learns to invert the non-ideality transfer function analytically

This approach is inspired by physics-informed neural networks (PINNs) but adapted
for analog circuit calibration.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional


class PhysicsInformedCalibrator(nn.Module):
    """
    Physics-Informed Neural Calibrator for analog neural networks.
    
    The calibrator takes:
    1. SPICE/solver output voltages (y_spice)
    2. Circuit parameters (mismatch, offset, quantization, drift)
    
    And outputs calibrated predictions that match the ideal digital activations.
    
    The loss function includes:
    1. Standard MSE loss between calibrated and ideal outputs
    2. Physics consistency loss (ensures calibration respects circuit constraints)
    3. Gradient-based regularization (smooth calibration surface)
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """
        Initialize the physics-informed calibrator.
        
        Args:
            input_dim: Output dimension of the neural network layer
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Network architecture
        layers = []
        # Input: y_spice + circuit parameters (concatenated)
        # Circuit parameters: mismatch_sigma, offset_voltage, quant_bits, drift_tau
        param_dim = 4
        total_input_dim = input_dim + param_dim
        
        layers.append(nn.Linear(total_input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Learnable physics weights for combining different loss terms
        self.physics_weight = nn.Parameter(torch.tensor(1.0))
        self.gradient_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, 
                y_spice: torch.Tensor, 
                circuit_params: Dict[str, float]) -> torch.Tensor:
        """
        Forward pass of the physics-informed calibrator.
        
        Args:
            y_spice: SPICE/solver output voltages [batch_size, input_dim]
            circuit_params: Dictionary of circuit parameters
                - mismatch_sigma: Resistor mismatch standard deviation
                - offset_voltage: Op-amp input offset voltage
                - quant_bits: DAC/ADC quantization bits
                - drift_tau: Drift time constant
        
        Returns:
            y_cal: Calibrated output voltages [batch_size, input_dim]
        """
        batch_size = y_spice.shape[0]
        
        # Extract circuit parameters and broadcast to batch
        mismatch = torch.full((batch_size, 1), circuit_params['mismatch_sigma'], 
                            device=y_spice.device)
        offset = torch.full((batch_size, 1), circuit_params['offset_voltage'],
                           device=y_spice.device)
        quant = torch.full((batch_size, 1), circuit_params['quant_bits'],
                          device=y_spice.device)
        drift = torch.full((batch_size, 1), circuit_params['drift_tau'],
                         device=y_spice.device)
        
        # Concatenate SPICE output with circuit parameters
        x = torch.cat([y_spice, mismatch, offset, quant, drift], dim=1)
        
        # Apply neural network
        y_cal = self.network(x)
        
        return y_cal
    
    def physics_loss(self, 
                     y_spice: torch.Tensor, 
                     y_cal: torch.Tensor,
                     circuit_params: Dict[str, float]) -> torch.Tensor:
        """
        Compute physics-informed loss terms.
        
        This loss ensures the calibration respects known circuit constraints:
        1. Monotonicity: Calibration should preserve monotonic relationships
        2. Boundedness: Calibrated values should stay within reasonable bounds
        3. Smoothness: Small changes in input should produce small changes in output
        
        Args:
            y_spice: SPICE output
            y_cal: Calibrated output
            circuit_params: Circuit parameters
        
        Returns:
            Physics loss term
        """
        # Monotonicity loss: ensure calibration doesn't reverse order
        # For each output dimension, check that relative ordering is preserved
        batch_size = y_spice.shape[0]
        if batch_size > 1:
            # Compute pairwise differences
            diff_spice = y_spice[1:] - y_spice[:-1]
            diff_cal = y_cal[1:] - y_cal[:-1]
            
            # Penalize sign changes
            sign_product = torch.sign(diff_spice) * torch.sign(diff_cal)
            monotonicity_loss = torch.mean((sign_product < 0).float())
        else:
            monotonicity_loss = torch.tensor(0.0, device=y_spice.device)
        
        # Boundedness loss: ensure calibrated values are within reasonable range
        # Use circuit parameters to determine bounds
        mismatch = circuit_params['mismatch_sigma']
        offset = circuit_params['offset_voltage']
        
        # Expected maximum deviation due to non-idealities
        max_deviation = 3 * mismatch + 2 * offset  # 3-sigma bound
        
        # Calibrated values should not exceed this bound from SPICE output
        bound_loss = torch.mean(torch.relu(
            torch.abs(y_cal - y_spice) - max_deviation
        ))
        
        # Smoothness loss: gradient penalty
        if batch_size > 1:
            gradient = torch.abs(y_cal[1:] - y_cal[:-1])
            smoothness_loss = torch.mean(gradient)
        else:
            smoothness_loss = torch.tensor(0.0, device=y_spice.device)
        
        # Combine physics losses
        physics_loss = monotonicity_loss + bound_loss + smoothness_loss
        
        return physics_loss
    
    def compute_loss(self,
                    y_spice: torch.Tensor,
                    y_ideal: torch.Tensor,
                    circuit_params: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with physics-informed terms.
        
        Args:
            y_spice: SPICE/solver output
            y_ideal: Ideal digital output (target)
            circuit_params: Circuit parameters
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # Forward pass
        y_cal = self.forward(y_spice, circuit_params)
        
        # Standard MSE loss
        mse_loss = nn.MSELoss()(y_cal, y_ideal)
        
        # Physics-informed loss
        phys_loss = self.physics_loss(y_spice, y_cal, circuit_params)
        
        # Gradient-based smoothness loss (separate from physics)
        if y_spice.requires_grad:
            grad_output = torch.autograd.grad(
                y_cal.sum(), y_spice, create_graph=True, retain_graph=True
            )[0]
            gradient_loss = torch.mean(grad_output ** 2)
        else:
            gradient_loss = torch.tensor(0.0, device=y_spice.device)
        
        # Combine losses with learnable weights
        total_loss = (mse_loss + 
                     self.physics_weight * phys_loss + 
                     self.gradient_weight * gradient_loss)
        
        loss_dict = {
            'mse': mse_loss.item(),
            'physics': phys_loss.item(),
            'gradient': gradient_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


class PhysicsInformedCalibrationTrainer:
    """
    Trainer for the Physics-Informed Neural Calibrator.
    """
    
    def __init__(self,
                 calibrator: PhysicsInformedCalibrator,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5):
        """
        Initialize the trainer.
        
        Args:
            calibrator: PhysicsInformedCalibrator instance
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.calibrator = calibrator
        self.optimizer = torch.optim.Adam(
            calibrator.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
    def train_epoch(self,
                   y_spice: torch.Tensor,
                   y_ideal: torch.Tensor,
                   circuit_params: Dict[str, float]) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            y_spice: SPICE outputs [batch_size, input_dim]
            y_ideal: Ideal outputs [batch_size, input_dim]
            circuit_params: Circuit parameters
        
        Returns:
            Dictionary of loss values
        """
        self.calibrator.train()
        self.optimizer.zero_grad()
        
        # Compute loss
        loss, loss_dict = self.calibrator.compute_loss(
            y_spice, y_ideal, circuit_params
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss_dict
    
    def validate(self,
                y_spice: torch.Tensor,
                y_ideal: torch.Tensor,
                circuit_params: Dict[str, float]) -> Dict[str, float]:
        """
        Validate the calibrator.
        
        Args:
            y_spice: SPICE outputs
            y_ideal: Ideal outputs
            circuit_params: Circuit parameters
        
        Returns:
            Dictionary of validation metrics
        """
        self.calibrator.eval()
        
        with torch.no_grad():
            y_cal = self.calibrator.forward(y_spice, circuit_params)
            
            # Compute metrics
            mse = nn.MSELoss()(y_cal, y_ideal).item()
            mae = nn.L1Loss()(y_cal, y_ideal).item()
            
            # Compute correlation
            y_cal_flat = y_cal.flatten()
            y_ideal_flat = y_ideal.flatten()
            correlation = torch.corrcoef(
                torch.stack([y_cal_flat, y_ideal_flat])
            )[0, 1].item()
        
        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation
        }
    
    def train(self,
              y_spice_train: torch.Tensor,
              y_ideal_train: torch.Tensor,
              y_spice_val: torch.Tensor,
              y_ideal_val: torch.Tensor,
              circuit_params: Dict[str, float],
              num_epochs: int = 100,
              verbose: bool = True) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            y_spice_train: Training SPICE outputs
            y_ideal_train: Training ideal outputs
            y_spice_val: Validation SPICE outputs
            y_ideal_val: Validation ideal outputs
            circuit_params: Circuit parameters
            num_epochs: Number of training epochs
            verbose: Whether to print progress
        
        Returns:
            Dictionary of training history
        """
        history = {
            'train_loss': [],
            'val_mse': [],
            'val_correlation': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        for epoch in range(num_epochs):
            # Train
            train_loss_dict = self.train_epoch(
                y_spice_train, y_ideal_train, circuit_params
            )
            
            # Validate
            val_metrics = self.validate(
                y_spice_val, y_ideal_val, circuit_params
            )
            
            # Update learning rate
            self.scheduler.step(val_metrics['mse'])
            
            # Record history
            history['train_loss'].append(train_loss_dict['total'])
            history['val_mse'].append(val_metrics['mse'])
            history['val_correlation'].append(val_metrics['correlation'])
            
            # Early stopping
            if val_metrics['mse'] < best_val_loss:
                best_val_loss = val_metrics['mse']
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss={train_loss_dict['total']:.4f}, "
                      f"Val MSE={val_metrics['mse']:.4f}, "
                      f"Val R={val_metrics['correlation']:.4f}")
        
        return history


def create_physics_informed_calibrator(input_dim: int,
                                       hidden_dim: int = 64,
                                       num_layers: int = 3) -> PhysicsInformedCalibrator:
    """
    Factory function to create a physics-informed calibrator.
    
    Args:
        input_dim: Output dimension of the neural network layer
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
    
    Returns:
        PhysicsInformedCalibrator instance
    """
    return PhysicsInformedCalibrator(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1
    )
