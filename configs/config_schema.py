"""
Configuration Schema Validation
==============================

Provides type-safe configuration validation using Pydantic to prevent
runtime errors from malformed configuration files.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class DatasetConfig(BaseModel):
    """Dataset configuration schema."""
    name: str = Field(default="iris", description="Dataset name: xor, iris, mnist, fashion")
    subset_size: int = Field(default=150, ge=1, description="Number of samples for evaluation")
    downsample_size: int = Field(default=8, ge=1, le=28, description="Downsample MNIST/Fashion to (size x size)")


class ModelConfig(BaseModel):
    """Model training configuration schema."""
    hidden_dims: List[int] = Field(default=[16, 8], description="Hidden layer dimensions")
    epochs: int = Field(default=20, ge=1, le=1000, description="Training epochs")
    lr: float = Field(default=0.01, gt=0, le=1.0, description="Learning rate")
    batch_size: int = Field(default=16, ge=1, le=1024, description="Batch size")
    noise_aware_training: bool = Field(default=False, description="Enable hardware-resilient training")


class AnalogConfig(BaseModel):
    """Analog non-ideality configuration schema."""
    noise_sigma: float = Field(default=0.05, ge=0.0, le=1.0, description="Weight noise std dev")
    quantization_bits: int = Field(default=6, ge=1, le=32, description="Quantization resolution")
    drift_tau: float = Field(default=1e5, gt=0, description="Drift time constant (seconds)")
    drift_time: float = Field(default=1e3, ge=0, description="Elapsed time for drift")
    saturation_vmax: float = Field(default=2.5, gt=0, description="Maximum output voltage")
    resistor_mismatch: float = Field(default=0.01, ge=0.0, le=1.0, description="Resistor tolerance std dev")
    opamp_offset: float = Field(default=0.002, ge=0.0, le=1.0, description="Op-amp input offset voltage")


class CircuitConfig(BaseModel):
    """Circuit simulation configuration schema."""
    r_ref: float = Field(default=10000.0, gt=0, description="Reference resistance (Ohms)")
    v_ref: float = Field(default=1.0, gt=0, description="Reference voltage (Volts)")
    backend: str = Field(default="numerical", description="Simulation backend: ngspice, numerical")
    save_netlists: bool = Field(default=True, description="Save generated netlists")

    @field_validator('backend')
    @classmethod
    def validate_backend(cls, v):
        allowed = ['ngspice', 'numerical', 'ltspice']
        if v not in allowed:
            raise ValueError(f'backend must be one of {allowed}, got {v}')
        return v


class CalibrationConfig(BaseModel):
    """Calibration configuration schema."""
    method: str = Field(default="polynomial", description="Calibration method: affine, polynomial, learned, hmac")
    poly_degree: int = Field(default=3, ge=1, le=5, description="Polynomial degree")
    learned_epochs: int = Field(default=100, ge=1, le=1000, description="Learned calibrator epochs")
    learned_lr: float = Field(default=0.01, gt=0, le=1.0, description="Learned calibrator learning rate")

    @field_validator('method')
    @classmethod
    def validate_method(cls, v):
        allowed = ['affine', 'polynomial', 'learned', 'hmac']
        if v not in allowed:
            raise ValueError(f'method must be one of {allowed}, got {v}')
        return v


class OpenAnalogNNConfig(BaseModel):
    """Main configuration schema."""
    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    analog: AnalogConfig = Field(default_factory=AnalogConfig)
    circuit: CircuitConfig = Field(default_factory=CircuitConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'OpenAnalogNNConfig':
        """Load configuration from YAML file with validation."""
        import yaml
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
