"""
MLflow Experiment Tracker
=========================

Replaces manual JSON-based tracking with MLflow for proper
experiment management, artifact logging, and comparison.
"""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import torch
import numpy as np
from typing import Dict, Any, Optional
import os


class MLflowTracker:
    """
    MLflow-based experiment tracking for OpenAnalogNN.
    """

    def __init__(self, experiment_name: str = "OpenAnalogNN", tracking_uri: Optional[str] = None):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local ./mlruns)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.run = None

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new MLflow run."""
        self.run = mlflow.start_run(run_name=run_name)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        if self.run:
            mlflow.end_run(status=status)
            self.run = None

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration parameters."""
        # Flatten nested dict for MLflow logging
        flat_config = self._flatten_dict(config)
        mlflow.log_params(flat_config)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: torch.nn.Module, model_name: str = "digital_baseline") -> None:
        """Log PyTorch model."""
        mlflow.pytorch.log_model(model, model_name)

    def log_sklearn_model(self, model, model_name: str = "calibrator") -> None:
        """Log scikit-learn model."""
        mlflow.sklearn.log_model(model, model_name)

    def log_artifact(self, artifact_path: str, artifact_type: str = "file") -> None:
        """Log artifact (figure, netlist, etc.)."""
        if os.path.exists(artifact_path):
            mlflow.log_artifact(artifact_path, artifact_path=artifact_type)

    def log_figure(self, figure, artifact_file: str) -> None:
        """Log matplotlib figure as artifact."""
        mlflow.log_figure(figure, artifact_file)

    def log_dataset(self, dataset_info: Dict[str, Any]) -> None:
        """Log dataset information."""
        mlflow.log_params({f"dataset_{k}": v for k, v in dataset_info.items()})

    def log_calibration_results(self, results: Dict[str, Any]) -> None:
        """Log calibration benchmark results."""
        for method_name, metrics in results.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{method_name}_{metric_name}", value)

    def log_spice_results(self, results: Dict[str, Any]) -> None:
        """Log SPICE simulation results."""
        for key, value in results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"spice_{key}", value)

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
        """Flatten nested dictionary for MLflow logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (int, float, str, bool)):
                items.append((new_key, v))
            # Skip complex types that MLflow can't handle
        return dict(items)

    @staticmethod
    def compare_runs(experiment_name: str, metric: str = "rmse_post_calibration") -> None:
        """Compare runs in an experiment."""
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        runs = client.search_runs(experiment.experiment_id, order_by=[f"metrics.{metric} ASC"])
        
        print(f"\n=== Run Comparison for {metric} ===")
        for run in runs[:10]:  # Top 10 runs
            print(f"Run ID: {run.info.run_id}")
            print(f"  {metric}: {run.data.metrics.get(metric, 'N/A')}")
            print(f"  Status: {run.info.status}")
            print()

    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        status = "FAILED" if exc_type else "FINISHED"
        self.end_run(status=status)
