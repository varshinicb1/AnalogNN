"""
Simple Experiment Tracker
==========================

Lightweight JSON-based experiment tracking without external dependencies.
Logs configuration, metrics, and artifacts for reproducibility.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
import hashlib


class ExperimentTracker:
    """
    Tracks experiments with configuration, metrics, and artifacts.
    Stores results in JSON format for easy analysis and reproducibility.
    """

    def __init__(self, experiment_name: str, output_dir: str = "./experiments"):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique experiment ID from timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{timestamp}"
        
        # Create experiment directory
        self.experiment_dir = os.path.join(output_dir, self.experiment_id)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize experiment data
        self.data = {
            'experiment_id': self.experiment_id,
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'status': 'running',
            'config': {},
            'metrics': {},
            'artifacts': []
        }
        
        # Config hash for deduplication
        self.config_hash = None

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        self.data['config'] = config
        # Compute hash for config deduplication
        config_str = json.dumps(config, sort_keys=True)
        self.config_hash = hashlib.md5(config_str.encode()).hexdigest()
        self.data['config_hash'] = self.config_hash
        self._save()

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics at a given step."""
        if step is not None:
            if 'metrics_by_step' not in self.data:
                self.data['metrics_by_step'] = {}
            self.data['metrics_by_step'][step] = metrics
        else:
            self.data['metrics'].update(metrics)
        self._save()

    def log_artifact(self, artifact_path: str, artifact_type: str = "file") -> None:
        """Log an artifact (figure, netlist, etc.)."""
        if os.path.exists(artifact_path):
            # Copy artifact to experiment directory
            import shutil
            artifact_name = os.path.basename(artifact_path)
            dest_path = os.path.join(self.experiment_dir, artifact_name)
            shutil.copy2(artifact_path, dest_path)
            
            self.data['artifacts'].append({
                'name': artifact_name,
                'type': artifact_type,
                'path': dest_path,
                'original_path': artifact_path
            })
            self._save()

    def log_parameter(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        if 'parameters' not in self.data:
            self.data['parameters'] = {}
        self.data['parameters'][key] = value
        self._save()

    def finish(self, status: str = "completed") -> None:
        """Mark experiment as finished."""
        self.data['end_time'] = datetime.now().isoformat()
        self.data['status'] = status
        
        # Compute duration
        start = datetime.fromisoformat(self.data['start_time'])
        end = datetime.fromisoformat(self.data['end_time'])
        duration = (end - start).total_seconds()
        self.data['duration_seconds'] = duration
        
        self._save()

    def _save(self) -> None:
        """Save experiment data to JSON file."""
        output_path = os.path.join(self.experiment_dir, 'experiment.json')
        with open(output_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        return {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'status': self.data['status'],
            'duration_seconds': self.data.get('duration_seconds'),
            'metrics': self.data['metrics'],
            'config_hash': self.config_hash
        }

    @staticmethod
    def list_experiments(output_dir: str = "./experiments") -> list:
        """List all experiments in the output directory."""
        if not os.path.exists(output_dir):
            return []
        
        experiments = []
        for exp_name in os.listdir(output_dir):
            exp_path = os.path.join(output_dir, exp_name)
            if os.path.isdir(exp_path):
                json_path = os.path.join(exp_path, 'experiment.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    experiments.append(data)
        
        # Sort by start time (newest first)
        experiments.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        return experiments

    @staticmethod
    def load_experiment(experiment_id: str, output_dir: str = "./experiments") -> Optional[Dict]:
        """Load experiment data by ID."""
        exp_path = os.path.join(output_dir, experiment_id)
        json_path = os.path.join(exp_path, 'experiment.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        return None
