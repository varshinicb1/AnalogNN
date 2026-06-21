"""
Experiment Pipeline Stages
===========================

Separate pipeline stages to break circular dependencies.
Each stage is a self-contained function that can be tested independently.
"""

import torch
from typing import Dict, Tuple, Optional, Callable
import copy


def train_baseline_stage(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    num_features: int,
    num_classes: int,
    config: Dict
) -> Tuple[torch.nn.Module, Dict]:
    """
    Stage 1: Train digital baseline model.
    """
    from experiments.models import DigitalMLP, train_model
    from reproduce.reproducibility import ReproducibilityManager
    
    m_cfg = config['model']
    seed = config.get('seed', 42)
    
    ReproducibilityManager.set_seed(seed)
    
    analog_config = config['analog'] if m_cfg.get('noise_aware_training', False) else None
    model = DigitalMLP(num_features, m_cfg['hidden_dims'], num_classes, analog_config=analog_config)
    
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=m_cfg['epochs'],
        lr=m_cfg['lr'],
        batch_size=m_cfg['batch_size'],
        seed=seed
    )
    
    return model, history


def simulate_analog_stage(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    X_test: torch.Tensor,
    config: Dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stage 2: Simulate analog layer with non-idealities.
    Returns (y_ideal, y_sim).
    """
    from analog_layers.analog_linear import AnalogLinear
    from spice.spice_runner import SpiceRunner
    
    # Create analog layer from digital weights
    digital_linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
    digital_linear.weight.data = weight
    if bias is not None:
        digital_linear.bias.data = bias
    
    analog_layer = AnalogLinear.from_digital(digital_linear, config=config['analog'])
    
    # Get ideal output
    with torch.no_grad():
        y_ideal = digital_linear(X_test)
    
    # Get simulated output via SPICE or fallback
    runner = SpiceRunner(config=config)
    y_sim = runner.run(weight, bias, X_test,
                      r_ref=config['circuit']['r_ref'],
                      v_ref=config['circuit']['v_ref'])
    
    return y_ideal, y_sim


def calibration_benchmark_stage(
    y_ideal: torch.Tensor,
    y_sim: torch.Tensor,
    weight_matrix: torch.Tensor,
    input_data: torch.Tensor,
    labels: torch.Tensor,
    rest_of_network_fn: Optional[Callable],
    config: Dict
) -> Dict:
    """
    Stage 3: Benchmark calibration algorithms.
    """
    from calibration.benchmark import CalibrationBenchmarker
    
    seed = config.get('seed', 42)
    
    benchmarker = CalibrationBenchmarker(
        y_ideal=y_ideal,
        y_sim=y_sim,
        weight_matrix=weight_matrix,
        input_data=input_data,
        labels=labels,
        rest_of_network_fn=rest_of_network_fn
    )
    
    return benchmarker.run_benchmark(train_ratio=0.5, seed=seed)


def parity_evaluation_stage(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    config: Dict,
    n_samples: int = 10
) -> Dict:
    """
    Stage 4: SPICE parity evaluation.
    """
    from benchmarks.spice_parity import SpiceParityBenchmarker
    
    n_parity = min(n_samples, len(X_test))
    parity_benchmarker = SpiceParityBenchmarker(config=config)
    
    return parity_benchmarker.evaluate_parity(
        weight, bias, X_test[:n_parity], y_test[:n_parity]
    )


def circuit_optimization_stage(
    weight_matrix: torch.Tensor,
    config: Dict
) -> Dict:
    """
    Stage 5: Circuit resistance optimization.
    """
    from calibration.circuit_optimization import CircuitOptimizer
    
    optimizer = CircuitOptimizer(
        weight_matrix=weight_matrix,
        area_budget=1e8,
        pelgrom_constant=1e-3,
        temperature_K=300.0,
        bandwidth_Hz=1e6,
        gbw_Hz=1e7
    )
    
    return optimizer.optimize_resistance_allocation()


def limitation_analysis_stage(
    y_ideal: torch.Tensor,
    y_sim: torch.Tensor,
    weight_matrix: torch.Tensor,
    labels: torch.Tensor,
    input_data: torch.Tensor,
    rest_of_network_fn: Optional[Callable],
    config: Dict
) -> Dict:
    """
    Stage 6: Limitation and failure mode analysis.
    """
    from validation.limitation_analysis import CalibrationLimitationAnalyzer
    
    analyzer = CalibrationLimitationAnalyzer(
        y_ideal=y_ideal,
        y_sim=y_sim,
        weight_matrix=weight_matrix,
        labels=labels,
        input_data=input_data,
        rest_of_network_fn=rest_of_network_fn
    )
    
    mismatch_cliff = analyzer.find_mismatch_cliff(
        mismatch_range=[0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]
    )
    saturation_analysis = analyzer.analyze_saturation_failures(
        input_scales=[0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
    )
    shannon_capacity = analyzer.compute_shannon_capacity(
        snr_db_range=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    )
    
    return {
        'mismatch_cliff': mismatch_cliff,
        'saturation_analysis': saturation_analysis,
        'shannon_capacity': shannon_capacity
    }


def statistical_trials_stage(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    X_test: torch.Tensor,
    y_ideal: torch.Tensor,
    y_test: torch.Tensor,
    rest_of_network_fn: Optional[Callable],
    config: Dict,
    n_trials: int = 3
) -> Dict:
    """
    Stage 7: Multi-trial statistical analysis.
    """
    from spice.spice_runner import SpiceRunner
    from calibration.hmac import HMACCalibrator
    from validation.metrics import compute_metrics
    from validation.statistical_analysis import StatisticalAnalysis
    
    trials_data = []
    seed = config.get('seed', 42)
    
    for trial in range(n_trials):
        cfg = copy.deepcopy(config)
        cfg['analog']['seed'] = seed + trial
        
        runner_t = SpiceRunner(config=cfg)
        y_sim_t = runner_t.run(weight, bias, X_test,
                               r_ref=config['circuit']['r_ref'],
                               v_ref=config['circuit']['v_ref'])
        
        cal = HMACCalibrator(weight_matrix=weight, polynomial_degree=1)
        split_t = int(len(X_test) * 0.5)
        cal.fit(y_sim_t[:split_t], y_ideal[:split_t], 
               weight_matrix=weight, input_data=X_test[:split_t])
        y_cal_t = cal.calibrate(y_sim_t[split_t:])
        
        m_trial = compute_metrics(
            y_ideal[split_t:], y_sim_t[split_t:], y_cal_t, y_test[split_t:],
            rest_of_network_fn=rest_of_network_fn
        )
        trials_data.append(m_trial)
    
    return StatisticalAnalysis.aggregate_runs(trials_data)
