"""
Tests for new novel features:
- Analog-Aware NAS
- Adversarial Training
- Energy Modeling
- HuggingFace Integration
"""

import torch
import pytest
import numpy as np


def test_analog_nas_search():
    """Test Analog-Aware NAS."""
    from nas.analog_nas import AnalogNASSearch
    
    # Small dataset
    X_train = torch.randn(50, 10)
    y_train = torch.randint(0, 3, (50,))
    X_test = torch.randn(20, 10)
    y_test = torch.randint(0, 3, (20,))
    
    config = {'resistor_mismatch': 0.01, 'noise_sigma': 0.01}
    
    nas = AnalogNASSearch(
        input_dim=10,
        output_dim=3,
        analog_config=config,
        search_space={
            'hidden_dims': [[32], [64]],
        }
    )
    
    # Generate candidates
    candidates = nas.generate_candidates(max_candidates=2)
    assert len(candidates) == 2
    
    # Run search (very small)
    best = nas.search(X_train, y_train, X_test, y_test, max_candidates=2, epochs=5)
    
    assert best.accuracy > 0
    assert best.analog_robustness_score > 0
    assert best.energy_efficiency > 0


def test_adversarial_training():
    """Test adversarial training."""
    from training.adversarial_training import AnalogAdversarialTrainer
    from experiments.models import DigitalMLP
    
    # Create model
    model = DigitalMLP(input_dim=10, hidden_dims=[32], output_dim=3)
    
    config = {'resistor_mismatch': 0.01, 'noise_sigma': 0.01}
    
    trainer = AnalogAdversarialTrainer(
        model=model,
        analog_config=config,
        epsilon=0.1,
        attack_steps=3,
        attack_lr=0.01
    )
    
    # Create data
    X = torch.randn(20, 10)
    y = torch.randint(0, 3, (20,))
    
    # Test adversarial attack
    X_adv, attack_info = trainer.adversarial_attack(X, y)
    assert X_adv.shape == X.shape
    assert 'attack_success_rate' in attack_info
    
    # Test training step
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = trainer.train_step(X, y, optimizer, lambda_adv=0.5)
    
    assert 'loss' in metrics
    assert 'accuracy_clean' in metrics
    assert 'accuracy_robust' in metrics


def test_energy_model():
    """Test energy efficiency modeling."""
    from energy.analog_energy_model import AnalogEnergyModel
    
    # Create energy model
    energy_model = AnalogEnergyModel(tech_node='28nm')
    
    # Test resistor network energy
    weight = torch.randn(64, 32)
    x = torch.randn(10, 32)
    
    energy = energy_model.resistor_network_energy(weight, x)
    
    assert energy.total_energy_J > 0
    assert energy.static_power_W > 0
    assert energy.tops_per_watt > 0
    
    # Test DAC/ADC energy
    dac_e = energy_model.dac_energy(n_bits=8, n_channels=32)
    assert dac_e > 0
    
    adc_e = energy_model.adc_energy(n_bits=8, n_channels=64)
    assert adc_e > 0
    
    # Test comparison with digital
    comparison = energy_model.compare_with_digital(weight, x)
    assert 'efficiency_vs_gpu' in comparison
    assert comparison['efficiency_vs_gpu'] > 0


def test_energy_model_full_layer():
    """Test full layer energy analysis."""
    from energy.analog_energy_model import AnalogEnergyModel
    
    energy_model = AnalogEnergyModel(tech_node='28nm')
    
    weight = torch.randn(64, 32)
    x = torch.randn(10, 32)
    
    energy = energy_model.full_layer_energy(
        weight, x,
        r_ref=10e3,
        v_ref=1.0,
        dac_bits=8,
        adc_bits=8
    )
    
    assert 'total_energy_J' in energy
    assert 'dac_energy_J' in energy
    assert 'adc_energy_J' in energy
    assert energy['total_energy_J'] > 0


def test_huggingface_wrapper():
    """Test HuggingFace wrapper (if transformers available)."""
    pytest.skip("HuggingFace tests require model download - skipping by default")
    try:
        from huggingface.analog_transformers import HuggingFaceAnalogWrapper
        from transformers import GPT2LMHeadModel
    except ImportError:
        pytest.skip("transformers not installed")
    
    # Load small model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    config = {'resistor_mismatch': 0.01, 'noise_sigma': 0.01}
    
    wrapper = HuggingFaceAnalogWrapper(model, config)
    
    # Convert to analog
    analog_model = wrapper.convert_to_analog()
    
    # Check conversion
    stats = wrapper.get_layer_statistics()
    assert stats['n_layers'] > 0
    assert stats['total_parameters'] > 0


def test_analog_llm_benchmark():
    """Test AnalogLLMBenchmark (if transformers available)."""
    pytest.skip("HuggingFace tests require model download - skipping by default")
    try:
        from huggingface.analog_transformers import AnalogLLMBenchmark
    except ImportError:
        pytest.skip("transformers not installed")
    
    benchmark = AnalogLLMBenchmark(
        model_name='gpt2',
        analog_config={'resistor_mismatch': 0.01}
    )
    
    # Load models
    digital_model, analog_model = benchmark.load_model()
    
    assert digital_model is not None
    assert analog_model is not None
    
    # Benchmark inference (very small)
    results = benchmark.benchmark_inference(
        prompt="Hello",
        max_new_tokens=5,
        n_runs=2
    )
    
    assert 'digital_avg_latency' in results
    assert 'analog_avg_latency' in results


def test_comprehensive_benchmark():
    """Test comprehensive benchmark suite."""
    from benchmarks.comprehensive_benchmark import ComprehensiveBenchmark
    
    benchmark = ComprehensiveBenchmark()
    
    # Run in quick mode
    results = benchmark.run_all(
        skip_huggingface=True,
        skip_nas=True,
        quick_mode=True
    )
    
    assert 'core_layers' in results
    assert 'circuit_sim' in results
    assert 'calibration' in results
    assert 'energy' in results


if __name__ == "__main__":
    # Run tests
    print("Testing Analog NAS...")
    test_analog_nas_search()
    print("PASSED")
    
    print("\nTesting Adversarial Training...")
    test_adversarial_training()
    print("PASSED")
    
    print("\nTesting Energy Model...")
    test_energy_model()
    test_energy_model_full_layer()
    print("PASSED")
    
    print("\nTesting Comprehensive Benchmark...")
    test_comprehensive_benchmark()
    print("PASSED")
    
    print("\nAll tests passed!")
