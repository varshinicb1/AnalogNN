"""
HuggingFace Transformer Integration
====================================

Scale analog simulation to real-world transformer models (GPT-2, BERT, LLaMA).

This enables:
- Running HuggingFace models through analog simulation
- Evaluating analog non-ideality impact on language models
- Energy efficiency analysis for LLMs on analog hardware
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


class HuggingFaceAnalogWrapper:
    """
    Wraps HuggingFace transformer models with analog non-ideality simulation.
    
    Replaces all nn.Linear layers with AnalogLinear layers while
    preserving the model architecture and pretrained weights.
    """
    
    def __init__(self,
                 model,
                 analog_config: Dict,
                 layers_to_convert: Optional[List[str]] = None):
        """
        Args:
            model: HuggingFace model (e.g., GPT2LMHeadModel, BertModel)
            analog_config: Configuration for analog non-idealities
            layers_to_convert: List of layer name patterns to convert
                             (None = convert all linear layers)
        """
        self.model = model
        self.analog_config = analog_config
        self.layers_to_convert = layers_to_convert
        
        # Track converted layers
        self.converted_layers = {}
        
    def convert_to_analog(self) -> nn.Module:
        """
        Convert all nn.Linear layers to AnalogLinear.
        
        Returns the modified model with analog layers.
        """
        from analog_layers.analog_linear import AnalogLinear
        
        # Find all linear layers
        linear_layers = self._find_linear_layers()
        
        print(f"Found {len(linear_layers)} linear layers to convert")
        
        # Convert each layer
        for name, module in linear_layers.items():
            # Check if should convert
            if self.layers_to_convert is not None:
                if not any(pattern in name for pattern in self.layers_to_convert):
                    continue
            
            # Create analog version
            analog_layer = AnalogLinear.from_digital(module, config=self.analog_config)
            
            # Replace in model
            self._replace_module(self.model, name, analog_layer)
            self.converted_layers[name] = analog_layer
        
        print(f"Converted {len(self.converted_layers)} layers to analog")
        
        return self.model
    
    def _find_linear_layers(self) -> Dict[str, nn.Linear]:
        """Find all nn.Linear layers in the model."""
        linear_layers = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers[name] = module
        
        return linear_layers
    
    def _replace_module(self, model: nn.Module, name: str, new_module: nn.Module):
        """Replace a module by name."""
        parts = name.split('.')
        parent = model
        
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        setattr(parent, parts[-1], new_module)
    
    def forward(self, *args, **kwargs):
        """Forward pass through the analog model."""
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generate text using the analog model."""
        return self.model.generate(*args, **kwargs)
    
    def get_layer_statistics(self) -> Dict:
        """Get statistics about converted layers."""
        stats = {
            'n_layers': len(self.converted_layers),
            'total_parameters': 0,
            'layer_names': list(self.converted_layers.keys())
        }
        
        for name, layer in self.converted_layers.items():
            stats['total_parameters'] += layer.weight.numel()
            if layer.bias is not None:
                stats['total_parameters'] += layer.bias.numel()
        
        return stats


class AnalogLLMBenchmark:
    """
    Benchmark analog LLM performance against digital baseline.
    
    Metrics:
    - Perplexity (language modeling quality)
    - Inference latency
    - Energy consumption
    - Accuracy degradation from non-idealities
    """
    
    def __init__(self,
                 model_name: str = 'gpt2',
                 analog_config: Dict = None):
        """
        Args:
            model_name: HuggingFace model name (e.g., 'gpt2', 'bert-base-uncased')
            analog_config: Analog non-ideality configuration
        """
        self.model_name = model_name
        self.analog_config = analog_config or {
            'resistor_mismatch': 0.01,
            'noise_sigma': 0.01,
            'opamp_offset': 0.002,
            'quantization_bits': 8,
            'saturation_vmax': 2.5
        }
        
        self.digital_model = None
        self.analog_model = None
        
    def load_model(self):
        """Load HuggingFace model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers library required: pip install transformers")
        
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.digital_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Create analog version
        wrapper = HuggingFaceAnalogWrapper(self.digital_model, self.analog_config)
        self.analog_model = wrapper.convert_to_analog()
        
        return self.digital_model, self.analog_model
    
    def evaluate_perplexity(self,
                           text: str,
                           stride: int = 512) -> Dict[str, float]:
        """
        Evaluate perplexity on given text.
        
        Returns perplexity for both digital and analog models.
        """
        if self.digital_model is None:
            self.load_model()
        
        # Tokenize
        encodings = self.tokenizer(text, return_tensors='pt')
        input_ids = encodings.input_ids
        
        max_length = self.digital_model.config.n_positions if hasattr(self.digital_model.config, 'n_positions') else 1024
        
        # Evaluate digital model
        digital_ppl = self._compute_perplexity(self.digital_model, input_ids, max_length, stride)
        
        # Evaluate analog model
        analog_ppl = self._compute_perplexity(self.analog_model, input_ids, max_length, stride)
        
        return {
            'digital_perplexity': digital_ppl,
            'analog_perplexity': analog_ppl,
            'perplexity_degradation': (analog_ppl - digital_ppl) / digital_ppl * 100
        }
    
    def _compute_perplexity(self,
                           model: nn.Module,
                           input_ids: torch.Tensor,
                           max_length: int,
                           stride: int) -> float:
        """Compute perplexity for a model."""
        model.eval()
        
        nlls = []  # Negative log likelihoods
        seq_len = input_ids.size(1)
        
        with torch.no_grad():
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                
                input_chunk = input_ids[:, begin_loc:end_loc]
                
                # Forward pass
                outputs = model(input_chunk, labels=input_chunk)
                loss = outputs.loss
                
                nlls.append(loss.item())
                
                if end_loc == seq_len:
                    break
        
        # Compute perplexity
        avg_nll = np.mean(nlls)
        ppl = np.exp(avg_nll)
        
        return ppl
    
    def benchmark_inference(self,
                           prompt: str = "Hello, my name is",
                           max_new_tokens: int = 50,
                           n_runs: int = 10) -> Dict:
        """
        Benchmark inference latency and quality.
        
        Returns latency, throughput, and generated text for both models.
        """
        import time
        
        if self.digital_model is None:
            self.load_model()
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        results = {
            'digital': {'latencies': [], 'outputs': []},
            'analog': {'latencies': [], 'outputs': []}
        }
        
        # Benchmark digital
        for _ in range(n_runs):
            start = time.time()
            output = self.digital_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            latency = time.time() - start
            
            results['digital']['latencies'].append(latency)
            results['digital']['outputs'].append(
                self.tokenizer.decode(output[0], skip_special_tokens=True)
            )
        
        # Benchmark analog
        for _ in range(n_runs):
            start = time.time()
            output = self.analog_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            latency = time.time() - start
            
            results['analog']['latencies'].append(latency)
            results['analog']['outputs'].append(
                self.tokenizer.decode(output[0], skip_special_tokens=True)
            )
        
        # Compute statistics
        return {
            'digital_avg_latency': np.mean(results['digital']['latencies']),
            'analog_avg_latency': np.mean(results['analog']['latencies']),
            'digital_throughput': max_new_tokens / np.mean(results['digital']['latencies']),
            'analog_throughput': max_new_tokens / np.mean(results['analog']['latencies']),
            'digital_sample_output': results['digital']['outputs'][0],
            'analog_sample_output': results['analog']['outputs'][0]
        }
    
    def energy_analysis(self, sample_input: torch.Tensor = None) -> Dict:
        """
        Estimate energy consumption for the LLM.
        
        Uses the AnalogEnergyModel to compute detailed energy breakdown.
        """
        from energy.analog_energy_model import AnalogEnergyModel
        
        if self.analog_model is None:
            self.load_model()
        
        # Create energy model
        energy_model = AnalogEnergyModel(tech_node='28nm')
        
        # Create sample input if not provided
        if sample_input is None:
            sample_input = torch.randint(0, 1000, (1, 32))  # batch=1, seq=32
        
        # Estimate energy
        energy = energy_model.estimate_model_energy(self.analog_model, sample_input)
        
        # Compare with digital
        n_params = sum(p.numel() for p in self.analog_model.parameters())
        digital_energy_estimate = n_params * 10e-12  # 10 pJ per parameter (GPU estimate)
        
        return {
            'analog_energy': energy,
            'digital_energy_estimate_J': digital_energy_estimate,
            'energy_savings_ratio': digital_energy_estimate / energy['total_energy_J']
        }


def benchmark_gpt2_analog():
    """
    Example: Benchmark GPT-2 with analog non-idealities.
    
    This demonstrates the full pipeline for evaluating a real LLM
    on simulated analog hardware.
    """
    print("=" * 80)
    print("GPT-2 Analog Hardware Benchmark")
    print("=" * 80)
    
    # Create benchmark
    benchmark = AnalogLLMBenchmark(
        model_name='gpt2',
        analog_config={
            'resistor_mismatch': 0.01,
            'noise_sigma': 0.01,
            'opamp_offset': 0.002,
            'quantization_bits': 8,
            'saturation_vmax': 2.5
        }
    )
    
    # Load models
    digital_model, analog_model = benchmark.load_model()
    
    # Evaluate perplexity
    print("\nEvaluating perplexity...")
    test_text = "The quick brown fox jumps over the lazy dog. " * 10
    ppl_results = benchmark.evaluate_perplexity(test_text)
    
    print(f"Digital perplexity: {ppl_results['digital_perplexity']:.2f}")
    print(f"Analog perplexity: {ppl_results['analog_perplexity']:.2f}")
    print(f"Degradation: {ppl_results['perplexity_degradation']:.2f}%")
    
    # Benchmark inference
    print("\nBenchmarking inference...")
    inference_results = benchmark.benchmark_inference(
        prompt="Once upon a time",
        max_new_tokens=20,
        n_runs=5
    )
    
    print(f"Digital latency: {inference_results['digital_avg_latency']*1000:.2f} ms")
    print(f"Analog latency: {inference_results['analog_avg_latency']*1000:.2f} ms")
    print(f"Digital throughput: {inference_results['digital_throughput']:.2f} tokens/s")
    print(f"Analog throughput: {inference_results['analog_throughput']:.2f} tokens/s")
    
    # Energy analysis
    print("\nEnergy analysis...")
    energy_results = benchmark.energy_analysis()
    
    print(f"Analog energy: {energy_results['analog_energy']['total_energy_J']:.2e} J")
    print(f"Digital energy: {energy_results['digital_energy_estimate_J']:.2e} J")
    print(f"Energy savings: {energy_results['energy_savings_ratio']:.1f}x")
    
    return {
        'perplexity': ppl_results,
        'inference': inference_results,
        'energy': energy_results
    }


if __name__ == "__main__":
    results = benchmark_gpt2_analog()
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)
