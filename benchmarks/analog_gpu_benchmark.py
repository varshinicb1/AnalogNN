"""
Analog vs GPU Performance Benchmark
===================================

Comprehensive benchmark comparing analog neural network inference
against GPU performance across multiple metrics:

- Latency (time per token/inference)
- Power consumption (estimated)
- Throughput (tokens/second, inferences/second)
- Accuracy (with and without analog non-idealities)
- Energy efficiency (operations per joule)

Theoretical Basis:
- Analog matrix multiplication: O(1) time, O(n) power (current-mode)
- Digital GPU: O(log n) time, O(n log n) power (parallel but limited by memory bandwidth)
- Analog advantage: Constant-time matrix operations, no memory fetch overhead
- Analog disadvantage: Non-idealities (noise, mismatch, drift) reduce accuracy
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt

# Import analog layers
import sys
sys.path.append('..')
from analog_layers.analog_llm import AnalogLLM, AnalogLLMInferenceEngine
from analog_layers.analog_transformer import AnalogTransformerBlock
from analog_layers.analog_linear import AnalogLinear


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model_name: str
    device: str
    batch_size: int
    seq_len: int
    latency_ms: float
    throughput_tokens_per_sec: float
    power_watts: float
    energy_joules: float
    accuracy: float
    memory_mb: float


class GPUBenchmark:
    """
    GPU benchmark using standard PyTorch models.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize GPU benchmark.
        
        Args:
            device: Device to use ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def benchmark_linear_layer(self,
                              in_features: int,
                              out_features: int,
                              batch_size: int = 32,
                              num_runs: int = 100) -> BenchmarkResult:
        """
        Benchmark a linear layer on GPU.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            batch_size: Batch size
            num_runs: Number of benchmark runs
        
        Returns:
            BenchmarkResult with performance metrics
        """
        # Create model
        model = nn.Linear(in_features, out_features).to(self.device)
        model.eval()
        
        # Warmup
        x = torch.randn(batch_size, in_features).to(self.device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        
        # Benchmark
        torch.cuda.synchronize() if self.device == "cuda" else None
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                output = model(x)
        
        torch.cuda.synchronize() if self.device == "cuda" else None
        end_time = time.time()
        
        # Calculate metrics
        avg_latency_ms = (end_time - start_time) * 1000 / num_runs
        throughput = batch_size / ((end_time - start_time) / num_runs)
        
        # Estimate power (GPU typical: 200-300W for inference)
        power_watts = 250.0 if self.device == "cuda" else 65.0
        energy_joules = power_watts * (end_time - start_time) / num_runs
        
        # Memory usage
        memory_mb = torch.cuda.memory_allocated() / 1024 / 1024 if self.device == "cuda" else 0
        
        return BenchmarkResult(
            model_name=f"Linear_{in_features}_{out_features}",
            device=self.device,
            batch_size=batch_size,
            seq_len=1,
            latency_ms=avg_latency_ms,
            throughput_tokens_per_sec=throughput,
            power_watts=power_watts,
            energy_joules=energy_joules,
            accuracy=1.0,  # Digital baseline
            memory_mb=memory_mb
        )
    
    def benchmark_transformer_block(self,
                                   embed_dim: int,
                                   num_heads: int,
                                   batch_size: int = 32,
                                   seq_len: int = 128,
                                   num_runs: int = 50) -> BenchmarkResult:
        """
        Benchmark a transformer block on GPU.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            batch_size: Batch size
            seq_len: Sequence length
            num_runs: Number of benchmark runs
        
        Returns:
            BenchmarkResult with performance metrics
        """
        # Create standard transformer block
        from transformers import BertModel, BertConfig
        
        config = BertConfig(
            hidden_size=embed_dim,
            num_attention_heads=num_heads,
            intermediate_size=embed_dim * 4,
            num_hidden_layers=1
        )
        model = BertModel(config).to(self.device)
        model.eval()
        
        # Warmup
        x = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
        for _ in range(5):
            with torch.no_grad():
                _ = model(x)
        
        # Benchmark
        torch.cuda.synchronize() if self.device == "cuda" else None
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                output = model(x)
        
        torch.cuda.synchronize() if self.device == "cuda" else None
        end_time = time.time()
        
        # Calculate metrics
        avg_latency_ms = (end_time - start_time) * 1000 / num_runs
        throughput = (batch_size * seq_len) / ((end_time - start_time) / num_runs)
        
        # Estimate power
        power_watts = 250.0 if self.device == "cuda" else 65.0
        energy_joules = power_watts * (end_time - start_time) / num_runs
        
        # Memory usage
        memory_mb = torch.cuda.memory_allocated() / 1024 / 1024 if self.device == "cuda" else 0
        
        return BenchmarkResult(
            model_name=f"Transformer_{embed_dim}_{num_heads}",
            device=self.device,
            batch_size=batch_size,
            seq_len=seq_len,
            latency_ms=avg_latency_ms,
            throughput_tokens_per_sec=throughput,
            power_watts=power_watts,
            energy_joules=energy_joules,
            accuracy=1.0,
            memory_mb=memory_mb
        )


class AnalogBenchmark:
    """
    Analog benchmark using analog neural network layers.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize analog benchmark.
        
        Args:
            device: Device to use (analog simulation runs on CPU)
        """
        self.device = device
        
    def benchmark_linear_layer(self,
                              in_features: int,
                              out_features: int,
                              batch_size: int = 32,
                              num_runs: int = 100,
                              enable_mismatch: bool = True,
                              mismatch_sigma: float = 0.01) -> BenchmarkResult:
        """
        Benchmark an analog linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            batch_size: Batch size
            num_runs: Number of benchmark runs
            enable_mismatch: Enable resistor mismatch
            mismatch_sigma: Mismatch standard deviation
        
        Returns:
            BenchmarkResult with performance metrics
        """
        # Create analog model
        model = AnalogLinear(
            in_features=in_features,
            out_features=out_features,
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma
        ).to(self.device)
        model.eval()
        
        # Warmup
        x = torch.randn(batch_size, in_features).to(self.device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                output = model(x)
        
        end_time = time.time()
        
        # Calculate metrics
        avg_latency_ms = (end_time - start_time) * 1000 / num_runs
        throughput = batch_size / ((end_time - start_time) / num_runs)
        
        # Estimate power (analog: much lower than GPU)
        # Analog matrix multiplication: ~10x more efficient
        num_ops = in_features * out_features * batch_size
        power_watts = num_ops * 1e-6  # Watts (analog estimate)
        energy_joules = power_watts * (end_time - start_time) / num_runs
        
        # Calculate accuracy degradation due to mismatch
        # Compare with ideal digital computation
        model_ideal = AnalogLinear(
            in_features=in_features,
            out_features=out_features,
            enable_mismatch=False
        ).to(self.device)
        model_ideal.load_state_dict(model.state_dict())
        model_ideal.eval()
        
        with torch.no_grad():
            output_analog = model(x)
            output_ideal = model_ideal(x)
        
        accuracy = 1.0 - torch.mean(torch.abs(output_analog - output_ideal)).item()
        
        return BenchmarkResult(
            model_name=f"AnalogLinear_{in_features}_{out_features}",
            device="analog",
            batch_size=batch_size,
            seq_len=1,
            latency_ms=avg_latency_ms,
            throughput_tokens_per_sec=throughput,
            power_watts=power_watts,
            energy_joules=energy_joules,
            accuracy=accuracy,
            memory_mb=0  # Analog doesn't use memory in the same way
        )
    
    def benchmark_transformer_block(self,
                                   embed_dim: int,
                                   num_heads: int,
                                   batch_size: int = 32,
                                   seq_len: int = 128,
                                   num_runs: int = 50,
                                   enable_mismatch: bool = True,
                                   mismatch_sigma: float = 0.01) -> BenchmarkResult:
        """
        Benchmark an analog transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            batch_size: Batch size
            seq_len: Sequence length
            num_runs: Number of benchmark runs
            enable_mismatch: Enable resistor mismatch
            mismatch_sigma: Mismatch standard deviation
        
        Returns:
            BenchmarkResult with performance metrics
        """
        # Create analog transformer block
        model = AnalogTransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma
        ).to(self.device)
        model.eval()
        
        # Warmup
        x = torch.randn(batch_size, seq_len, embed_dim).to(self.device)
        for _ in range(5):
            with torch.no_grad():
                _ = model(x)
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                output, _ = model(x)
        
        end_time = time.time()
        
        # Calculate metrics
        avg_latency_ms = (end_time - start_time) * 1000 / num_runs
        throughput = (batch_size * seq_len) / ((end_time - start_time) / num_runs)
        
        # Estimate power
        num_ops = seq_len * seq_len * embed_dim * batch_size * 4  # Rough estimate
        power_watts = num_ops * 1e-6
        energy_joules = power_watts * (end_time - start_time) / num_runs
        
        # Calculate accuracy
        model_ideal = AnalogTransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            enable_mismatch=False
        ).to(self.device)
        model_ideal.eval()
        
        with torch.no_grad():
            output_analog, _ = model(x)
            output_ideal, _ = model_ideal(x)
        
        accuracy = 1.0 - torch.mean(torch.abs(output_analog - output_ideal)).item()
        
        return BenchmarkResult(
            model_name=f"AnalogTransformer_{embed_dim}_{num_heads}",
            device="analog",
            batch_size=batch_size,
            seq_len=seq_len,
            latency_ms=avg_latency_ms,
            throughput_tokens_per_sec=throughput,
            power_watts=power_watts,
            energy_joules=energy_joules,
            accuracy=accuracy,
            memory_mb=0
        )


class BenchmarkSuite:
    """
    Complete benchmark suite comparing analog and GPU performance.
    """
    
    def __init__(self, output_dir: str = "./benchmarks/results"):
        """
        Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        self.gpu_benchmark = GPUBenchmark()
        self.analog_benchmark = AnalogBenchmark()
        self.results = []
        
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark across multiple configurations."""
        configs = [
            # Linear layer benchmarks
            {"type": "linear", "in_features": 512, "out_features": 512, "batch_size": 32},
            {"type": "linear", "in_features": 1024, "out_features": 1024, "batch_size": 32},
            {"type": "linear", "in_features": 4096, "out_features": 4096, "batch_size": 16},
            
            # Transformer block benchmarks
            {"type": "transformer", "embed_dim": 512, "num_heads": 8, "batch_size": 32, "seq_len": 128},
            {"type": "transformer", "embed_dim": 768, "num_heads": 12, "batch_size": 16, "seq_len": 256},
            {"type": "transformer", "embed_dim": 1024, "num_heads": 16, "batch_size": 8, "seq_len": 512},
        ]
        
        for config in configs:
            print(f"\nBenchmarking: {config}")
            
            if config["type"] == "linear":
                # GPU benchmark
                gpu_result = self.gpu_benchmark.benchmark_linear_layer(
                    in_features=config["in_features"],
                    out_features=config["out_features"],
                    batch_size=config["batch_size"]
                )
                self.results.append(gpu_result)
                
                # Analog benchmark
                analog_result = self.analog_benchmark.benchmark_linear_layer(
                    in_features=config["in_features"],
                    out_features=config["out_features"],
                    batch_size=config["batch_size"]
                )
                self.results.append(analog_result)
                
            elif config["type"] == "transformer":
                # GPU benchmark
                gpu_result = self.gpu_benchmark.benchmark_transformer_block(
                    embed_dim=config["embed_dim"],
                    num_heads=config["num_heads"],
                    batch_size=config["batch_size"],
                    seq_len=config["seq_len"]
                )
                self.results.append(gpu_result)
                
                # Analog benchmark
                analog_result = self.analog_benchmark.benchmark_transformer_block(
                    embed_dim=config["embed_dim"],
                    num_heads=config["num_heads"],
                    batch_size=config["batch_size"],
                    seq_len=config["seq_len"]
                )
                self.results.append(analog_result)
        
        self.save_results()
        self.plot_results()
        
    def save_results(self):
        """Save benchmark results to JSON."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        results_dict = [asdict(r) for r in self.results]
        
        with open(f"{self.output_dir}/benchmark_results.json", 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to {self.output_dir}/benchmark_results.json")
    
    def plot_results(self):
        """Generate comparison plots."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Group results by model
        model_names = set(r.model_name for r in self.results)
        
        for model_name in model_names:
            model_results = [r for r in self.results if r.model_name == model_name]
            
            if len(model_results) < 2:
                continue
            
            gpu_result = next((r for r in model_results if r.device != "analog"), None)
            analog_result = next((r for r in model_results if r.device == "analog"), None)
            
            if not gpu_result or not analog_result:
                continue
            
            # Create comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Latency comparison
            axes[0, 0].bar(['GPU', 'Analog'], [gpu_result.latency_ms, analog_result.latency_ms])
            axes[0, 0].set_ylabel('Latency (ms)')
            axes[0, 0].set_title(f'Latency Comparison - {model_name}')
            
            # Throughput comparison
            axes[0, 1].bar(['GPU', 'Analog'], [gpu_result.throughput_tokens_per_sec, analog_result.throughput_tokens_per_sec])
            axes[0, 1].set_ylabel('Throughput (tokens/sec)')
            axes[0, 1].set_title(f'Throughput Comparison - {model_name}')
            
            # Power comparison
            axes[1, 0].bar(['GPU', 'Analog'], [gpu_result.power_watts, analog_result.power_watts])
            axes[1, 0].set_ylabel('Power (W)')
            axes[1, 0].set_title(f'Power Comparison - {model_name}')
            
            # Accuracy comparison
            axes[1, 1].bar(['GPU', 'Analog'], [gpu_result.accuracy, analog_result.accuracy])
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title(f'Accuracy Comparison - {model_name}')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{model_name}_comparison.png", dpi=300)
            plt.close()
        
        print(f"Plots saved to {self.output_dir}/")
    
    def print_summary(self):
        """Print summary of benchmark results."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        model_names = set(r.model_name for r in self.results)
        
        for model_name in model_names:
            model_results = [r for r in self.results if r.model_name == model_name]
            
            gpu_result = next((r for r in model_results if r.device != "analog"), None)
            analog_result = next((r for r in model_results if r.device == "analog"), None)
            
            if not gpu_result or not analog_result:
                continue
            
            print(f"\n{model_name}:")
            print(f"  Latency: GPU={gpu_result.latency_ms:.2f}ms, Analog={analog_result.latency_ms:.2f}ms")
            print(f"  Throughput: GPU={gpu_result.throughput_tokens_per_sec:.2f} tok/s, Analog={analog_result.throughput_tokens_per_sec:.2f} tok/s")
            print(f"  Power: GPU={gpu_result.power_watts:.2f}W, Analog={analog_result.power_watts:.2f}W")
            print(f"  Accuracy: GPU={gpu_result.accuracy:.4f}, Analog={analog_result.accuracy:.4f}")
            print(f"  Power Savings: {(gpu_result.power_watts - analog_result.power_watts) / gpu_result.power_watts * 100:.1f}%")
            print(f"  Accuracy Loss: {(gpu_result.accuracy - analog_result.accuracy) * 100:.2f}%")


if __name__ == "__main__":
    suite = BenchmarkSuite()
    suite.run_comprehensive_benchmark()
    suite.print_summary()
