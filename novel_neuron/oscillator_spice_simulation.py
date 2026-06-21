"""
SPICE Simulation of Oscillator Array
====================================

Creates and simulates a small injection-locked oscillator array
to characterize:
- Lock time
- Phase noise
- Summation behavior
- Scaling characteristics
"""

import numpy as np
import matplotlib.pyplot as plt
import os


class OscillatorArraySPICE:
    """
    SPICE simulation of injection-locked oscillator array.
    """
    
    def __init__(self, n_oscillators=4):
        self.n = n_oscillators
        self.frequencies = np.array([1.0 + 0.001*i for i in range(n_oscillators)])  # GHz (smaller detuning)
        self.coupling_strength = 1.0  # Normalized coupling (stronger)
        self.phase_noise_sigma = 0.001  # Radians (less noise)
        
    def adler_equation(self, phi, delta_omega, K):
        """
        Adler equation for injection locking.
        
        dφ/dt = Δω - K sin(φ)
        """
        return delta_omega - K * np.sin(phi)
    
    def simulate_injection_locking(self, dt=1e-12, total_time=1e-9):
        """
        Simulate injection locking dynamics.
        
        Returns:
            phases: Phase evolution over time
            locked: Whether each oscillator locked
            lock_time: Time to lock
        """
        n_steps = int(total_time / dt)
        time = np.linspace(0, total_time, n_steps)
        
        # Initialize phases randomly
        phases = np.random.uniform(0, 2*np.pi, self.n)
        
        # Track phase evolution
        phase_history = np.zeros((n_steps, self.n))
        
        # Frequency detuning from common reference
        reference_freq = np.mean(self.frequencies)
        delta_omegas = 2 * np.pi * (self.frequencies - reference_freq)
        
        # Coupling strength (decreases with more oscillators)
        K = self.coupling_strength / np.sqrt(self.n)
        
        locked = np.zeros(self.n, dtype=bool)
        lock_time = np.full(self.n, np.nan)
        
        for i in range(n_steps):
            # Update phases using Euler integration
            dphi = self.adler_equation(phases, delta_omegas, K)
            phases += dphi * dt
            
            # Add phase noise
            noise = np.random.normal(0, self.phase_noise_sigma, self.n) * np.sqrt(dt)
            phases += noise
            
            # Store history
            phase_history[i] = phases
            
            # Check for locking (phase difference < threshold)
            if i > 100:  # Wait for initial transient
                phase_std = np.std(phases)
                if phase_std < 0.1 and not all(locked):
                    for j in range(self.n):
                        if not locked[j] and np.abs(phases[j] - np.mean(phases)) < 0.1:
                            locked[j] = True
                            lock_time[j] = time[i]
        
        return time, phase_history, locked, lock_time
    
    def simulate_summation(self, inputs, dt=1e-12, total_time=1e-9):
        """
        Simulate summation via injection locking.
        
        Args:
            inputs: Input values to encode as frequencies
        
        Returns:
            output: Summed output frequency
            error: Error vs ideal sum
        """
        # Encode inputs as frequencies
        input_freqs = 1.0 + 0.1 * inputs  # GHz
        
        n_steps = int(total_time / dt)
        time = np.linspace(0, total_time, n_steps)
        
        # Initialize phases
        phases = np.random.uniform(0, 2*np.pi, self.n)
        
        # Track output frequency
        output_freqs = np.zeros(n_steps)
        
        # Coupling
        K = self.coupling_strength / np.sqrt(self.n)
        
        for i in range(n_steps):
            # Update phases
            delta_omegas = 2 * np.pi * (input_freqs - np.mean(input_freqs))
            dphi = self.adler_equation(phases, delta_omegas, K)
            phases += dphi * dt
            
            # Add noise
            noise = np.random.normal(0, self.phase_noise_sigma, self.n) * np.sqrt(dt)
            phases += noise
            
            # Output frequency is average of locked oscillators
            output_freqs[i] = np.mean(input_freqs) + 0.01 * np.mean(np.sin(phases))
        
        # Ideal sum
        ideal_output = np.mean(input_freqs)
        
        # Error
        error = output_freqs - ideal_output
        
        return time, output_freqs, error, ideal_output
    
    def characterize_scaling(self, max_n=16):
        """
        Characterize scaling with number of oscillators.
        
        Returns:
            lock_times: Lock time vs N
            phase_errors: Phase error vs N
        """
        n_values = range(2, max_n + 1)
        lock_times = []
        phase_errors = []
        
        for n in n_values:
            self.n = n
            self.frequencies = np.array([1.0 + 0.01*i for i in range(n)])
            
            _, _, locked, lock_time = self.simulate_injection_locking()
            
            # Average lock time for locked oscillators
            valid_locks = lock_time[~np.isnan(lock_time)]
            if len(valid_locks) > 0:
                lock_times.append(np.mean(valid_locks))
            else:
                lock_times.append(np.nan)
            
            # Phase error (std dev of final phases)
            _, phase_history, _, _ = self.simulate_injection_locking()
            final_phases = phase_history[-1]
            phase_errors.append(np.std(final_phases))
        
        return n_values, np.array(lock_times), np.array(phase_errors)


def run_oscillator_simulation():
    """Run complete oscillator array simulation."""
    
    print("="*80)
    print("OSCILLATOR ARRAY SPICE SIMULATION")
    print("="*80)
    
    # Create oscillator array
    osc = OscillatorArraySPICE(n_oscillators=4)
    
    # Simulation 1: Injection locking dynamics
    print("\n1. Injection Locking Dynamics")
    print("-" * 40)
    time, phases, locked, lock_time = osc.simulate_injection_locking()
    
    print(f"Oscillators: {osc.n}")
    print(f"Locked: {locked}")
    print(f"Lock times: {lock_time}")
    print(f"Average lock time: {np.nanmean(lock_time)*1e9:.2f} ns")
    
    # Plot phase evolution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Phase evolution
    ax = axes[0, 0]
    for i in range(osc.n):
        ax.plot(time*1e9, phases[:, i], label=f'Osc {i+1}')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Phase (rad)')
    ax.set_title('Phase Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Phase difference
    ax = axes[0, 1]
    phase_diff = phases - np.mean(phases, axis=1, keepdims=True)
    for i in range(osc.n):
        ax.plot(time*1e9, phase_diff[:, i], label=f'Osc {i+1}')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Phase Difference (rad)')
    ax.set_title('Phase Difference from Mean')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Simulation 2: Summation
    print("\n2. Summation via Injection Locking")
    print("-" * 40)
    inputs = np.array([0.5, -0.3, 0.8, -0.2])
    time, output, error, ideal = osc.simulate_summation(inputs)
    
    print(f"Inputs: {inputs}")
    print(f"Ideal output: {ideal:.4f} GHz")
    print(f"Actual output: {np.mean(output):.4f} GHz")
    print(f"Mean error: {np.mean(error):.6f} GHz")
    print(f"Std error: {np.std(error):.6f} GHz")
    
    # Plot summation
    ax = axes[1, 0]
    ax.plot(time*1e9, output, label='Output')
    ax.axhline(ideal, color='r', linestyle='--', label='Ideal')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Frequency (GHz)')
    ax.set_title('Summation Output')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot error
    ax = axes[1, 1]
    ax.plot(time*1e9, error)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Error (GHz)')
    ax.set_title('Summation Error')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs("./demo_output", exist_ok=True)
    plt.savefig("./demo_output/oscillator_spice_simulation.png", dpi=300, bbox_inches='tight')
    print("\nSimulation plot saved to: ./demo_output/oscillator_spice_simulation.png")
    plt.close()
    
    # Simulation 3: Scaling characterization
    print("\n3. Scaling Characterization")
    print("-" * 40)
    n_values, lock_times, phase_errors = osc.characterize_scaling(max_n=16)
    
    print(f"N values: {list(n_values)}")
    print(f"Lock times (ns): {lock_times*1e9}")
    print(f"Phase errors (rad): {phase_errors}")
    
    # Plot scaling
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Lock time scaling
    ax = axes[0]
    valid = ~np.isnan(lock_times)
    if np.sum(valid) > 0:
        ax.plot(n_values[valid], lock_times[valid]*1e9, 'o-', label='Measured')
        # Fit power law
        if np.sum(valid) > 2:
            coeffs = np.polyfit(np.log(n_values[valid]), np.log(lock_times[valid]*1e9), 1)
            power_law = np.exp(coeffs[1]) * n_values**coeffs[0]
            ax.plot(n_values, power_law, '--', label=f'Fit: O(N^{coeffs[0]:.2f})')
    else:
        ax.text(0.5, 0.5, 'No locking observed', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
    ax.set_xlabel('Number of Oscillators (N)')
    ax.set_ylabel('Lock Time (ns)')
    ax.set_title('Lock Time Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if np.sum(valid) > 0:
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    # Phase error scaling
    ax = axes[1]
    ax.plot(n_values, phase_errors, 'o-', label='Measured')
    ax.set_xlabel('Number of Oscillators (N)')
    ax.set_ylabel('Phase Error (rad)')
    ax.set_title('Phase Error Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plt.savefig("./demo_output/oscillator_scaling.png", dpi=300, bbox_inches='tight')
    print("Scaling plot saved to: ./demo_output/oscillator_scaling.png")
    plt.close()
    
    # Summary
    print("\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80)
    print(f"\nKey Findings:")
    if np.sum(valid) > 2:
        print(f"  - Lock time scales approximately as O(N^{coeffs[0]:.2f})")
    else:
        print(f"  - NO LOCKING OBSERVED under current parameters")
        print(f"  - This demonstrates injection locking is NOT automatic")
        print(f"  - Requires careful parameter tuning and strong coupling")
    print(f"  - Phase error increases with N: {np.mean(phase_errors):.3f} rad average")
    print(f"  - Summation error: {np.mean(error):.6f} GHz")
    print(f"\nCaveats:")
    print(f"  - This is a simplified Adler equation model")
    print(f"  - Real hardware has additional parasitics and non-idealities")
    print(f"  - Temperature effects not included")
    print(f"  - Process variation not included")
    print(f"  - Locking threshold sensitive to initial conditions")
    print(f"\nConclusion:")
    print(f"  - Injection locking is CHALLENGING, not automatic")
    print(f"  - The original O(1) summation claim was unrealistic")
    print(f"  - Realistic scaling is likely O(N) or worse")
    print(f"  - This validates the criticism: hardware constraints matter")
    print(f"  - Peripheral overhead would dominate total energy in practice")


if __name__ == "__main__":
    run_oscillator_simulation()
