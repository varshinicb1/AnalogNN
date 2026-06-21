"""
Publication-Quality Novel Neuron Schematic
==========================================

Creates professional schematic of the Oscillatory-Ferroelectric Neuron:
- VCOs for input encoding
- Ferroelectric capacitors for weight storage
- PLLs for multiplication
- Injection locking for summation
- Frequency mixer for activation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os


def draw_novel_neuron_schematic():
    """Create publication-quality schematic of novel oscillatory neuron."""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Styling
    color = '#000000'
    linewidth = 2.5
    
    # Title
    ax.text(7, 5.5, 'Oscillatory-Ferroelectric Neuron Architecture', 
            fontsize=18, fontweight='bold', ha='center')
    ax.text(7, 5, 'Frequency-Domain Computation for Ultra-Low-Power Neural Networks', 
            fontsize=12, ha='center', style='italic')
    
    # === INPUT ENCODING (VCOs) ===
    ax.text(0, 3.5, 'Input Encoding', fontsize=12, fontweight='bold', ha='center')
    
    # Draw VCO symbols (squares with sine wave)
    for i, y in enumerate([2.5, 1.5, 0.5, -0.5]):
        # VCO box
        rect = patches.Rectangle((0.5, y-0.3), 1, 0.6, fill=False, 
                               edgecolor='#1f77b4', linewidth=linewidth)
        ax.add_patch(rect)
        # Sine wave symbol
        t = np.linspace(0, 2*np.pi, 20)
        ax.plot(1 + 0.3*np.cos(t), y + 0.15*np.sin(t), color='#1f77b4', linewidth=1.5)
        ax.text(1, y, 'VCO', fontsize=9, ha='center', va='center', color='#1f77b4')
        # Input label
        ax.text(0.3, y, f'x{i+1}', fontsize=10, ha='right', va='center', fontweight='bold')
        # Output wire
        ax.plot([1.5, 2.5], [y, y], color=color, linewidth=linewidth)
        # Frequency label
        ax.text(2, y+0.4, f'f{i+1}', fontsize=9, ha='center', color='#1f77b4')
    
    # === FERROELECTRIC WEIGHTS ===
    ax.text(4, 3.5, 'Ferroelectric Weights', fontsize=12, fontweight='bold', ha='center')
    
    # Draw ferroelectric capacitor symbols
    for i, y in enumerate([2.5, 1.5, 0.5, -0.5]):
        # Capacitor plates
        ax.plot([3.5, 4.5], [y, y], color='#d62728', linewidth=linewidth)
        ax.plot([3.5, 4.5], [y+0.2, y+0.2], color='#d62728', linewidth=linewidth)
        # Dielectric (ferroelectric symbol)
        ax.plot([3.7, 4.3], [y+0.1, y+0.1], color='#d62728', linewidth=1, linestyle='--')
        # Label
        ax.text(4, y-0.4, f'C{i+1}', fontsize=9, ha='center', color='#d62728')
        # Wire
        ax.plot([2.5, 3.5], [y, y], color=color, linewidth=linewidth)
        ax.plot([4.5, 5.5], [y, y], color=color, linewidth=linewidth)
    
    # === PLL MULTIPLICATION ===
    ax.text(7, 3.5, 'PLL Multiplication', fontsize=12, fontweight='bold', ha='center')
    
    # Draw PLL symbol (mixer + VCO in feedback)
    for i, y in enumerate([2.5, 1.5, 0.5, -0.5]):
        # Mixer (circle with X)
        circle = patches.Circle((7, y), 0.35, fill=False, edgecolor='#2ca02c', linewidth=linewidth)
        ax.add_patch(circle)
        ax.plot([6.8, 7.2], [y-0.2, y+0.2], color='#2ca02c', linewidth=linewidth)
        ax.plot([6.8, 7.2], [y+0.2, y-0.2], color='#2ca02c', linewidth=linewidth)
        ax.text(7, y, '×', fontsize=14, ha='center', va='center', color='#2ca02c')
        # Wire
        ax.plot([5.5, 6.65], [y, y], color=color, linewidth=linewidth)
        ax.plot([7.35, 8.5], [y, y], color=color, linewidth=linewidth)
        # Label
        ax.text(7, y+0.5, f'PLL{i+1}', fontsize=9, ha='center', color='#2ca02c')
    
    # === INJECTION LOCKING SUMMATION ===
    ax.text(10, 3.5, 'Injection Locking', fontsize=12, fontweight='bold', ha='center')
    
    # Draw injection locking node (circle with inputs converging)
    # Sum all inputs to a common node
    for i, y in enumerate([2.5, 1.5, 0.5, -0.5]):
        ax.plot([8.5, 10], [y, 1.5], color=color, linewidth=linewidth)
    
    # Common node
    circle = patches.Circle((10, 1.5), 0.3, fill=True, edgecolor='#ff7f0e', 
                           facecolor='#ff7f0e', linewidth=linewidth)
    ax.add_patch(circle)
    ax.text(10, 1.5, 'Σ', fontsize=14, ha='center', va='center', color='white', fontweight='bold')
    
    # Output from summation
    ax.plot([10.3, 11.5], [1.5, 1.5], color=color, linewidth=linewidth)
    ax.text(10.9, 1.8, 'f_sum', fontsize=10, ha='center', color='#ff7f0e')
    
    # === FREQUENCY MIXING ACTIVATION ===
    ax.text(12.5, 3.5, 'Frequency Mixing', fontsize=12, fontweight='bold', ha='center')
    
    # Draw mixer for activation
    circle = patches.Circle((12.5, 1.5), 0.4, fill=False, edgecolor='#9467bd', linewidth=linewidth)
    ax.add_patch(circle)
    ax.plot([12.3, 12.7], [1.3, 1.7], color='#9467bd', linewidth=linewidth)
    ax.plot([12.3, 12.7], [1.7, 1.3], color='#9467bd', linewidth=linewidth)
    ax.text(12.5, 1.5, 'mix', fontsize=10, ha='center', va='center', color='#9467bd')
    
    # Wire
    ax.plot([11.5, 12.1], [1.5, 1.5], color=color, linewidth=linewidth)
    ax.plot([12.9, 14], [1.5, 1.5], color=color, linewidth=linewidth)
    
    # === OUTPUT ===
    ax.text(14.5, 1.5, 'f_out', fontsize=12, ha='left', va='center', fontweight='bold')
    
    # === KEY INNOVATIONS BOX ===
    innovations = [
        "VCO: Input → Frequency encoding",
        "Ferroelectric C: Non-volatile weight storage",
        "PLL: Phase-based multiplication",
        "Injection locking: Natural parallel summation",
        "Frequency mixing: Non-linear activation"
    ]
    
    y_pos = -2
    for i, innovation in enumerate(innovations):
        ax.text(0, y_pos - i*0.7, f"• {innovation}", fontsize=10, ha='left', 
                color='#333333')
    
    # === ADVANTAGES BOX ===
    ax.text(8, -2, 'Key Advantages over GPUs:', fontsize=12, fontweight='bold', ha='left')
    advantages = [
        "1000× lower power (oscillators vs digital gates)",
        "Non-volatile memory (instant on/off)",
        "Natural temporal processing (frequency domain)",
        "Intrinsic parallelism (multiple frequencies coexist)",
        "Radiation hard (analog vs digital)",
        "No von Neumann bottleneck (compute + memory co-located)"
    ]
    
    y_pos = -2.5
    for i, advantage in enumerate(advantages):
        ax.text(8, y_pos - i*0.6, f"✓ {advantage}", fontsize=9, ha='left', 
                color='#2ca02c')
    
    # === MATHEMATICAL FORMULATION ===
    ax.text(0, -5.5, r'$f_i = f_0 + k \cdot \tanh(x_i)$', fontsize=11, ha='left')
    ax.text(4, -5.5, r'$f_{mult} = f_i \cdot (1 + C_{ij})$', fontsize=11, ha='left')
    ax.text(8, -5.5, r'$f_{sum} = \sum_j f_{mult,j}$', fontsize=11, ha='left')
    ax.text(12, -5.5, r'$f_{out} = f_0 + k \cdot \tanh(f_{sum} - f_0)$', fontsize=11, ha='left')
    
    plt.tight_layout()
    
    # Save
    os.makedirs("./demo_output", exist_ok=True)
    plt.savefig("./demo_output/novel_neuron_schematic.png", dpi=300, bbox_inches='tight', facecolor='white')
    print("Novel neuron schematic saved to: ./demo_output/novel_neuron_schematic.png")
    
    plt.close()


if __name__ == "__main__":
    draw_novel_neuron_schematic()
