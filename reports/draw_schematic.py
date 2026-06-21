"""
Programmatic Publication-Quality Schematic Generator
===================================================

Generates a precise, clean, and authentic circuit schematic for the differential
summing-subtractor neural network neuron model using pure matplotlib.
Saves the result directly as a high-resolution PNG for publication.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_resistor(ax, start, end, label="", label_pos="above"):
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    vx, vy = -uy, ux  # Orthogonal vector
    
    # Zig-zag dimensions - cleaner, more professional
    num_turns = 4
    lead_in = 0.25
    zig_len = 0.8
    
    pts = [(x1, y1)]
    # Lead-in end
    pts.append((x1 + lead_in * dx, y1 + lead_in * dy))
    
    # Zig-zag peaks
    zig_start_x = x1 + lead_in * dx
    zig_start_y = y1 + lead_in * dy
    zig_dx, zig_dy = zig_len * dx, zig_len * dy
    
    for i in range(num_turns * 2):
        frac = (i + 0.5) / (num_turns * 2)
        px = zig_start_x + frac * zig_dx
        py = zig_start_y + frac * zig_dy
        side = 1 if i % 2 == 0 else -1
        amp = 0.12  # Smaller amplitude for cleaner look
        pts.append((px + side * amp * vx, py + side * amp * vy))
        
    # Lead-out start and end
    pts.append((x1 + (lead_in + zig_len) * dx, y1 + (lead_in + zig_len) * dy))
    pts.append((x2, y2))
    
    xs, ys = zip(*pts)
    ax.plot(xs, ys, 'k-', lw=2.0)
    
    # Add label with professional LaTeX formatting
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        offset = 0.35
        lx, ly = mx + offset * vx, my + offset * vy
        ha = 'center'
        va = 'bottom' if label_pos == "above" else 'top'
        ax.text(lx, ly, label, fontsize=11, ha=ha, va=va, fontweight='bold')

def draw_opamp(ax, pos, label="OA"):
    x, y = pos
    w, h = 1.5, 1.4
    
    # Op-amp triangle - cleaner design
    pts = [
        (x, y - h/2),
        (x, y + h/2),
        (x + w, y),
        (x, y - h/2)
    ]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, 'k-', lw=2.5)
    
    # Fill triangle for professional look
    ax.fill(xs, ys, facecolor='#f0f0f0', edgecolor='k', lw=2.5)
    
    # Plus/Minus input labels inside triangle
    ax.text(x + 0.2, y + 0.3, '-', fontsize=16, ha='left', va='center', weight='bold')
    ax.text(x + 0.2, y - 0.3, '+', fontsize=14, ha='left', va='center', weight='bold')
    
    # Text label inside/below opamp
    ax.text(x + 0.5, y, label, fontsize=11, ha='left', va='center', weight='bold')
    
    # Return terminal coordinates: (IN-, IN+, OUT)
    return (x, y + 0.35), (x, y - 0.35), (x + w, y)

def draw_ground(ax, pos):
    x, y = pos
    # Vertical line down
    ax.plot([x, x], [y, y - 0.18], 'k-', lw=2.0)
    # Ground plates - cleaner design
    ax.plot([x - 0.25, x + 0.25], [y - 0.18, y - 0.18], 'k-', lw=2.0)
    ax.plot([x - 0.15, x + 0.15], [y - 0.24, y - 0.24], 'k-', lw=2.0)
    ax.plot([x - 0.05, x + 0.05], [y - 0.30, y - 0.30], 'k-', lw=2.0)

def draw_source(ax, pos, label=""):
    x, y = pos
    r = 0.35
    circle = plt.Circle((x, y), r, fill=False, color='k', lw=2.0)
    ax.add_patch(circle)
    
    # Signs inside circle
    ax.text(x, y + 0.12, '+', fontsize=12, ha='center', va='center', weight='bold')
    ax.text(x, y - 0.14, '-', fontsize=14, ha='center', va='center', weight='bold')
    
    # Ground feed
    ax.plot([x, x], [y - r, y - r - 0.12], 'k-', lw=2.0)
    draw_ground(ax, (x, y - r - 0.12))
    
    if label:
        ax.text(x - 0.5, y, label, fontsize=11, ha='right', va='center', fontweight='bold')
        
    return (x, y + r)

def generate_schematic():
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Plot boundaries - much larger canvas
    ax.set_xlim(-2, 14)
    ax.set_ylim(-8, 8)
    
    # Background
    ax.set_facecolor('#ffffff')
    
    # Title - moved higher
    ax.text(6, 7.2, 'Differential Summing-Subtractor Neural Network Architecture', 
            fontsize=18, ha='center', va='center', weight='bold')
    
    # ------------------ INPUT VOLTAGE SOURCES ------------------
    # Spread inputs vertically with more spacing
    v1_out = draw_source(ax, (0, 4.5), label=r"$V_{in,1}$")
    v2_out = draw_source(ax, (0, 2.5), label=r"$V_{in,2}$")
    vbias_out = draw_source(ax, (0, 0.5), label=r"$V_{bias}$")
    v3_out = draw_source(ax, (0, -2.5), label=r"$V_{in,N-1}$")
    v4_out = draw_source(ax, (0, -4.5), label=r"$V_{in,N}$")
    
    # Vertical dots between inputs to indicate arbitrary size
    ax.text(0, 1.5, r"$\vdots$", fontsize=18, ha='center', va='center')
    ax.text(0, -1.0, r"$\vdots$", fontsize=18, ha='center', va='center')
    
    # ------------------ OP-AMP SUMMING STAGES ------------------
    # Op-Amp 1 (Positive weight summer) - moved higher
    op1_in_neg, op1_in_pos, op1_out = draw_opamp(ax, (5.0, 3.0), label="OA1")
    # Ground positive input
    draw_ground(ax, (5.0, 2.6))
    
    # Op-Amp 2 (Negative weight summer) - moved lower
    op2_in_neg, op2_in_pos, op2_out = draw_opamp(ax, (5.0, -3.0), label="OA2")
    # Ground positive input
    draw_ground(ax, (5.0, -3.4))
    
    # ------------------ RESISTORS & JUNCTION 1 (Positive) ------------------
    # Junction bus at x=4.0, from y=1.0 to y=4.5
    ax.plot([4.0, 4.0], [1.0, 4.5], 'k-', lw=2.0)
    ax.plot([4.0, op1_in_neg[0]], [op1_in_neg[1], op1_in_neg[1]], 'k-', lw=2.0) # Junction to IN-
    
    # Resistor 1: V_in,1 to positive summer junction
    ax.plot([v1_out[0], 1.0], [v1_out[1], v1_out[1]], 'k-', lw=2.0)
    draw_resistor(ax, (1.0, v1_out[1]), (2.8, v1_out[1]), label=r"$R_{11}^+$", label_pos="above")
    ax.plot([2.8, 4.0], [v1_out[1], v1_out[1]], 'k-', lw=2.0)
    
    # Resistor 2: V_in,2 to positive summer junction
    ax.plot([v2_out[0], 1.0], [v2_out[1], v2_out[1]], 'k-', lw=2.0)
    draw_resistor(ax, (1.0, v2_out[1]), (2.8, v2_out[1]), label=r"$R_{12}^+$", label_pos="above")
    ax.plot([2.8, 4.0], [v2_out[1], v2_out[1]], 'k-', lw=2.0)
    
    # Resistor Bias to positive summer junction
    ax.plot([vbias_out[0], 1.0], [vbias_out[1], vbias_out[1]], 'k-', lw=2.0)
    ax.plot([1.0, 1.0], [0.5, 1.0], 'k-', lw=2.0) # vertical feed up to 1.0
    draw_resistor(ax, (1.0, 1.0), (2.8, 1.0), label=r"$R_{bias}^+$", label_pos="above")
    ax.plot([2.8, 4.0], [1.0, 1.0], 'k-', lw=2.0)
    
    # ------------------ RESISTORS & JUNCTION 2 (Negative) ------------------
    # Junction bus at x=4.0, from y=-4.5 to y=-1.0
    ax.plot([4.0, 4.0], [-4.5, -1.0], 'k-', lw=2.0)
    ax.plot([4.0, op2_in_neg[0]], [op2_in_neg[1], op2_in_neg[1]], 'k-', lw=2.0) # Junction to IN-
    
    # Resistor Bias to negative summer junction
    ax.plot([vbias_out[0], 1.0], [vbias_out[1], vbias_out[1]], 'k-', lw=2.0)
    ax.plot([1.0, 1.0], [0.5, -1.0], 'k-', lw=2.0) # vertical feed down to -1.0
    draw_resistor(ax, (1.0, -1.0), (2.8, -1.0), label=r"$R_{bias}^-$", label_pos="above")
    ax.plot([2.8, 4.0], [-1.0, -1.0], 'k-', lw=2.0)
    
    # Resistor 3: V_in,N-1 to negative summer junction
    ax.plot([v3_out[0], 1.0], [v3_out[1], v3_out[1]], 'k-', lw=2.0)
    draw_resistor(ax, (1.0, v3_out[1]), (2.8, v3_out[1]), label=r"$R_{N1}^-$", label_pos="above")
    ax.plot([2.8, 4.0], [v3_out[1], v3_out[1]], 'k-', lw=2.0)
    
    # Resistor 4: V_in,N to negative summer junction
    ax.plot([v4_out[0], 1.0], [v4_out[1], v4_out[1]], 'k-', lw=2.0)
    draw_resistor(ax, (1.0, v4_out[1]), (2.8, v4_out[1]), label=r"$R_{N2}^-$", label_pos="above")
    ax.plot([2.8, 4.0], [v4_out[1], v4_out[1]], 'k-', lw=2.0)
    
    # ------------------ FEEDBACK LOOPS (ZERO-CROSSING DESIGN) ------------------
    # Feedback Op-Amp 1: Starts exactly at the op-amp input node, avoiding crossings
    ax.plot([4.8, 4.8], [op1_in_neg[1], 5.0], 'k-', lw=2.0)
    draw_resistor(ax, (4.8, 5.0), (6.8, 5.0), label=r"$R_{f,pos}$")
    ax.plot([6.8, 6.8], [5.0, op1_out[1]], 'k-', lw=2.0)
    ax.plot([op1_out[0], 7.2], [op1_out[1], op1_out[1]], 'k-', lw=2.0) # Out connection
    
    # Feedback Op-Amp 2: Starts exactly at the op-amp input node, avoiding crossings
    ax.plot([4.8, 4.8], [op2_in_neg[1], -5.0], 'k-', lw=2.0)
    draw_resistor(ax, (4.8, -5.0), (6.8, -5.0), label=r"$R_{f,neg}$", label_pos="below")
    ax.plot([6.8, 6.8], [-5.0, op2_out[1]], 'k-', lw=2.0)
    ax.plot([op2_out[0], 7.2], [op2_out[1], op2_out[1]], 'k-', lw=2.0) # Out connection
    
    # ------------------ SUBTRACTOR STAGE (Op-Amp 3) ------------------
    op3_in_neg, op3_in_pos, op3_out = draw_opamp(ax, (9.5, 0.0), label="OA3")
    
    # Connection from Summer 1 Out to IN- of Subtractor
    ax.plot([op1_out[0], 7.5], [op1_out[1], op1_out[1]], 'k-', lw=2.0)
    ax.plot([7.5, 7.5], [op1_out[1], 0.4], 'k-', lw=2.0)
    draw_resistor(ax, (7.5, 0.4), (9.0, 0.4), label=r"$R_1$")
    ax.plot([9.0, op3_in_neg[0]], [0.4, op3_in_neg[1]], 'k-', lw=2.0)
    
    # Connection from Summer 2 Out to IN+ of Subtractor
    ax.plot([op2_out[0], 7.5], [op2_out[1], op2_out[1]], 'k-', lw=2.0)
    ax.plot([7.5, 7.5], [op2_out[1], -0.4], 'k-', lw=2.0)
    draw_resistor(ax, (7.5, -0.4), (9.0, -0.4), label=r"$R_3$")
    ax.plot([9.0, op3_in_pos[0]], [-0.4, op3_in_pos[1]], 'k-', lw=2.0)
    
    # Feedback loop for Subtractor (IN- to OUT)
    ax.plot([9.7, 9.7], [0.4, 1.5], 'k-', lw=2.0)
    draw_resistor(ax, (9.7, 1.5), (11.5, 1.5), label=r"$R_2$")
    ax.plot([11.5, 11.5], [1.5, op3_out[1]], 'k-', lw=2.0)
    
    # Ground Resistor for Subtractor (IN+ to Ground)
    ax.plot([9.7, 9.7], [-0.4, -1.5], 'k-', lw=2.0)
    draw_resistor(ax, (9.7, -1.5), (9.7, -2.8), label=r"$R_4$", label_pos="right")
    draw_ground(ax, (9.7, -2.8))
    
    # Final output line
    ax.plot([op3_out[0], 12.5], [op3_out[1], op3_out[1]], 'k-', lw=2.5)
    ax.arrow(12.5, op3_out[1], 0.3, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
    
    # Voltages labeling - moved further away
    ax.text(7.0, op1_out[1] + 0.4, r"$V_{out,pos}$", fontsize=12, ha='center', weight='bold')
    ax.text(7.0, op2_out[1] - 0.6, r"$V_{out,neg}$", fontsize=12, ha='center', weight='bold')
    ax.text(12.9, op3_out[1] + 0.4, r"$V_{out}$", fontsize=13, ha='left', weight='bold')
    
    # Formula annotations placed with more spacing
    ax.text(5.5, 5.8, r"$V_{out,pos} = -\sum \frac{R_{f,pos}}{R_{ij}^+} V_{in,j} - \frac{R_{f,pos}}{R_{bias}^+} V_{bias}$", fontsize=10, ha='center', color='blue')
    ax.text(5.5, -5.8, r"$V_{out,neg} = -\sum \frac{R_{f,neg}}{R_{ij}^-} V_{in,j} - \frac{R_{f,neg}}{R_{bias}^-} V_{bias}$", fontsize=10, ha='center', color='blue')
    ax.text(10.5, -4.0, r"$V_{out} = V_{out,pos} - V_{out,neg}$" + "\n" + r"$\quad = \sum w_{ij} V_{in,j} + b_i$", fontsize=11, ha='center', color='darkgreen', bbox=dict(boxstyle="round,pad=0.4", fc="#e1f5fe", ec="gray", lw=1.5))
    
    # Stage Titles placed with more spacing
    ax.text(2.0, 5.5, "Synaptic Input Resistors", fontsize=12, ha='center', weight='bold')
    ax.text(5.5, 4.0, "Positive Summer (OA1)", fontsize=12, ha='center', weight='bold')
    ax.text(5.5, -4.0, "Negative Summer (OA2)", fontsize=12, ha='center', weight='bold')
    ax.text(10.0, 2.5, "Differential Subtractor\n(Unity Gain, OA3)", fontsize=12, ha='center', weight='bold')
    
    # Save the output
    os.makedirs("./figures", exist_ok=True)
    plt.savefig("./figures/circuit_schematic.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Successfully drew publication-quality circuit schematic programmatically!")

if __name__ == "__main__":
    generate_schematic()
