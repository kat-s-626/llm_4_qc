import numpy as np
import matplotlib.pyplot as plt
from parse_accuracy_file import parse_accuracy_file
import sys
from visualization.constants import apply_plot_style, PLOT_COLORS

apply_plot_style()

# Text color helper function
def get_text_color(value):
    """Determine text color based on background luminosity"""
    # For viridis colormap, use black for lighter backgrounds, white for darker
    if np.isnan(value):
        return 'black'
    return 'white' if value < 50 else 'black'

# File paths - can be passed as command line arguments or use defaults
if len(sys.argv) >= 3:
    grover_files = sys.argv[1].split(',')
    universal_files = sys.argv[2].split(',')
else:
    # Default file paths (modify these to your actual file locations)
    grover_files = [
        'grover_gates_py_combined_qwen3_8b/grover_gates_py_test_1_summary.txt',
        'grover_gates_py_combined_qwen3_8b/grover_gates_py_test_2a_summary.txt', 
        'grover_gates_py_combined_qwen3_8b/grover_gates_py_test_2b_summary.txt',
        'grover_gates_py_combined_qwen3_8b/grover_gates_py_test_3_summary.txt'
    ]
    universal_files = [
        'random_standard_gates_py_qwen3_8b/random_gates_py_test_1_summary.txt',
        'random_standard_gates_py_qwen3_8b/random_gates_py_test_2a_summary.txt',
        'random_standard_gates_py_qwen3_8b/random_gates_py_test_2b_summary.txt',
        'random_standard_gates_py_qwen3_8b/random_gates_py_test_3_summary.txt'
    ]

# Data
test_sets = ['Set 1', 'Set 2a', 'Set 2b', 'Set 3']

# Parse Grover Gate criteria data (4 criteria x 4 test sets)
grover_criteria_matrix = []
for filepath in grover_files:
    try:
        data = parse_accuracy_file(filepath)
        grover_criteria_matrix.append(data['reasoning_criteria'])
    except FileNotFoundError:
        print(f"Warning: {filepath} not found, using default values")
        grover_criteria_matrix.append([0.0] * 4)

grover_criteria_matrix = np.array(grover_criteria_matrix).T  # Transpose to get (4 x 4)

# Parse Universal Gate criteria data (4 criteria x 4 test sets)
universal_criteria_matrix = []
for filepath in universal_files:
    try:
        data = parse_accuracy_file(filepath)
        universal_criteria_matrix.append(data['reasoning_criteria'])
    except FileNotFoundError:
        print(f"Warning: {filepath} not found, using default values")
        universal_criteria_matrix.append([0.0] * 4)

universal_criteria_matrix = np.array(universal_criteria_matrix).T  # Transpose to get (4 x 4)

criteria_labels = [
    'RC1: <circuit_reasoning> present',
    'RC2: <quantum_state> present',
    'RC3: Outcome probability present',
    'RC4: Correct quantum state length'
]

# Create figure with 2 heatmaps side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Panel (a): Heatmap of criteria for Grover Gate
ax = axes[0]
im = ax.imshow(grover_criteria_matrix, aspect='auto', 
               vmin=0, vmax=100, interpolation='nearest')

ax.set_xticks(np.arange(len(test_sets)))
ax.set_yticks(np.arange(len(criteria_labels)))
ax.set_xticklabels(test_sets, fontsize=9)
ax.set_yticklabels(criteria_labels, fontsize=9)
ax.set_title('Grover Gate Set', fontsize=11, pad=10)

# Add text annotations with proper color selection
for i in range(len(criteria_labels)):
    for j in range(len(test_sets)):
        val = grover_criteria_matrix[i, j]
        if np.isnan(val):
            text = 'N/A'
            color = PLOT_COLORS['grey']
        else:
            text = f'{val:.1f}' if val < 100 else '100'
            color = get_text_color(val)
        ax.text(j, i, text, ha='center', va='center',
                fontsize=8, color=color, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Accuracy (%)', rotation=270, labelpad=20)

ax.text(-0.15, 1.05, '(a)', transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top')

# Panel (b): Heatmap of criteria for Universal Gate
ax = axes[1]
im = ax.imshow(universal_criteria_matrix, aspect='auto', 
               vmin=0, vmax=100, interpolation='nearest')

ax.set_xticks(np.arange(len(test_sets)))
ax.set_yticks(np.arange(len(criteria_labels)))
ax.set_xticklabels(test_sets, fontsize=9)
ax.set_yticklabels(criteria_labels, fontsize=9)
ax.set_title('Universal Gate Set', fontsize=11, pad=10)

# Add text annotations with proper color selection
for i in range(len(criteria_labels)):
    for j in range(len(test_sets)):
        val = universal_criteria_matrix[i, j]
        if np.isnan(val):
            text = 'N/A'
            color = PLOT_COLORS['grey']
        else:
            text = f'{val:.1f}' if val < 100 else '100'
            color = get_text_color(val)
        ax.text(j, i, text, ha='center', va='center',
                fontsize=8, color=color, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Accuracy (%)', rotation=270, labelpad=20)

ax.text(-0.15, 1.05, '(b)', transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top')

plt.tight_layout()

# Save both PDF and PNG with higher resolution
plt.savefig('reasoning_accuracy_heatmaps.pdf', dpi=600, bbox_inches='tight')
plt.savefig('reasoning_accuracy_heatmaps.png', dpi=600, bbox_inches='tight')
print("Saved: reasoning_accuracy_heatmaps.pdf")
print("Saved: reasoning_accuracy_heatmaps.png")
plt.show()
