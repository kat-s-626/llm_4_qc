import numpy as np
import matplotlib.pyplot as plt
from parse_accuracy_file import parse_accuracy_file
import sys
from plot_style_constants import apply_plot_style, PLOT_COLORS

apply_plot_style()

# Colors
primary = PLOT_COLORS['purple']
universal_color = PLOT_COLORS['orange']
neutral = PLOT_COLORS['grey']

# File paths - can be passed as command line arguments or use defaults
if len(sys.argv) >= 3:
    grover_files = sys.argv[1].split(',')
    universal_files = sys.argv[2].split(',')
else:
    # Default file paths (modify these to your actual file locations)
    grover_files = [
        'grover_grpo (1)/grover_gates_py_test_1_summary.txt',
        'grover_grpo (1)/grover_gates_py_test_2a_summary.txt', 
        'grover_grpo (1)/grover_gates_py_test_2b_summary.txt',
        'grover_grpo (1)/grover_gates_py_test_3_summary.txt'
    ]
    universal_files = [
        'random_grpo (1)/random_gates_py_test_1_summary.txt',
        'random_grpo (1)/random_gates_py_test_2a_summary.txt',
        'random_grpo (1)/random_gates_py_test_2b_summary.txt',
        'random_grpo (1)/random_gates_py_test_3_summary.txt'
    ]

# Data
test_sets = ['Set 1', 'Set 2a', 'Set 2b', 'Set 3']

# Parse Grover Gate set data
grover_overall_acc = []
for filepath in grover_files:
    try:
        data = parse_accuracy_file(filepath)
        grover_overall_acc.append(data['overall_format'])
    except FileNotFoundError:
        print(f"Warning: {filepath} not found, using default value")
        grover_overall_acc.append(0.0)

# Parse Universal Gate set data
universal_overall_acc = []
for filepath in universal_files:
    try:
        data = parse_accuracy_file(filepath)
        universal_overall_acc.append(data['overall_format'])
    except FileNotFoundError:
        print(f"Warning: {filepath} not found, using default value")
        universal_overall_acc.append(0.0)

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

x_pos = np.arange(len(test_sets))
bar_width = 0.35

bars1 = ax.bar(x_pos - bar_width/2, grover_overall_acc, bar_width, 
               label='Grover Gate', color=primary, 
               alpha=0.9, edgecolor='white', linewidth=0.8)
bars2 = ax.bar(x_pos + bar_width/2, universal_overall_acc, bar_width,
               label='Universal Gate', color=universal_color, 
               alpha=0.9, edgecolor='white', linewidth=0.8)

# Add value labels
for i, val in enumerate(grover_overall_acc):
    ax.text(i - bar_width/2, val + 2, f'{val:.1f}%', 
            ha='center', va='bottom', fontsize=8, fontweight='bold')
for i, val in enumerate(universal_overall_acc):
    ax.text(i + bar_width/2, val + 2, f'{val:.1f}%', 
            ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Test Set')
ax.set_ylabel('Overall Format Accuracy (%)')
ax.set_xticks(x_pos)
ax.set_xticklabels(test_sets, fontsize=9)
ax.set_ylim([0, 110])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=100, color=neutral, linestyle='--', 
           linewidth=1, alpha=0.5, zorder=0)
ax.legend(loc='upper right', frameon=False)

plt.tight_layout()

# Save both PDF and PNG with higher resolution
plt.savefig('format_accuracy_barchart.pdf', dpi=600, bbox_inches='tight')
plt.savefig('format_accuracy_barchart.png', dpi=600, bbox_inches='tight')
print("Saved: format_accuracy_barchart.pdf")
print("Saved: format_accuracy_barchart.png")
plt.show()
