import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from plot_style_constants import apply_plot_style, PLOT_COLORS

apply_plot_style()

# Colors
ours_primary = PLOT_COLORS['purple']
neutral = PLOT_COLORS['grey']

# Purple gradient for heatmap
PURPLE_GRADIENT_CONTRAST = [
    '#ddd6fe', '#c4b5fd', '#a78bfa', '#8b5cf6',
    '#7c3aed', '#6d28d9', '#5b21b6', '#4c1d95', '#3b0764',
]

cmap = LinearSegmentedColormap.from_list('thesis_purple_contrast', 
                                         PURPLE_GRADIENT_CONTRAST, N=256)

def get_text_color(value):
    if np.isnan(value):
        return PLOT_COLORS['grey']
    return 'black' if value > 70 else 'white'

# Data - Universal Gate Set
test_sets = ['Set 1', 'Set 2a', 'Set 2b', 'Set 3']

# Overall reasoning format accuracy
overall_reasoning = [89.88, 87.50, 84.60, 0.00]

# Reasoning format criteria (4 criteria x 4 test sets)
reasoning_matrix = np.array([
    [91.10, 91.10, 88.00, 99.90],  # RC1: Wrapped in XML tags
    [91.10, 91.10, 88.00, 90.38],  # RC2: Quantum state tags
    [0.00, 91.00, 87.90, 98.75],   # RC3: Sequential numbering
    [91.00, 87.60, 84.70, 0.00],   # RC4: Measurement section
])

reasoning_labels = [
    'RC1: Wrapped in <circuit> </circuit> tags',
    'RC2: Each state in <quantum_state> </quantum_state> tags',
    'RC3: Measurement outcomes present',
    'RC4: Each quantum state length equal 2^num_qubits',
]

# Create figure with two panels
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Panel (a): Overall reasoning format accuracy bar chart
ax = axes[0]
x_pos = np.arange(len(test_sets))

bars = ax.bar(x_pos, overall_reasoning, color=ours_primary, 
              alpha=0.9, edgecolor='white', linewidth=0.8)

# Add value labels
for i, val in enumerate(overall_reasoning):
    ax.text(i, val + 2, f'{val:.1f}%', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Test Set')
ax.set_ylabel('Overall Reasoning Format Accuracy (%)')
ax.set_xticks(x_pos)
ax.set_xticklabels(test_sets, fontsize=9)
ax.set_ylim([0, 110])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=100, color=neutral, linestyle='--', 
           linewidth=1, alpha=0.5, zorder=0)
ax.text(-0.15, 1.05, '(a)', transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top')

# Panel (b): Heatmap of reasoning criteria
ax = axes[1]
im = ax.imshow(reasoning_matrix, aspect='auto', cmap=cmap, 
               vmin=0, vmax=100, interpolation='nearest')

ax.set_xticks(np.arange(len(test_sets)))
ax.set_yticks(np.arange(len(reasoning_labels)))
ax.set_xticklabels(test_sets, fontsize=9)
ax.set_yticklabels(reasoning_labels, fontsize=8.5)

# Add text annotations
for i in range(len(reasoning_labels)):
    for j in range(len(test_sets)):
        val = reasoning_matrix[i, j]
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

# Save both formats
plt.savefig('reasoning_format_accuracy.pdf', dpi=300, bbox_inches='tight')
plt.savefig('reasoning_format_accuracy.png', dpi=300, bbox_inches='tight')
print("Saved: reasoning_format_accuracy.pdf")
print("Saved: reasoning_format_accuracy.png")
plt.show()