import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
from plot_style_constants import apply_plot_style, PLOT_COLORS

apply_plot_style()

MODEL_COLORS = {
    'SFT': PLOT_COLORS['purple'],
    'GRPO': PLOT_COLORS['teal'],
    'SFT+GRPO': '#3b82f6',
    'Qwen3-8B': PLOT_COLORS['orange'],
    'GPT-OSS-120B': '#ef4444',
}


BASE_DIR = Path('./baselines')

# Define file paths for each model and gate set
# Format: {model_name: {gate_set: file_path}}
FILE_PATHS = {
    'SFT': {
        'grover': BASE_DIR / 'grover_gate_sft_steps' / 'grover_gates_py_test_1_summary.txt',
        'universal': BASE_DIR / 'random_gate_sft_steps' / 'random_gates_py_test_1_summary.txt'
    },
    'GRPO': {
        'grover': BASE_DIR / 'grover_grpo' / 'grover_gates_py_test_1_summary.txt',
        'universal': BASE_DIR / 'random_grpo' / 'random_gates_py_test_1_summary.txt'
    },
    'SFT+GRPO': {
        'grover': BASE_DIR / 'grover_sft_grpo' / 'grover_gates_py_test_1_summary.txt',
        'universal': BASE_DIR / 'random_sft_grpo' / 'random_gates_py_test_1_summary.txt'
    },
    'Qwen3-8B': {
        'grover': BASE_DIR / 'Qwen_Qwen3-8B' / 'grover_gates_py_test_1_summary.txt',
        'universal': BASE_DIR / 'Qwen_Qwen3-8B' / 'random_gates_py_test_1_summary.txt'
    },
    'GPT-OSS-120B': {
        'grover': BASE_DIR / 'openai_gpt-oss-120b' / 'grover_gates_py_test_1_summary.txt',
        'universal': BASE_DIR / 'openai_gpt-oss-120b' / 'random_gates_py_test_1_summary.txt'
    }
}


def extract_tvd_from_file(file_path, metric_type='qubits'):
    """
    Extract Avg TVD values from a summary file.
    
    Args:
        file_path: Path to the summary file
        metric_type: One of 'qubits', 'depth', 'gates'
    
    Returns:
        Dictionary mapping qubit/depth/gate ranges to TVD values
    """
    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    tvd_data = {}
    
    if metric_type == 'qubits':
        # Look for "METRICS BY NUMBER OF QUBITS:"
        section_match = re.search(r'METRICS BY NUMBER OF QUBITS:(.*?)(?=\n  METRICS|\Z)', content, re.DOTALL)
        if section_match:
            section = section_match.group(1)
            # Extract each qubit count
            for qubit in [1, 2, 3, 4, 5]:
                pattern = rf'{qubit}_qubits \(n=\d+\):.*?Avg TVD Top-k:\s+([\d.]+)'
                match = re.search(pattern, section, re.DOTALL)
                if match:
                    tvd_data[qubit] = float(match.group(1))
    
    elif metric_type == 'depth':
        # Look for "METRICS BY CIRCUIT DEPTH:"
        section_match = re.search(r'METRICS BY CIRCUIT DEPTH:(.*?)(?=\n  METRICS|\Z)', content, re.DOTALL)
        if section_match:
            section = section_match.group(1)
            # Extract depth ranges
            depth_ranges = {
                '1-10': 'Depth 1-10',
                '11-20': 'Depth 11-20',
                '21-30': 'Depth 21-30',
                '31-40': 'Depth 31-40',
                '41-50': 'Depth 41-50'
            }
            for key, label in depth_ranges.items():
                pattern = rf'{label} \(n=\d+\):.*?Avg TVD Top-k:\s+([\d.]+)'
                match = re.search(pattern, section, re.DOTALL)
                if match:
                    tvd_data[key] = float(match.group(1))
    
    elif metric_type == 'gates':
        # Look for "METRICS BY NUMBER OF GATES:"
        section_match = re.search(r'METRICS BY NUMBER OF GATES:(.*?)(?=\Z)', content, re.DOTALL)
        if section_match:
            section = section_match.group(1)
            # Extract gate ranges
            gate_ranges = {
                '1-10': 'Gates 1-10',
                '11-20': 'Gates 11-20',
                '21-30': 'Gates 21-30',
                '31-40': 'Gates 31-40',
                '41-50': 'Gates 41-50'
            }
            for key, label in gate_ranges.items():
                pattern = rf'{label} \(n=\d+\):.*?Avg TVD Top-k:\s+([\d.]+)'
                match = re.search(pattern, section, re.DOTALL)
                if match:
                    tvd_data[key] = float(match.group(1))
    
    return tvd_data


def collect_all_data():
    """Collect TVD data for all models, gate sets, and metric types."""
    data = {
        'qubits': {'grover': {}, 'universal': {}},
        'depth': {'grover': {}, 'universal': {}},
        'gates': {'grover': {}, 'universal': {}}
    }
    
    for model_name, paths in FILE_PATHS.items():
        for gate_set, file_path in paths.items():
            for metric_type in ['qubits', 'depth', 'gates']:
                tvd_values = extract_tvd_from_file(file_path, metric_type)
                data[metric_type][gate_set][model_name] = tvd_values
                print(f"{model_name} - {gate_set} - {metric_type}: {tvd_values}")
    
    return data


def plot_all_metrics(data):
    """Create 6 bar charts (2 gate sets × 3 metric types)."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, left=0.08, right=0.85, top=0.95, bottom=0.05)
    
    models = ['SFT', 'GRPO', 'SFT+GRPO', 'Qwen3-8B', 'GPT-OSS-120B']
    width = 0.15
    
    # Create all axes
    axes = [[fig.add_subplot(gs[row, col]) for col in range(2)] for row in range(3)]
    
    # Row 0: Number of Qubits
    for col_idx, gate_set in enumerate(['grover', 'universal']):
        ax = axes[0][col_idx]
        qubits = [1, 2, 3, 4, 5]
        x = np.arange(len(qubits))
        
        # Filter out SFT+GRPO for grover gate set
        plot_models = [m for m in models if not (gate_set == 'grover' and m == 'SFT+GRPO')]
        
        for i, model in enumerate(plot_models):
            offset = width * (i - 2)
            tvd_values = []
            for qubit in qubits:
                model_data = data['qubits'][gate_set].get(model, {})
                tvd_values.append(model_data.get(qubit, 0))
            
            ax.bar(x + offset, tvd_values, width, 
                     label=model, color=MODEL_COLORS[model], alpha=0.9)
        
        ax.set_xlabel('Number of Qubits')
        ax.set_ylabel('Mean TVD')
        ax.set_xticks(x)
        ax.set_xticklabels(qubits)
        ax.set_ylim([0, max(1.0, max([v for model in models for v in data['qubits'][gate_set].get(model, {}).values()] or [0]) * 1.1)])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        gate_label = 'Grover Sets' if gate_set == 'grover' else 'Universal Sets'
        ax.text(-0.15, 1.10, f'({"ab"[col_idx]}) {gate_label}', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')
    
    # Row 1: Circuit Depth
    for col_idx, gate_set in enumerate(['grover', 'universal']):
        ax = axes[1][col_idx]
        depth_labels = ['1-10', '11-20', '21-30', '31-40', '41-50']
        x = np.arange(len(depth_labels))
        
        # Filter out SFT+GRPO for grover gate set
        plot_models = [m for m in models if not (gate_set == 'grover' and m == 'SFT+GRPO')]
        
        for i, model in enumerate(plot_models):
            offset = width * (i - 2)
            tvd_values = []
            for depth_range in depth_labels:
                model_data = data['depth'][gate_set].get(model, {})
                tvd_values.append(model_data.get(depth_range, 0))
            
            ax.bar(x + offset, tvd_values, width, 
                     label=model, color=MODEL_COLORS[model], alpha=0.9)
        
        ax.set_xlabel('Circuit Depth')
        ax.set_ylabel('Mean TVD')
        ax.set_xticks(x)
        ax.set_xticklabels(depth_labels, fontsize=8)
        ax.set_ylim([0, max(1.0, max([v for model in models for v in data['depth'][gate_set].get(model, {}).values()] or [0]) * 1.1)])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        gate_label = 'Grover Sets' if gate_set == 'grover' else 'Universal Sets'
        ax.text(-0.15, 1.10, f'({"cd"[col_idx]}) {gate_label}', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')
    
    # Row 2: Number of Gates
    for col_idx, gate_set in enumerate(['grover', 'universal']):
        ax = axes[2][col_idx]
        gate_labels = ['1-10', '11-20', '21-30', '31-40', '41-50']
        x = np.arange(len(gate_labels))
        
        # Filter out SFT+GRPO for grover gate set
        plot_models = [m for m in models if not (gate_set == 'grover' and m == 'SFT+GRPO')]
        
        for i, model in enumerate(plot_models):
            offset = width * (i - 2)
            tvd_values = []
            for gate_range in gate_labels:
                model_data = data['gates'][gate_set].get(model, {})
                tvd_values.append(model_data.get(gate_range, 0))
            
            ax.bar(x + offset, tvd_values, width, 
                     label=model, color=MODEL_COLORS[model], alpha=0.9)
        
        ax.set_xlabel('Number of Gates')
        ax.set_ylabel('Mean TVD')
        ax.set_xticks(x)
        ax.set_xticklabels(gate_labels, fontsize=8)
        ax.set_ylim([0, max(1.0, max([v for model in models for v in data['gates'][gate_set].get(model, {}).values()] or [0]) * 1.1)])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        gate_label = 'Grover Sets' if gate_set == 'grover' else 'Universal Sets'
        ax.text(-0.15, 1.10, f'({"ef"[col_idx]}) {gate_label}', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')
    
    # Create a single legend for all plots on the right side
    # Use the Universal Sets plot (column 1) to get all 5 models for the legend
    handles, labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5),
               frameon=False, fontsize=10)
    
    plt.savefig('figures/results/tvd_by_metrics.pdf', dpi=1000, bbox_inches='tight')
    plt.savefig('figures/results/tvd_by_metrics.png', dpi=1000, bbox_inches='tight')
    print("\nSaved: figures/results/tvd_by_metrics.pdf")
    print("Saved: figures/results/tvd_by_metrics.png")
    plt.close()


if __name__ == '__main__':
    # Ensure output directory exists
    output_dir = Path('figures') / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Collecting data from summary files...")
    data = collect_all_data()
    
    print("\nGenerating plots...")
    plot_all_metrics(data)
    
    print("\nDone!")