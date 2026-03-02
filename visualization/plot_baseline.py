import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from pathlib import Path
from matplotlib.gridspec import GridSpec
from plot_style_constants import apply_plot_style, PLOT_COLORS


# Apply  plot style
apply_plot_style()

# Base directory
BASE_DIR = Path('./baselines')


def extract_metric_from_summary(file_path, metric_name):
    """Extract a metric value from a summary file."""
    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return None
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Different patterns for different metrics
    patterns = {
        'Average TVD Top-k (k=15)': r'Average TVD Top-k \(k=15\):\s+([\d.]+)',
        'Average Tokens': r'Average Tokens:\s+([\d.]+)',
        'JSON Parse Success Rate': r'JSON Parse Success Rate:\s+([\d.]+)%',
        'Total Samples': r'Total Samples:\s+(\d+)',
    }
    
    if metric_name not in patterns:
        print(f"Warning: Unknown metric: {metric_name}")
        return None
    
    match = re.search(patterns[metric_name], content)
    if match:
        return float(match.group(1))
    else:
        print(f"Warning: Could not find '{metric_name}' in {file_path.name}")
        return None


def count_perfect_tvd(file_path):
    """Count samples with TVD = 0 from summary file."""
    if not file_path.exists():
        return 0
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Look for individual sample TVDs in the detailed results section
    # This is a simplified approach - we'll use the aggregate metric
    # Count occurrences where TVD is exactly 0.000
    tvd_match = re.search(r'Average TVD Top-k \(k=15\):\s+([\d.]+)', content)
    total_match = re.search(r'Total Samples:\s+(\d+)', content)
    
    if tvd_match and total_match:
        avg_tvd = float(tvd_match.group(1))
        total_samples = int(total_match.group(1))
        
        # If average is 0, all samples are perfect
        if avg_tvd == 0.0:
            return total_samples
    
    return 0


def load_model_data(model_dir):
    """
    Load metrics for a model from its 4 summary files.
    Returns a dictionary with metrics for Python Set 1, Python Set 2, QASM Set 1, QASM Set 2.
    """
    model_path = BASE_DIR / model_dir
    
    if not model_path.exists():
        print(f"Warning: Model directory not found: {model_dir}")
        return None
    
    # Files: baseline_reasoning_eval_1 through baseline_reasoning_eval_4
    # 1 = Python Set 1, 2 = Python Set 2, 3 = QASM Set 1, 4 = QASM Set 2
    files = {
        'python_set1': model_path / 'baseline_reasoning_eval_1_eval_summary.txt',
        'python_set2': model_path / 'baseline_reasoning_eval_2_eval_summary.txt',
        'qasm_set1': model_path / 'baseline_reasoning_eval_3_eval_summary.txt',
        'qasm_set2': model_path / 'baseline_reasoning_eval_4_eval_summary.txt',
    }
    
    data = {}
    
    for key, file_path in files.items():
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            data[key] = {
                'tvd': None,
                'tokens': None,
                'parse_rate': None,
                'total_samples': None,
                'perfect_tvd_count': 0,
            }
            continue
        
        tvd = extract_metric_from_summary(file_path, 'Average TVD Top-k (k=15)')
        tokens = extract_metric_from_summary(file_path, 'Average Tokens')
        parse_rate = extract_metric_from_summary(file_path, 'JSON Parse Success Rate')
        total_samples = extract_metric_from_summary(file_path, 'Total Samples')
        perfect_tvd = count_perfect_tvd(file_path)
        
        data[key] = {
            'tvd': tvd,
            'tokens': tokens,
            'parse_rate': parse_rate / 100.0 if parse_rate is not None else None,  # Convert percentage to fraction
            'total_samples': int(total_samples) if total_samples is not None else None,
            'perfect_tvd_count': perfect_tvd,
        }
    
    return data


def calculate_token_efficiency(perfect_count, total_samples, avg_tokens):
    """
    Calculate token efficiency as: (Perfect Match % / Avg Tokens per 1K)
    Perfect Match = samples with TVD = 0
    """
    if total_samples is None or total_samples == 0 or avg_tokens is None or avg_tokens == 0:
        return 0.0
    
    perfect_percentage = (perfect_count / total_samples) * 100
    tokens_per_1k = avg_tokens / 1000
    
    if tokens_per_1k == 0:
        return 0.0
    
    return perfect_percentage / tokens_per_1k


def main():
    print("="*80)
    print("Loading baseline data from directories...")
    print("="*80)
    
    # Model directories and display names
    models_config = [
        ('Qwen_Qwen3-0.6B', 'Qwen3-0.6B'),
        ('Qwen_Qwen3-1.7B', 'Qwen3-1.7B'),
        ('Qwen_Qwen3-4B', 'Qwen3-4B'),
        ('Qwen_Qwen3-8B', 'Qwen3-8B'),
        ('meta-llama_Meta-Llama-3.1-8B-Instruct', 'Llama-3.1-8B'),
        ('Qwen_Qwen3-14B', 'Qwen3-14B'),
        ('openai_gpt-oss-20b', 'GPT-OSS-20B'),
        ('Qwen_Qwen3-32B', 'Qwen3-32B'),
        ('meta-llama_Llama-3.3-70B-Instruct', 'Llama-3.3-70B'),
        ('openai_gpt-oss-120b', 'GPT-OSS-120B'),
    ]
    
    models = []
    tvd_set1_py = []
    tvd_set1_qa = []
    tvd_set2_py = []
    tvd_set2_qa = []
    
    parse_set1_py = []
    parse_set1_qa = []
    parse_set2_py = []
    parse_set2_qa = []
    
    tokens_set1_py = []
    tokens_set1_qa = []
    tokens_set2_py = []
    tokens_set2_qa = []
    
    token_eff_set1_py = []
    token_eff_set1_qa = []
    token_eff_set2_py = []
    token_eff_set2_qa = []
    
    for model_dir, model_name in models_config:
        print(f"\nLoading data for {model_name}...")
        data = load_model_data(model_dir)
        
        if data is None:
            print(f"  Skipping {model_name} - directory not found")
            continue
        
        models.append(model_name)
        
        # Extract TVD
        tvd_set1_py.append(data['python_set1']['tvd'] if data['python_set1']['tvd'] is not None else 0)
        tvd_set2_py.append(data['python_set2']['tvd'] if data['python_set2']['tvd'] is not None else 0)
        tvd_set1_qa.append(data['qasm_set1']['tvd'] if data['qasm_set1']['tvd'] is not None else 0)
        tvd_set2_qa.append(data['qasm_set2']['tvd'] if data['qasm_set2']['tvd'] is not None else 0)
        
        # Extract Parse Rate
        parse_set1_py.append(data['python_set1']['parse_rate'] if data['python_set1']['parse_rate'] is not None else 0)
        parse_set2_py.append(data['python_set2']['parse_rate'] if data['python_set2']['parse_rate'] is not None else 0)
        parse_set1_qa.append(data['qasm_set1']['parse_rate'] if data['qasm_set1']['parse_rate'] is not None else 0)
        parse_set2_qa.append(data['qasm_set2']['parse_rate'] if data['qasm_set2']['parse_rate'] is not None else 0)
        
        # Extract Tokens (convert to thousands)
        tokens_set1_py.append(data['python_set1']['tokens'] / 1000 if data['python_set1']['tokens'] is not None else 0)
        tokens_set2_py.append(data['python_set2']['tokens'] / 1000 if data['python_set2']['tokens'] is not None else 0)
        tokens_set1_qa.append(data['qasm_set1']['tokens'] / 1000 if data['qasm_set1']['tokens'] is not None else 0)
        tokens_set2_qa.append(data['qasm_set2']['tokens'] / 1000 if data['qasm_set2']['tokens'] is not None else 0)
        
        # Calculate Token Efficiency
        eff1_py = calculate_token_efficiency(
            data['python_set1']['perfect_tvd_count'],
            data['python_set1']['total_samples'],
            data['python_set1']['tokens']
        )
        eff2_py = calculate_token_efficiency(
            data['python_set2']['perfect_tvd_count'],
            data['python_set2']['total_samples'],
            data['python_set2']['tokens']
        )
        eff1_qa = calculate_token_efficiency(
            data['qasm_set1']['perfect_tvd_count'],
            data['qasm_set1']['total_samples'],
            data['qasm_set1']['tokens']
        )
        eff2_qa = calculate_token_efficiency(
            data['qasm_set2']['perfect_tvd_count'],
            data['qasm_set2']['total_samples'],
            data['qasm_set2']['tokens']
        )
        
        token_eff_set1_py.append(eff1_py)
        token_eff_set2_py.append(eff2_py)
        token_eff_set1_qa.append(eff1_qa)
        token_eff_set2_qa.append(eff2_qa)
        
        print(f"  Python Set 1 - TVD: {data['python_set1']['tvd']}, Tokens: {data['python_set1']['tokens']}, Efficiency: {eff1_py:.2f}")
        print(f"  Python Set 2 - TVD: {data['python_set2']['tvd']}, Tokens: {data['python_set2']['tokens']}, Efficiency: {eff2_py:.2f}")
        print(f"  QASM Set 1   - TVD: {data['qasm_set1']['tvd']}, Tokens: {data['qasm_set1']['tokens']}, Efficiency: {eff1_qa:.2f}")
        print(f"  QASM Set 2   - TVD: {data['qasm_set2']['tvd']}, Tokens: {data['qasm_set2']['tokens']}, Efficiency: {eff2_qa:.2f}")
    
    print(f"\n{'='*80}")
    print(f"Loaded data for {len(models)} models")
    print("="*80)
    
    # Create 4x2 grid of subplots with space for legend
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(4, 3, figure=fig, width_ratios=[1, 1, 0.15], wspace=0.3, hspace=0.5)
    
    # Create subplot axes
    axes = []
    for row in range(4):
        row_axes = []
        for col in range(2):
            ax = fig.add_subplot(gs[row, col])
            row_axes.append(ax)
        axes.append(row_axes)
    axes = np.array(axes)
    
    x = np.arange(len(models))
    width = 0.35
    
    # ============================================================================
    # Row 1: TVD (Total Variation Distance)
    # ============================================================================
    
    # Set 1 TVD
    ax = axes[0, 0]
    bars1 = ax.bar(x - width/2, tvd_set1_py, width, label='Python', 
                   color=PLOT_COLORS['purple'], alpha=0.9, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, tvd_set1_qa, width, label='QASM', 
                   color=PLOT_COLORS['orange'], alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Mean TVD')
    ax.set_ylim([0, max(max(tvd_set1_py), max(tvd_set1_qa)) * 1.1])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0.0, color=PLOT_COLORS['grey'], linestyle='--', linewidth=1, alpha=0.5)
    ax.text(-0.1, 1.05, '(a) Set 1: Mean TVD', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom', ha='left')
    
    # Set 2 TVD
    ax = axes[0, 1]
    bars1 = ax.bar(x - width/2, tvd_set2_py, width, label='Python', 
                   color=PLOT_COLORS['purple'], alpha=0.9, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, tvd_set2_qa, width, label='QASM', 
                   color=PLOT_COLORS['orange'], alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Mean TVD')
    ax.set_ylim([0, max(max(tvd_set2_py), max(tvd_set2_qa)) * 1.1])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0.0, color=PLOT_COLORS['grey'], linestyle='--', linewidth=1, alpha=0.5)
    ax.text(-0.1, 1.05, '(b) Set 2: Mean TVD', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom', ha='left')
    
    # ============================================================================
    # Row 2: Parse Success Rate
    # ============================================================================
    
    # Set 1 Parse Rate
    ax = axes[1, 0]
    bars1 = ax.bar(x - width/2, parse_set1_py, width, label='Python', 
                   color=PLOT_COLORS['purple'], alpha=0.9, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, parse_set1_qa, width, label='QASM', 
                   color=PLOT_COLORS['orange'], alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Parse Success Rate')
    ax.set_ylim([0, 1.1])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=1.0, color=PLOT_COLORS['grey'], linestyle='--', linewidth=1, alpha=0.5)
    ax.text(-0.1, 1.05, '(c) Set 1: Parse Success Rate', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom', ha='left')
    
    # Set 2 Parse Rate
    ax = axes[1, 1]
    bars1 = ax.bar(x - width/2, parse_set2_py, width, label='Python', 
                   color=PLOT_COLORS['purple'], alpha=0.9, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, parse_set2_qa, width, label='QASM', 
                   color=PLOT_COLORS['orange'], alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Parse Success Rate')
    ax.set_ylim([0, 1.1])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=1.0, color=PLOT_COLORS['grey'], linestyle='--', linewidth=1, alpha=0.5)
    ax.text(-0.1, 1.05, '(d) Set 2: Parse Success Rate', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom', ha='left')
    
    # ============================================================================
    # Row 3: Average Tokens (in thousands)
    # ============================================================================
    
    # Set 1 Tokens
    ax = axes[2, 0]
    bars1 = ax.bar(x - width/2, tokens_set1_py, width, label='Python', 
                   color=PLOT_COLORS['purple'], alpha=0.9, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, tokens_set1_qa, width, label='QASM', 
                   color=PLOT_COLORS['orange'], alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Average Tokens (thousands)')
    max_tokens = max(max(tokens_set1_py), max(tokens_set1_qa))
    ax.set_ylim([0, max_tokens * 1.1])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.1, 1.05, '(e) Set 1: Token Usage', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom', ha='left')
    
    # Set 2 Tokens
    ax = axes[2, 1]
    bars1 = ax.bar(x - width/2, tokens_set2_py, width, label='Python', 
                   color=PLOT_COLORS['purple'], alpha=0.9, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, tokens_set2_qa, width, label='QASM', 
                   color=PLOT_COLORS['orange'], alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Average Tokens (thousands)')
    max_tokens = max(max(tokens_set2_py), max(tokens_set2_qa))
    ax.set_ylim([0, max_tokens * 1.1])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.1, 1.05, '(f) Set 2: Token Usage', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom', ha='left')
    
    # ============================================================================
    # Row 4: Token Efficiency (Perfect Match % / Avg Tokens per 1k)
    # ============================================================================
    
    # Set 1 Token Efficiency
    ax = axes[3, 0]
    bars1 = ax.bar(x - width/2, token_eff_set1_py, width, label='Python', 
                   color=PLOT_COLORS['purple'], alpha=0.9, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, token_eff_set1_qa, width, label='QASM', 
                   color=PLOT_COLORS['orange'], alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Token Efficiency')
    max_eff = max(max(token_eff_set1_py), max(token_eff_set1_qa))
    ax.set_ylim([0, max_eff * 1.1 if max_eff > 0 else 1])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.1, 1.05, '(g) Set 1: Token Efficiency', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom', ha='left')
    
    # Set 2 Token Efficiency
    ax = axes[3, 1]
    bars1 = ax.bar(x - width/2, token_eff_set2_py, width, label='Python', 
                   color=PLOT_COLORS['purple'], alpha=0.9, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, token_eff_set2_qa, width, label='QASM', 
                   color=PLOT_COLORS['orange'], alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Token Efficiency')
    max_eff = max(max(token_eff_set2_py), max(token_eff_set2_qa))
    ax.set_ylim([0, max_eff * 1.1 if max_eff > 0 else 1])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.1, 1.05, '(h) Set 2: Token Efficiency', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom', ha='left')
    
    # Add a single shared legend in the dedicated right column
    legend_ax = fig.add_subplot(gs[:, 2])
    legend_ax.axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, frameon=False, loc='upper center', fontsize=11)
    
    # Save figure
    output_dir = Path('figures/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig('figures/results/baseline_comprehensive_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/results/baseline_comprehensive_comparison.png', dpi=800, bbox_inches='tight')
    plt.close()
    
    print("\nFigure saved: figures/results/baseline_comprehensive_comparison.pdf")
    print("Figure saved: figures/results/baseline_comprehensive_comparison.png")
    
    # Print statistics
    print("\n" + "="*80)
    print("MODEL STATISTICS")
    print("="*80)
    
    for i, model in enumerate(models):
        print(f"\n{model}:")
        print(f"  Set 1:")
        print(f"    TVD          - Python: {tvd_set1_py[i]:.3f}, QASM: {tvd_set1_qa[i]:.3f}")
        print(f"    Parse Rate   - Python: {parse_set1_py[i]:.3f}, QASM: {parse_set1_qa[i]:.3f}")
        print(f"    Avg Tokens   - Python: {tokens_set1_py[i]:.2f}k, QASM: {tokens_set1_qa[i]:.2f}k")
        print(f"    Token Eff.   - Python: {token_eff_set1_py[i]:.2f}, QASM: {token_eff_set1_qa[i]:.2f}")
        print(f"  Set 2:")
        print(f"    TVD          - Python: {tvd_set2_py[i]:.3f}, QASM: {tvd_set2_qa[i]:.3f}")
        print(f"    Parse Rate   - Python: {parse_set2_py[i]:.3f}, QASM: {parse_set2_qa[i]:.3f}")
        print(f"    Avg Tokens   - Python: {tokens_set2_py[i]:.2f}k, QASM: {tokens_set2_qa[i]:.2f}k")
        print(f"    Token Eff.   - Python: {token_eff_set2_py[i]:.2f}, QASM: {token_eff_set2_qa[i]:.2f}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
