import re
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob
from plot_style_constants import apply_plot_style, PLOT_COLORS

apply_plot_style()

def parse_log_file(log_path):
    """
    Parse RL training log file and extract metrics.
    
    Args:
        log_path: Path to log file
        
    Returns:
        data_dict: Dictionary mapping step -> {metric_name: value}
    """
    data_dict = {}
    
    # Metrics we want to extract
    metrics = [
        'response_length/max',
        'response_length/min',
        'response_length/mean',
        'response_length/clip_ratio',
        'critic/rewards/mean',
        'actor/entropy'
    ]
    
    with open(log_path, 'rb') as f:
        for line_bytes in f:
            try:
                line = line_bytes.decode('utf-8')
            except UnicodeDecodeError:
                continue
            
            # Look for step number
            step_match = re.search(r'step:(\d+)', line)
            if not step_match:
                continue
                
            step = int(step_match.group(1))
            
            if step not in data_dict:
                data_dict[step] = {}
            
            # Extract each metric
            for metric in metrics:
                # Escape special characters in metric name for regex
                escaped_metric = re.escape(metric)
                pattern = rf'{escaped_metric}:([-+]?[\d.]+(?:[eE][-+]?\d+)?)'
                match = re.search(pattern, line)
                
                if match:
                    value = float(match.group(1))
                    data_dict[step][metric] = value
    
    return data_dict


def merge_log_files(log_paths):
    """
    Parse and merge multiple log files.
    
    Args:
        log_paths: List of paths to log files
        
    Returns:
        steps: Sorted list of step numbers
        metrics_data: Dictionary mapping metric_name -> list of values
    """
    merged_data = {}
    
    for log_path in log_paths:
        print(f"Parsing log file: {log_path}")
        data_dict = parse_log_file(log_path)
        
        for step, metrics in data_dict.items():
            if step not in merged_data:
                merged_data[step] = {}
            
            for metric, value in metrics.items():
                if metric in merged_data[step] and merged_data[step][metric] != value:
                    print(f"Warning: Duplicate {metric} at step {step}. Using value from {log_path}")
                merged_data[step][metric] = value
    
    # Sort by step number
    sorted_steps = sorted(merged_data.keys())
    
    # Organize data by metric
    metric_names = ['response_length/max', 'response_length/min', 
                    'response_length/clip_ratio', 'critic/rewards/mean', 'actor/entropy']
    
    metrics_data = {metric: [] for metric in metric_names}
    
    for step in sorted_steps:
        for metric in metric_names:
            value = merged_data[step].get(metric, None)
            metrics_data[metric].append(value)
    
    return sorted_steps, metrics_data


def plot_single_metric(steps, values, metric_name, ylabel, output_path):
    """
    Plot a single metric following the style guide.
    
    Args:
        steps: List of step numbers
        values: List of metric values
        metric_name: Name of the metric (for title/label)
        ylabel: Y-axis label
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Filter out None values
    valid_indices = [i for i, v in enumerate(values) if v is not None]
    plot_steps = [steps[i] for i in valid_indices]
    plot_values = [values[i] for i in valid_indices]
    
    if not plot_values:
        print(f"Warning: No valid data for {metric_name}, skipping plot.")
        plt.close()
        return
    
    # Plot the metric
    ax.plot(plot_steps, plot_values, '-', 
            color=PLOT_COLORS['purple'], 
            linewidth=1.5)
    
    # Add markers if few data points
    if len(plot_values) < 20:
        ax.plot(plot_steps, plot_values, 'o', 
                color=PLOT_COLORS['purple'], 
                markersize=5)
    
    # Formatting
    ax.set_xlabel('Training Step')
    ax.set_ylabel(ylabel)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save figures
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    
    print(f"Saved {metric_name} plot to {output_path}")
    
    # Print summary
    print(f"\n=== {metric_name} Summary ===")
    print(f"Initial value: {plot_values[0]:.4f}")
    print(f"Final value: {plot_values[-1]:.4f}")
    print(f"Min value: {min(plot_values):.4f}")
    print(f"Max value: {max(plot_values):.4f}")
    
    plt.close()


def plot_all_metrics(steps, metrics_data, output_dir='plots'):
    """
    Generate separate plots for each metric.
    
    Args:
        steps: List of step numbers
        metrics_data: Dictionary mapping metric_name -> list of values
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define plot configurations
    plot_configs = {
        'response_length/max': {
            'ylabel': 'Max Response Length',
            'filename': 'rl_response_length_max.pdf'
        },
        'response_length/min': {
            'ylabel': 'Min Response Length',
            'filename': 'rl_response_length_min.pdf'
        },
        'response_length/clip_ratio': {
            'ylabel': 'Response Length Clip Ratio',
            'filename': 'rl_response_clip_ratio.pdf'
        },
        'critic/rewards/mean': {
            'ylabel': 'Mean Reward',
            'filename': 'rl_critic_rewards_mean.pdf'
        },
        'actor/entropy': {
            'ylabel': 'Actor Entropy',
            'filename': 'rl_actor_entropy.pdf'
        }
    }
    
    # Generate each plot
    for metric_name, config in plot_configs.items():
        if metric_name in metrics_data:
            output_path = os.path.join(output_dir, config['filename'])
            plot_single_metric(
                steps=steps,
                values=metrics_data[metric_name],
                metric_name=metric_name,
                ylabel=config['ylabel'],
                output_path=output_path
            )


def plot_combined_overview(steps, metrics_data, output_path='plots/rl_training_overview.pdf'):
    """
    Create a multi-panel overview figure with all five metrics.
    
    Args:
        steps: List of step numbers
        metrics_data: Dictionary mapping metric_name -> list of values
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # Plot configurations
    plot_configs = [
        ('response_length/max', 'Max Response Length'),
        ('response_length/min', 'Min Response Length'),
        ('response_length/clip_ratio', 'Clip Ratio'),
        ('critic/rewards/mean', 'Mean Reward'),
        ('actor/entropy', 'Actor Entropy'),
    ]
    
    for idx, (metric_name, ylabel) in enumerate(plot_configs):
        ax = axes[idx]
        values = metrics_data[metric_name]
        
        # Filter out None values
        valid_indices = [i for i, v in enumerate(values) if v is not None]
        plot_steps = [steps[i] for i in valid_indices]
        plot_values = [values[i] for i in valid_indices]
        
        if plot_values:
            ax.plot(plot_steps, plot_values, '-', 
                    color=PLOT_COLORS['purple'], 
                    linewidth=2)
            
            if len(plot_values) < 20:
                ax.plot(plot_steps, plot_values, 'o', 
                        color=PLOT_COLORS['purple'], 
                        markersize=4)
        
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add panel label
        ax.text(-0.15, 1.05, f'({chr(97+idx)})', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')
    
    # Hide the 6th subplot
    axes[5].set_visible(False)
    
    plt.tight_layout()
    
    # Save figures
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    
    print(f"\nSaved combined overview to {output_path}")
    plt.close()


# Main execution
if __name__ == "__main__":
    # Check if directory path provided
    if len(sys.argv) < 2:
        print("Usage: python plot_rl_training_logs.py <directory_path>")
        print("\nExample:")
        print("  python plot_rl_training_logs.py ./rl_logs/")
        print("\nThe script will automatically find and use all .log and .out files in the directory.")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    
    # Check if directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        sys.exit(1)
    
    # Find all log and .out files in the directory
    log_patterns = [
        os.path.join(directory_path, '*.log'),
        os.path.join(directory_path, '*.out'),
    ]
    
    log_files = []
    for pattern in log_patterns:
        log_files.extend(glob.glob(pattern))
    
    # Sort files for consistent ordering
    log_files = sorted(log_files)
    
    if not log_files:
        print(f"Error: No log files (.log or .out) found in directory '{directory_path}'")
        sys.exit(1)
    
    print(f"Found {len(log_files)} log file(s) in '{directory_path}':")
    for log_file in log_files:
        print(f"  - {os.path.basename(log_file)}")
    print()
    
    # Parse and merge log files
    print(f"Processing {len(log_files)} log file(s)...")
    steps, metrics_data = merge_log_files(log_files)
    
    print(f"\n=== Parsing Summary ===")
    print(f"Total unique steps: {len(steps)}")
    if len(steps) > 0:
        print(f"Step range: {min(steps)} - {max(steps)}")
    
    for metric_name, values in metrics_data.items():
        valid_count = len([v for v in values if v is not None])
        print(f"Found {valid_count} entries for {metric_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot individual metrics
    plot_all_metrics(steps, metrics_data)
    
    # Plot combined overview
    plot_combined_overview(steps, metrics_data)
    
    print("\n=== All plots generated successfully ===")