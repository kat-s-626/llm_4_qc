import os
import re
import pandas as pd
from pathlib import Path
import argparse


def parse_dir_config(config_str):
    """
    Parse directory config in format: dir_name:start_step:step_offset
    - start_step can be 'auto' to use the computed next step.
    Example: sft_grpo_50:auto:372
    """
    parts = config_str.split(':')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid --dir-config '{config_str}'. Expected format: dir_name:start_step:step_offset"
        )

    dir_name, start_step_raw, step_offset_raw = parts

    if start_step_raw.lower() == 'auto':
        start_step = None
    else:
        try:
            start_step = int(start_step_raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid start_step in --dir-config '{config_str}'. Use integer or 'auto'."
            ) from exc

    try:
        step_offset = int(step_offset_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid step_offset in --dir-config '{config_str}'. Use integer."
        ) from exc

    return dir_name, start_step, step_offset

def remove_ansi_codes(text):
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)

def remove_task_runner_prefix(text):
    return re.sub(r'\([A-Za-z]+ pid=\d+\)\s*', '', text)

def parse_step_metrics(content):
    """
    Parse metrics from a step block in the log file
    Returns dict of metric_name: value
    """
    metrics = {}
    
    # Clean the content
    content = remove_ansi_codes(content)
    content = remove_task_runner_prefix(content)
    
    # The metrics are formatted as key:value pairs separated by " - "
    # Example: step:920 - global_seqlen/min:5941 - critic/rewards/mean:0.14374999701976776
    
    # Split by " - " to get individual metric pairs
    parts = content.split(' - ')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Match pattern: metric_name:value (with various metric name formats)
        # Allow for slashes, underscores, and hyphens in metric names
        match = re.match(r'([a-zA-Z_/\-@]+):([0-9.eE+-]+)', part)
        if match:
            metric_name = match.group(1)
            value_str = match.group(2)
            try:
                value = float(value_str)
                metrics[metric_name] = value
            except ValueError:
                continue
    
    return metrics

def parse_grpo_log_file(filepath):
    """
    Parse a single .out file and extract all step metrics
    Returns list of dicts, each containing step number and metrics
    """
    print(f"  Parsing: {os.path.basename(filepath)}")
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Clean ANSI codes and TaskRunner prefixes
    content = remove_ansi_codes(content)
    content = remove_task_runner_prefix(content)
    
    # The metrics are on single lines with format: step:X - metric1:value1 - metric2:value2 ...
    # Use regex to find lines containing "step:\d+" followed by metrics
    step_pattern = r'step:(\d+)\s*-\s*([^\n]+)'
    matches = re.finditer(step_pattern, content)
    
    results = []
    for match in matches:
        step_num = int(match.group(1))
        metrics_line = match.group(2)
        
        # Parse metrics from this line
        metrics = parse_step_metrics(f"step:{step_num} - {metrics_line}")
        
        if metrics:  # Only add if we found metrics
            result = {
                'step': step_num,
                'source_file': os.path.basename(filepath)
            }
            result.update(metrics)
            results.append(result)
    
    return results

def renumber_steps(df, start_step, step_offset=0):
    """
    Renumber steps to be sequential starting from start_step
    Keeps original_step column for reference
    step_offset: offset to add to original steps (used for grover_sft_grpo_50)
    """
    df = df.copy()
    df['original_step'] = df['step']
    
    # Sort by original step to maintain order
    df = df.sort_values('original_step').reset_index(drop=True)
    
    # Create new sequential step numbers
    # For curriculum learning, we add step_offset to account for the restart
    if step_offset > 0:
        # The logs restart from 0/1, so we need to add the offset
        df['step'] = df['original_step'] + step_offset
    else:
        df['step'] = range(start_step, start_step + len(df))
    
    # Reorder columns to put step first
    cols = ['step', 'original_step'] + [col for col in df.columns if col not in ['step', 'original_step']]
    df = df[cols]
    
    return df

def process_single_directory(directory_path, output_dir, start_step=0, step_offset=0):
    """
    Process all .out files in a single directory
    Returns DataFrame and the next start_step for the next directory
    step_offset: offset to add to original steps (used for curriculum learning)
    """
    print(f"\nProcessing directory: {directory_path}")
    
    if not os.path.exists(directory_path):
        print(f"  Warning: Directory not found: {directory_path}")
        return None, start_step
    
    # Find all .out files
    out_files = sorted(Path(directory_path).glob('*.out'))
    
    if not out_files:
        print(f"  Warning: No .out files found in {directory_path}")
        return None, start_step
    
    print(f"  Found {len(out_files)} .out files")
    
    # Parse all files
    all_results = []
    for out_file in out_files:
        results = parse_grpo_log_file(str(out_file))
        all_results.extend(results)
    
    if not all_results:
        print(f"  Warning: No metrics found in any files")
        return None, start_step
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Remove duplicates (keep last occurrence of each step)
    # Sort by source_file to ensure consistent ordering
    df = df.sort_values(['step', 'source_file'])
    df = df.drop_duplicates(subset=['step'], keep='last')
    
    print(f"  Found {len(df)} unique steps")
    print(f"  Original step range: {df['step'].min()} to {df['step'].max()}")
    
    # Renumber steps sequentially
    df = renumber_steps(df, start_step, step_offset)
    
    print(f"  Renumbered step range: {df['step'].min()} to {df['step'].max()}")
    
    # Save individual directory results
    output_file = os.path.join(output_dir, f"{os.path.basename(directory_path)}_metrics.csv")
    df.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    
    # Return DataFrame and next start step
    next_start_step = df['step'].max() + 1
    return df, next_start_step

def main():
    parser = argparse.ArgumentParser(description='Parse GRPO metrics from SFT-GRPO directories and aggregate into CSV.')
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory where parsed CSVs will be saved.',
    )
    parser.add_argument(
        'directories',
        nargs='*',
    )
    parser.add_argument(
        '--dir-config',
        action='append',
        type=parse_dir_config,
        help=(
            "Directory processing config in format dir_name:start_step:step_offset. "
            "Repeat this flag for multiple directories. "
        ),
    )

    args = parser.parse_args()

    print("="*70)
    print("GRPO Metrics Parser - SFT-GRPO Directories")
    print("="*70)
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Define directories to process in order
    # sft_grpo: steps 0-372
    # sft_grpo_50: logs show steps starting from 0/1, but actually continues from 373
    directories = args.dir_config 
    
    all_dataframes = []
    next_start = 0
    
    # Process each directory
    for dir_name, start_step, step_offset in directories:
        # Use calculated start_step if not specified
        if start_step is None:
            start_step = next_start
        
        df, next_start = process_single_directory(dir_name, output_dir, start_step, step_offset)
        
        if df is not None:
            all_dataframes.append(df)
    
    # Aggregate all results
    if all_dataframes:
        print("\n" + "="*70)
        print("Aggregating results from all directories...")
        
        aggregated_df = pd.concat(all_dataframes, ignore_index=True)
        aggregated_df = aggregated_df.sort_values('step').reset_index(drop=True)
        
        # Remove any duplicate steps (shouldn't happen, but just in case)
        aggregated_df = aggregated_df.drop_duplicates(subset=['step'], keep='last')
        
        print(f"Total unique steps: {len(aggregated_df)}")
        print(f"Step range: {aggregated_df['step'].min()} to {aggregated_df['step'].max()}")
        
        # Save aggregated results
        output_file = 'sft_grpo_metrics_aggregated.csv'
        aggregated_df.to_csv(output_file, index=False)
        
       
        print(f"Saved aggregated results: {output_file}")
        
        # Print summary statistics
        print("\n" + "="*70)
        print("Summary Statistics")
        print("="*70)
        
        for dir_name, _, _ in directories:
            dir_data = aggregated_df[aggregated_df['source_file'].str.contains(dir_name)]
            if len(dir_data) > 0:
                print(f"\n{dir_name}:")
                print(f"  Steps: {len(dir_data)} (range: {dir_data['step'].min()}-{dir_data['step'].max()})")
                
                # Show a few key metrics if available
                if 'critic/rewards/mean' in dir_data.columns:
                    mean_reward = dir_data['critic/rewards/mean'].mean()
                    print(f"  Mean critic reward: {mean_reward:.4f}")
                
                if 'response_length/mean' in dir_data.columns:
                    mean_length = dir_data['response_length/mean'].mean()
                    print(f"  Mean response length: {mean_length:.1f}")
        
        print("\n" + "="*70)
        print("Parsing complete!")
        print("="*70)
        
        # Print column names for reference
        print(f"\nAvailable metrics ({len(aggregated_df.columns)} columns):")
        metrics = [col for col in aggregated_df.columns if col not in ['step', 'original_step', 'source_file']]
        for i, metric in enumerate(sorted(metrics), 1):
            print(f"  {i}. {metric}")
    
    else:
        print("\nNo data was successfully parsed from any directory.")

if __name__ == '__main__':
    main()
