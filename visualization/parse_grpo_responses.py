import os
import re
import csv
import pandas as pd
from pathlib import Path
import argparse


def remove_ansi_codes(text):
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def remove_task_runner_prefix(text):
    return re.sub(r'\([A-Za-z]+ pid=\d+\)\s*', '', text)


def extract_step_number(text):
    """
    Extract step number from lines like:
    step:0 - val-core/quantum_circuits/state_pred/reward/mean@1:0.1288
    """
    match = re.search(r'step:(\d+)\s+-\s+val-core', text)
    if match:
        return int(match.group(1))
    return None


def parse_model_responses(filepath, directory_name):
    """
    Parse a single log file and extract model responses.
    Returns a list of dictionaries with step number, response, and score.
    """
    responses = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Remove ANSI color codes and TaskRunner prefixes
    content = remove_ansi_codes(content)
    content = remove_task_runner_prefix(content)
    
    # Split by lines to find step markers first
    lines = content.split('\n')
    
    # Find all step markers and their positions
    step_info = []
    for i, line in enumerate(lines):
        if re.search(r'step:\d+\s+-\s+val-core', line):
            step_num = extract_step_number(line)
            if step_num is not None:
                step_info.append((i, step_num, line))
    
    # Now find responses between [prompt] and [score]
    # Pattern to find response blocks
    response_pattern = r'\[prompt\]\s*user\s*(.*?)\[score\]\s*([\d.]+)'
    
    matches = list(re.finditer(response_pattern, content, re.DOTALL))
    
    print(f"  Found {len(matches)} response blocks")
    print(f"  Found {len(step_info)} step markers")
    
    # Try to associate responses with steps
    for match in matches:
        response_text = match.group(1).strip()
        score = match.group(2).strip()
        
        # Find the position of this match in the original text
        match_start = match.start()
        
        # Find lines before this match to locate the nearest step marker
        lines_before = content[:match_start].split('\n')
        
        # Search backwards for a step marker
        associated_step = None
        for line in reversed(lines_before[-100:]):  # Check last 100 lines
            step_num = extract_step_number(line)
            if step_num is not None:
                associated_step = step_num
                break
        
        # Also check for [ground_truth] to extract that if present
        ground_truth = None
        remaining_text = content[match.end():match.end()+500]
        gt_match = re.search(r'\[ground_truth\]\s*(\{.*?\})', remaining_text)
        if gt_match:
            ground_truth = gt_match.group(1)
        
        responses.append({
            'step': associated_step,
            'original_step': associated_step,  # Will be updated during renumbering
            'directory': directory_name,
            'response': response_text,
            'score': score,
            'ground_truth': ground_truth
        })
    
    return responses


def parse_directory(directory_path, directory_name):
    """
    Parse all .out files in a directory.
    Returns a list of response dictionaries.
    """
    all_responses = []
    
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Error: Directory {directory_path} does not exist")
        return all_responses
    
    # Find all .out files
    out_files = list(directory.glob('*.out'))
    
    if not out_files:
        print(f"No .out files found in {directory_path}")
        return all_responses
    
    print(f"Found {len(out_files)} .out file(s)")
    
    for filepath in out_files:
        print(f"Processing {filepath.name}...")
        responses = parse_model_responses(filepath, directory_name)
        
        if responses:
            # Add source filename to each response
            for resp in responses:
                resp['source_file'] = filepath.name
            all_responses.extend(responses)
            print(f"  Extracted {len(responses)} responses")
        else:
            print(f"  No responses found")
    
    return all_responses


def renumber_responses(responses, start_step, step_mapping):
    """
    Renumber responses based on step mapping from metrics CSV.
    step_mapping is a dict of {original_step: new_step}
    """
    renumbered = []
    
    for resp in responses:
        resp_copy = resp.copy()
        original_step = resp['step']
        
        if original_step is not None and original_step in step_mapping:
            resp_copy['step'] = step_mapping[original_step]
        elif original_step is None:
            resp_copy['step'] = None
        else:
            # If not in mapping, keep original or mark as unmapped
            resp_copy['step'] = original_step
        
        renumbered.append(resp_copy)
    
    return renumbered


def save_responses(all_responses, output_dir, prefix="grpo"):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Sort responses by step for better organization
    sorted_responses = sorted(all_responses, key=lambda x: (x['step'] if x['step'] is not None else -1))
    
    # CSV summary data
    summary_data = []
    
    for resp in sorted_responses:
        step = resp['step']
        original_step = resp.get('original_step', step)
        directory = resp.get('directory', 'unknown')
        
        step_str = f"step_{step:04d}" if step is not None else f"unknown"
        
        # Save individual response to text file
        response_file = output_path / f"{prefix}_{step_str}_response.txt"
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(f"Source: {resp['source_file']}\n")
            f.write(f"Directory: {directory}\n")
            f.write(f"Step (renumbered): {step}\n")
            f.write(f"Step (original): {original_step}\n")
            f.write(f"Score: {resp['score']}\n")
            if resp['ground_truth']:
                f.write(f"Ground Truth: {resp['ground_truth']}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("RESPONSE:\n")
            f.write("=" * 60 + "\n\n")
            f.write(resp['response'])
        
        # Add to summary
        summary_data.append({
            'step': step,
            'original_step': original_step,
            'directory': directory,
            'source_file': resp['source_file'],
            'score': resp['score'],
            'ground_truth': resp['ground_truth'],
            'response_length': len(resp['response']),
            'response_file': response_file.name
        })
    
    print(f"Saved {len(sorted_responses)} responses to {output_path}")
    
    # Save CSV summary
    if summary_data:
        csv_file = output_path / f"{prefix}_responses_aggregated.csv"
        df = pd.DataFrame(summary_data)
        df = df.sort_values('step').reset_index(drop=True)
        df.to_csv(csv_file, index=False)
        
        print(f"Saved aggregated summary to {csv_file}")
        
        # Print statistics
        print(f"\nResponse Statistics:")
        print(f"  Total responses: {len(summary_data)}")
        if step is not None:
            print(f"  Step range: {df['step'].min()} to {df['step'].max()}")
        print(f"  Unique directories: {df['directory'].nunique()}")
        print(f"  Average response length: {df['response_length'].mean():.0f} characters")
        print(f"  Average score: {pd.to_numeric(df['score'], errors='coerce').mean():.4f}")
    
    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Parse GRPO model responses from random_subset .out files")
    parser.add_argument(
        '--dir-config',
        nargs='+',
        help=(
            "Directory processing config in format dir_name:directory_path. "
            "Repeat this flag for multiple directories. "
            "Example: --dir-config random_subset:random_subset random_subset_30:random_subset_30"
        ),
    )
    parser.add_argument(
        '--metrics-csv',
        type=str,
        default='grpo_random_metrics_aggregated.csv',
        help='Path to metrics CSV file for step renumbering'
    )
    args = parser.parse_args()
    
    print("="*70)
    print("GRPO Response Parser - Random Subset Directories")
    print("="*70)
    
    # Configuration for random_subset directories
    directories = []
    for config in args.dir_config:
        dir_name, dir_path = config.split(':')
        directories.append((dir_path, dir_name))

    
    # Load step mapping from metrics CSV if available
    metrics_csv = args.metrics_csv
    step_mappings = {}
    
    if os.path.exists(metrics_csv):
        print(f"\nLoading step mappings from {metrics_csv}...")
        metrics_df = pd.read_csv(metrics_csv)
        
        # Create step mapping for each directory
        for dir_path, dir_name in directories:
            dir_data = metrics_df[metrics_df['source_file'].str.contains(dir_name, na=False)]
            if len(dir_data) > 0:
                # Map original_step to renumbered step
                mapping = dict(zip(dir_data['original_step'], dir_data['step']))
                step_mappings[dir_name] = mapping
                print(f"  {dir_name}: {len(mapping)} step mappings loaded")
    else:
        print(f"\nWarning: {metrics_csv} not found. Step renumbering may not be accurate.")
    
    # Parse all directories
    all_responses = []
    
    for dir_path, dir_name in directories:
        print(f"\n{'='*70}")
        print(f"Processing directory: {dir_path}")
        print(f"{'='*70}")
        
        if not os.path.exists(dir_path):
            print(f"  Warning: Directory not found, skipping...")
            continue
        
        responses = parse_directory(dir_path, dir_name)
        
        if responses:
            # Apply step renumbering if mapping exists
            if dir_name in step_mappings:
                print(f"  Applying step renumbering using mapping...")
                responses = renumber_responses(responses, 0, step_mappings[dir_name])
            
            all_responses.extend(responses)
            print(f"  Total responses from this directory: {len(responses)}")
    
    if not all_responses:
        print("\n" + "="*70)
        print("No responses found in any directory")
        print("="*70)
        return
    
    # Save aggregated responses
    print("\n" + "="*70)
    print("Saving aggregated results...")
    print("="*70)
    
    output_directory = "grpo_parsed_responses_random"
    save_responses(all_responses, output_directory, prefix="grpo_random")
    
    print("\n" + "="*70)
    print("Done! Random subset responses parsed and aggregated.")
    print("="*70)


if __name__ == "__main__":
    main()
