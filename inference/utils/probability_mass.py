import json
import numpy as np
from pathlib import Path
import argparse

def verify_top15_coverage(dataset_path):
    """
    Verify that top-15 states capture >99.9% of probability mass.
    
    Args:
        dataset_path: Path to dataset file (parquet, json, or jsonl)
    
    Returns:
        Dictionary with statistics
    """
    # Load dataset based on file type
    if dataset_path.endswith('.parquet'):
        import pandas as pd
        df = pd.read_parquet(dataset_path)
        data = df.to_dict('records')
    elif dataset_path.endswith('.json'):
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")
    
    coverage_stats = []
    below_threshold_999 = []
    below_threshold_99 = []
    below_threshold_95 = []
    
    for idx, example in enumerate(data):
        # Parse ground truth probabilities
        ground_truth = example['ground_truth']
        
        # Handle both dict and string formats
        if isinstance(ground_truth, str):
            probs_dict = json.loads(ground_truth)
        else:
            probs_dict = ground_truth
        
        # Sort by probability (descending)
        sorted_states = sorted(probs_dict.items(), 
                              key=lambda x: x[1], 
                              reverse=True)
        
        # Calculate cumulative probability for top-15
        if len(sorted_states) >= 15:
            top15_prob = sum(prob for _, prob in sorted_states[:15])
        else:
            # If fewer than 15 states, sum all
            top15_prob = sum(prob for _, prob in sorted_states)
        
        coverage_stats.append(top15_prob)
        
        # Track examples below thresholds
        if top15_prob < 0.95:
            below_threshold_95.append({
                'index': idx,
                'coverage': top15_prob,
                'total_states': len(sorted_states),
                'top15_states': sorted_states[:15]
            })
        
        if top15_prob < 0.99:
            below_threshold_99.append({
                'index': idx,
                'coverage': top15_prob,
                'total_states': len(sorted_states),
                'top15_states': sorted_states[:15]
            })
        
        if top15_prob < 0.999:
            below_threshold_999.append({
                'index': idx,
                'coverage': top15_prob,
                'total_states': len(sorted_states),
                'top15_states': sorted_states[:15]
            })
    
    # Compute statistics
    coverage_array = np.array(coverage_stats)
    
    results = {
        'total_examples': len(data),
        'mean_coverage': coverage_array.mean(),
        'median_coverage': np.median(coverage_array),
        'min_coverage': coverage_array.min(),
        'max_coverage': coverage_array.max(),
        'std_coverage': coverage_array.std(),
        'percentile_99': np.percentile(coverage_array, 1),  # 1st percentile (worst 1%)
        'examples_below_95': len(below_threshold_95),
        'percent_below_95': (len(below_threshold_95) / len(data)) * 100,
        'examples_below_99': len(below_threshold_99),
        'percent_below_99': (len(below_threshold_99) / len(data)) * 100,
        'examples_below_999': len(below_threshold_999),
        'percent_below_999': (len(below_threshold_999) / len(data)) * 100,
        'examples_at_100': np.sum(coverage_array == 1.0),
        'percent_at_100': (np.sum(coverage_array == 1.0) / len(data)) * 100,
    }
    
    return results, below_threshold_999, below_threshold_99, below_threshold_95

def print_results(results, below_threshold_999, below_threshold_99, below_threshold_95, show_details=False):
    """Print formatted results."""
    print("=" * 70)
    print("TOP-15 PROBABILITY COVERAGE ANALYSIS")
    print("=" * 70)
    print(f"\nDataset Size: {results['total_examples']:,} examples")
    print(f"\nCoverage Statistics:")
    print(f"  Mean:     {results['mean_coverage']:.6f} ({results['mean_coverage']*100:.4f}%)")
    print(f"  Median:   {results['median_coverage']:.6f} ({results['median_coverage']*100:.4f}%)")
    print(f"  Std Dev:  {results['std_coverage']:.6f}")
    print(f"  Min:      {results['min_coverage']:.6f} ({results['min_coverage']*100:.4f}%)")
    print(f"  Max:      {results['max_coverage']:.6f} ({results['max_coverage']*100:.4f}%)")
    print(f"  1st %ile: {results['percentile_99']:.6f} ({results['percentile_99']*100:.4f}%)")
    
    print(f"\nThreshold Analysis:")
    print(f"  Examples with 100% coverage: {results['examples_at_100']:,} ({results['percent_at_100']:.2f}%)")
    print(f"  Examples below 99.9%: {results['examples_below_999']:,} ({results['percent_below_999']:.2f}%)")
    print(f"  Examples below 99.0%: {results['examples_below_99']:,} ({results['percent_below_99']:.2f}%)")
    print(f"  Examples below 95.0%: {results['examples_below_95']:,} ({results['percent_below_95']:.2f}%)")
    
    # Verification of claim
    print(f"\n" + "=" * 70)
    if results['mean_coverage'] > 0.999:
        print(f"✓ CLAIM VERIFIED: Top-15 captures >{results['mean_coverage']*100:.4f}% on average")
    else:
        print(f"✗ CLAIM NOT VERIFIED: Top-15 captures only {results['mean_coverage']*100:.4f}% on average")
    print("=" * 70)
    
    # Show examples below threshold if requested
    if show_details and below_threshold_999:
        print(f"\nExamples Below 99.9% Threshold ({len(below_threshold_999)} total):")
        print("-" * 70)
        for i, ex in enumerate(below_threshold_999[:10]):  # Show first 10
            print(f"\nExample {ex['index']}:")
            print(f"  Coverage: {ex['coverage']:.6f} ({ex['coverage']*100:.4f}%)")
            print(f"  Total non-zero states: {ex['total_states']}")
            print(f"  Top 5 states: {ex['top15_states'][:5]}")
        if len(below_threshold_999) > 10:
            print(f"\n  ... and {len(below_threshold_999) - 10} more examples")
    
    if show_details and below_threshold_99:
        print(f"\nExamples Below 99.0% Threshold ({len(below_threshold_99)} total):")
        print("-" * 70)
        for i, ex in enumerate(below_threshold_99[:10]):  # Show first 10
            print(f"\nExample {ex['index']}:")
            print(f"  Coverage: {ex['coverage']:.6f} ({ex['coverage']*100:.4f}%)")
            print(f"  Total non-zero states: {ex['total_states']}")
            print(f"  Top 5 states: {ex['top15_states'][:5]}")
        if len(below_threshold_99) > 10:
            print(f"\n  ... and {len(below_threshold_99) - 10} more examples")
    
    if show_details and below_threshold_95:
        print(f"\nExamples Below 95.0% Threshold ({len(below_threshold_95)} total):")
        print("-" * 70)
        for i, ex in enumerate(below_threshold_95[:10]):  # Show first 10
            print(f"\nExample {ex['index']}:")
            print(f"  Coverage: {ex['coverage']:.6f} ({ex['coverage']*100:.4f}%)")
            print(f"  Total non-zero states: {ex['total_states']}")
            print(f"  Top 5 states: {ex['top15_states'][:5]}")
        if len(below_threshold_95) > 10:
            print(f"\n  ... and {len(below_threshold_95) - 10} more examples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify top-15 probability coverage')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset file')
    parser.add_argument('--details', action='store_true', 
                       help='Show detailed information about examples below threshold')
    
    args = parser.parse_args()
    
    # Run analysis
    results, below_threshold_999, below_threshold_99, below_threshold_95 = verify_top15_coverage(args.dataset_path)
    
    # Print results
    print_results(results, below_threshold_999, below_threshold_99, below_threshold_95, show_details=args.details)
    
    # Export results to JSON
    output_path = Path(args.dataset_path).parent / 'top15_coverage_analysis.json'
    with open(output_path, 'w') as f:
        json.dump({
            'statistics': results,
            'examples_below_999_threshold': below_threshold_999,
            'examples_below_99_threshold': below_threshold_99,
            'examples_below_95_threshold': below_threshold_95
        }, f)
    
    print(f"\nDetailed results saved to: {output_path}")