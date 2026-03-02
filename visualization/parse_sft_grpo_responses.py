from email import parser
import os
import re
import csv
import pandas as pd
from pathlib import Path
import argparse


def parse_response_file(filepath):
    """
    Parse a response file with Index markers.
    Returns a list of dictionaries with index, prompt, and response.
    """
    responses = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split by the separator line
    sections = content.split('=' * 80)
    
    current_index = None
    current_prompt = None
    current_response = None
    
    for section in sections:
        if not section.strip():
            continue
        
        # Try to extract index
        index_match = re.search(r'Index:\s*(\d+)', section)
        if index_match:
            # Save previous response if exists
            if current_index is not None and current_response is not None:
                responses.append({
                    'index': current_index,
                    'prompt': current_prompt,
                    'response': current_response.strip()
                })
            
            current_index = int(index_match.group(1))
            
            # Extract prompt
            prompt_match = re.search(r'PROMPT:\s*\n(.*?)(?=\nRESPONSE:|\Z)', section, re.DOTALL)
            if prompt_match:
                current_prompt = prompt_match.group(1).strip()
            else:
                current_prompt = None
            
            # Extract response
            response_match = re.search(r'RESPONSE:\s*\n(.*)', section, re.DOTALL)
            if response_match:
                current_response = response_match.group(1).strip()
            else:
                current_response = None
    
    # Don't forget the last response
    if current_index is not None and current_response is not None:
        responses.append({
            'index': current_index,
            'prompt': current_prompt,
            'response': current_response.strip()
        })
    
    return responses


def extract_circuit_info(prompt):
    """Extract number of qubits and gates from the prompt."""
    if not prompt:
        return None, None
    
    # Extract number of qubits
    qubit_match = re.search(r'QuantumCircuit\((\d+)\)', prompt)
    num_qubits = int(qubit_match.group(1)) if qubit_match else None
    
    # Count number of gates (lines starting with 'circuit.')
    gate_lines = [line for line in prompt.split('\n') if line.strip().startswith('circuit.')]
    num_gates = len(gate_lines)
    
    return num_qubits, num_gates


def extract_final_distribution(response):
    """Extract the final probability distribution from the response."""
    if not response:
        return None
    
    # Look for the final dictionary format
    # Pattern: {"<bitstring>": <probability>, ...}
    dict_match = re.search(r'\{["\'].*?["\']\s*:\s*[\d.]+.*?\}', response, re.DOTALL)
    if dict_match:
        return dict_match.group(0)
    
    return None


def parse_directory(directory_path, test_name):
    """
    Parse response files in a directory.
    Returns a list of response dictionaries.
    """
    all_responses = []
    
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Error: Directory {directory_path} does not exist")
        return all_responses
    
    # Find the specific response file for this test
    response_file = directory / f"{test_name}_responses.txt"
    
    if not response_file.exists():
        print(f"Response file not found: {response_file}")
        return all_responses
    
    response_files = [response_file]
    print(f"Found response file: {response_file.name}")
    
    for filepath in response_files:
        print(f"Processing {filepath.name}...")
        responses = parse_response_file(filepath)
        
        if responses:
            # Add metadata to each response
            for resp in responses:
                resp['source_file'] = filepath.name
                resp['test_name'] = test_name
                resp['directory'] = directory_path
                
                # Extract circuit information
                num_qubits, num_gates = extract_circuit_info(resp.get('prompt'))
                resp['num_qubits'] = num_qubits
                resp['num_gates'] = num_gates
                
                # Extract final distribution
                resp['final_distribution'] = extract_final_distribution(resp.get('response'))
            
            all_responses.extend(responses)
            print(f"  Extracted {len(responses)} responses")
        else:
            print(f"  No responses found")
    
    return all_responses


def save_responses(all_responses, output_dir):
    """
    Save parsed responses to individual text files and an aggregated CSV summary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Sort responses by test name and index
    sorted_responses = sorted(all_responses, key=lambda x: (x.get('test_name', ''), x.get('index', -1)))
    
    # CSV summary data
    summary_data = []
    
    for resp in sorted_responses:
        index = resp.get('index', -1)
        test_name = resp.get('test_name', 'unknown')
        
        # Create subdirectory for each test
        test_dir = output_path / test_name
        test_dir.mkdir(exist_ok=True)
        
        # Save individual response to text file
        response_file = test_dir / f"response_{index:04d}.txt"
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(f"Source: {resp.get('source_file', 'unknown')}\n")
            f.write(f"Test: {test_name}\n")
            f.write(f"Index: {index}\n")
            f.write(f"Qubits: {resp.get('num_qubits', 'N/A')}\n")
            f.write(f"Gates: {resp.get('num_gates', 'N/A')}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("PROMPT:\n")
            f.write("=" * 60 + "\n\n")
            if resp.get('prompt'):
                f.write(resp['prompt'])
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("RESPONSE:\n")
            f.write("=" * 60 + "\n\n")
            if resp.get('response'):
                f.write(resp['response'])
        
        # Add to summary
        summary_data.append({
            'test_name': test_name,
            'index': index,
            'source_file': resp.get('source_file', 'unknown'),
            'num_qubits': resp.get('num_qubits'),
            'num_gates': resp.get('num_gates'),
            'prompt_length': len(resp.get('prompt', '')),
            'response_length': len(resp.get('response', '')),
            'has_final_distribution': resp.get('final_distribution') is not None,
            'response_file': str(response_file.relative_to(output_path))
        })
    
    print(f"Saved {len(sorted_responses)} responses to {output_path}")
    
    # Save CSV summary
    if summary_data:
        csv_file = output_path / f"{prefix}_aggregated.csv"
        df = pd.DataFrame(summary_data)
        df = df.sort_values(['test_name', 'index']).reset_index(drop=True)
        df.to_csv(csv_file, index=False)
        
        print(f"Saved aggregated summary to {csv_file}")
        
        # Print statistics
        print(f"\nResponse Statistics:")
        print(f"  Total responses: {len(summary_data)}")
        print(f"  Test sets: {df['test_name'].nunique()}")
        print(f"  Responses by test:")
        for test_name, group in df.groupby('test_name'):
            print(f"    {test_name}: {len(group)} responses")
        print(f"  Average response length: {df['response_length'].mean():.0f} characters")
        print(f"  Responses with final distribution: {df['has_final_distribution'].sum()}/{len(df)}")
        
        # Qubit distribution
        if df['num_qubits'].notna().any():
            print(f"\n  Qubit distribution:")
            qubit_counts = df['num_qubits'].value_counts().sort_index()
            for qubits, count in qubit_counts.items():
                print(f"    {int(qubits)} qubits: {count} circuits")
    
    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Parse SFT+GRPO response files for random gate sets")
    parser.add_argument(
        '--target_dir',
        type=str,
    )
    parser.add_argument(
        '--test_file_prefix',
        type=str,
    )
    parser.add_argument(
        '--output_dir',
        type=str,
    )
    args = parser.parse_args()

    
    print("="*70)
    print("SFT+GRPO Response Parser - Random Gate Sets")
    print("="*70)
    

    sft_grpo_dir = args.target_dir 
    
    # Test files to process
    test_files = [
        f'{args.test_file_prefix}_1',
        f'{args.test_file_prefix}_2a',
        f'{args.test_file_prefix}_2b',
        f'{args.test_file_prefix}_3'
    ]
    
    # Parse all test files
    all_responses = []
    
    for test_name in test_files:
        print(f"\n{'='*70}")
        print(f"Processing test: {test_name}")
        print(f"{'='*70}")
        
        responses = parse_directory(sft_grpo_dir, test_name)
        
        if responses:
            all_responses.extend(responses)
            print(f"  Total responses from {test_name}: {len(responses)}")
    
    if not all_responses:
        print("\n" + "="*70)
        print("No responses found")
        print("="*70)
        return
    
    # Save aggregated responses
    print("\n" + "="*70)
    print("Saving aggregated results...")
    print("="*70)
    
    output_directory = Path(args.output_dir) 
    save_responses(all_responses, output_directory)
    

if __name__ == "__main__":
    main()
