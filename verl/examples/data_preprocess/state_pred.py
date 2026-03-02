import argparse
import os
import json
import sys
import datasets
from typing import Dict, List, Any
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from constants import DatasetColumns

CIRCUIT_HASH = DatasetColumns.CIRCUIT_HASH
CIRCUIT_STRING = DatasetColumns.CIRCUIT_STRING
COMPLETION = DatasetColumns.COMPLETION
DATA_SOURCE = DatasetColumns.DATA_SOURCE
EXTRA_INFO = DatasetColumns.EXTRA_INFO
FORMATTED_COMPLETION = DatasetColumns.FORMATTED_COMPLETION
FORMATTED_PROMPT = DatasetColumns.FORMATTED_PROMPT
MSB = DatasetColumns.MSB_MEASUREMENT_PROBABILITIES
LSB = DatasetColumns.LSB_MEASUREMENT_PROBABILITIES
INDEX = DatasetColumns.INDEX
N_QUBITS = DatasetColumns.N_QUBITS
NUM_QUBITS = DatasetColumns.NUM_QUBITS
PROBABILITY_DISTRIBUTION = DatasetColumns.PROBABILITY_DISTRIBUTION
PROMPT = DatasetColumns.PROMPT
SPLIT = DatasetColumns.SPLIT
CIRCUIT_DEPTH = DatasetColumns.CIRCUIT_DEPTH

def load_json_lines(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON lines file where each line is a valid JSON object.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data


def format_prompt(data_item: Dict[str, Any], eos_token: str = "") -> Dict[str, str]:
    if MSB in data_item and data_item[MSB] is not None:
        """
        This template uses simplified QASM without headers/measurements.
        """
        # Extract required fields
        probability_distribution = data_item.get(MSB, {})
        if type(probability_distribution) is str:
            probability_distribution = json.loads(probability_distribution)
        
        # Get the hash from data_item and find matching entry in python_set
        
        circuit_str = data_item.get(CIRCUIT_STRING, "")
        
        instruction = (
            "Simulate this quantum circuit. Then provide the probability distribution of the measurement outcomes "
            "as: {\"<bitstring>\": <probability>, ...}"
            "Requirements: "
            "- Probabilities rounded to 3 decimal places "
            "- Sort by probability (descending) "
            "- Only include non-zero probabilities "
            "- Top 15 states only if >15 non-zero probabilities "
        )
        # Format the prompt
        target_circuit = f"\nQuantum Circuit: \n{circuit_str}"
        # target_circuit = f"\n{circuit_str}"
        formatted_prompt = f"{instruction}{target_circuit}\n\n"

        formatted_completion = data_item.get(CIRCUIT_STRING, "")
        
        # Format the completion (the expected output)
        # Ensure we only keep top-15 if needed
        if len(probability_distribution) > 15:
            sorted_probs = dict(sorted(probability_distribution.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:15])
            formatted_completion += f"{json.dumps(sorted_probs)}{eos_token}"
        else:
            formatted_completion += f"{json.dumps(probability_distribution)}{eos_token}"
    else: # For SFT_GRPO training
        if "problem" in data_item and data_item["problem"] and "generated_solution" in data_item and data_item["generated_solution"]:
            # For openMathReasoning format
            return {
                PROMPT: data_item.get("problem", ""),
                COMPLETION: data_item.get("generated_solution", "") + eos_token
            }
        else:
            # For openCodeReasoning format
            return {
                PROMPT: data_item.get("input", ""),
                COMPLETION: data_item.get("output", "") + eos_token
            }
    
    return {
        PROMPT: formatted_prompt,
        COMPLETION: formatted_completion
    }


def process_dataset(data: List[Dict[str, Any]], split: str, data_source: str) -> List[Dict[str, Any]]:
    """
    Process the dataset into the format similar to gsm8k.py
    """
    processed_data = []

    for idx, item in enumerate(data):
        formatted = format_prompt(item)
        
        # Extract fields for extra_info
        num_qubits = item.get(NUM_QUBITS, None)
        if num_qubits is None:
            num_qubits = item.get(N_QUBITS, None)
        msb = item.get(MSB, {})
        lsb = item.get(LSB, {})
        circuit_str = item.get(CIRCUIT_STRING, "")
        circuit_depth = item.get(CIRCUIT_DEPTH, None)
        circuit_hash = item.get(CIRCUIT_HASH, "")

        
        processed_item = {
            DATA_SOURCE: data_source,
            PROMPT: formatted[PROMPT],
            COMPLETION: formatted[COMPLETION],
            EXTRA_INFO: {
                SPLIT: split,
                INDEX: idx,
                CIRCUIT_HASH: circuit_hash,
                NUM_QUBITS: num_qubits,
                CIRCUIT_STRING: circuit_str,
                CIRCUIT_DEPTH: circuit_depth,
                MSB: json.dumps(msb),
                LSB: json.dumps(lsb),
                FORMATTED_PROMPT: formatted[PROMPT],
                FORMATTED_COMPLETION: formatted[COMPLETION]
            },
        }
        
        processed_data.append(processed_item)
    
    return processed_data


def main():
    parser = argparse.ArgumentParser(description="Process quantum circuit dataset to parquet format")
    parser.add_argument("--input_file", required=True, help="Path to input JSON lines file")
    parser.add_argument("--local_dir", default="~/data/state_pred", help="Local directory to save processed data")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory to copy data to (optional)")
    parser.add_argument("--train_split", type=float, default=0.9, help="Fraction of data for training (default: 0.9)")
    parser.add_argument("--data_source", default="quantum_circuits/state_pred_reasoning", help="Data source identifier")
    parser.add_argument("--shuffle", action="store_true", help="Whether to shuffle the training split")
    args = parser.parse_args()
    
    # Expand user path
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    # Load the JSON lines data
    print(f"Loading data from {args.input_file}...")
    data = load_json_lines(args.input_file)
    print(f"Loaded {len(data)} entries")

    if args.train_split < 1.0:
        split_idx = int(len(data) * args.train_split)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        print(f"Split data: {len(train_data)} train, {len(test_data)} test")
    else:
        train_data = data
        test_data = []
        print(f"Using full dataset for training: {len(train_data)} train, 0 test")

    print("Processing training data...")
    processed_train = process_dataset(train_data, "train", args.data_source)

    seed = 42
    if args.shuffle:
        random.seed(seed)
        random.shuffle(processed_train)
        print("Shuffled the training data")

    train_dataset = datasets.Dataset.from_list(processed_train)
    train_path = os.path.join(local_dir, "train.parquet")
    print(f"Saving training data to {train_path}...")
    train_dataset.to_parquet(train_path)

    if args.train_split < 1.0:
        print("Processing test data...")
        processed_test = process_dataset(test_data, "test", args.data_source)
        test_dataset = datasets.Dataset.from_list(processed_test)
        test_path = os.path.join(local_dir, "test.parquet")
        print(f"Saving test data to {test_path}...")
        test_dataset.to_parquet(test_path)
    
    print("Processing complete!")
    
    # Optional: Copy to HDFS if specified
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs
            print(f"Copying to HDFS: {args.hdfs_dir}")
            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
            print("HDFS copy complete!")
        except ImportError:
            print("Warning: verl.utils.hdfs_io not available, skipping HDFS copy")
        except Exception as e:
            print(f"Error copying to HDFS: {e}")
    
    # Print sample of processed data for verification
    print("\nSample processed entry:")
    print(json.dumps(processed_train[0]))


if __name__ == "__main__":
    main()