import argparse
import os
import json
import datasets
from typing import Dict, List, Any
import random
from config.constants import (
    DATASET_NUM_QUBITS,
    DATASET_CIRCUIT_DEPTH,
    DATASET_GATES_LIST,
    DATASET_CIRCUIT_HASH,
    DATASET_LSB_MEASUREMENT_PROBABILITIES,
    DATASET_MSB_MEASUREMENT_PROBABILITIES,
    DATASET_PYTHON_CODE,
    DATASET_NL_DESCRIPTION,
    DATASET_EXTRA_INFO,
    EXTRA_INFO_SPLIT,
    EXTRA_INFO_INDEX,
    EXTRA_INFO_NUM_GATES,
)


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
    """
    This template uses simplified QASM without headers/measurements.
    """
    # Extract required fields
    probability_distribution = data_item.get(DATASET_MSB_MEASUREMENT_PROBABILITIES, {})
    if type(probability_distribution) is str:
        probability_distribution = json.loads(probability_distribution)

    # sort the probability distribution
    probability_distribution = dict(sorted(probability_distribution.items(), key=lambda x: x[1], reverse=True))

    # round all to 3 decimal places
    probability_distribution = {k: round(v, 3) for k, v in probability_distribution.items()}

    # Get the hash from data_item and find matching entry in python_set    
    python_code_str = data_item.get(DATASET_PYTHON_CODE, "")
    
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
    target_circuit = f"\nQuantum Circuit: \n{python_code_str}"
    formatted_prompt = f"{instruction}{target_circuit}\n\n"

    formatted_completion = data_item.get(DATASET_NL_DESCRIPTION, "")
    
    # Format the completion (the expected output)
    # Ensure we only keep top-15 if needed
    if len(probability_distribution) > 15:
        sorted_probs = dict(list(probability_distribution.items())[:15])
        formatted_completion += f"{json.dumps(sorted_probs)}{eos_token}"
    else:
        formatted_completion += f"{json.dumps(probability_distribution)}{eos_token}"
    
    return {
        "prompt": formatted_prompt,
        "completion": formatted_completion
    }


def process_dataset(data: List[Dict[str, Any]], split: str, data_source: str) -> List[Dict[str, Any]]:
    """
    Process the dataset into the format similar to gsm8k.py
    """
    processed_data = []

    for idx, item in enumerate(data):
        formatted = format_prompt(item)
        
        # Extract fields for extra_info
        num_qubits = item.get(DATASET_NUM_QUBITS, None)
        if num_qubits is None:
            num_qubits = item.get("n_qubits", None)
        circuit_depth = item.get(DATASET_CIRCUIT_DEPTH, None)
        num_gates = item.get(EXTRA_INFO_NUM_GATES, None)
        circuit_hash = item.get(DATASET_CIRCUIT_HASH, "")
        gates_list = item.get(DATASET_GATES_LIST, [])
        num_gates = len(gates_list) if gates_list else num_gates
        python_code_str = item.get(DATASET_PYTHON_CODE, "")
        lsb_measurement_probabilities = item.get(DATASET_LSB_MEASUREMENT_PROBABILITIES, {})
        msb_measurement_probabilities = item.get(DATASET_MSB_MEASUREMENT_PROBABILITIES, {})

        
        processed_item = {
            "data_source": data_source,
            "prompt": formatted["prompt"],
            "completion": formatted["completion"],
            DATASET_EXTRA_INFO: {
                EXTRA_INFO_SPLIT: split,
                EXTRA_INFO_INDEX: idx,
                DATASET_NUM_QUBITS: num_qubits,
                DATASET_CIRCUIT_DEPTH: circuit_depth,
                EXTRA_INFO_NUM_GATES: num_gates,
                DATASET_CIRCUIT_HASH: circuit_hash,
                DATASET_GATES_LIST: json.dumps(gates_list),
                DATASET_PYTHON_CODE: python_code_str,
                DATASET_LSB_MEASUREMENT_PROBABILITIES: json.dumps(lsb_measurement_probabilities),
                DATASET_MSB_MEASUREMENT_PROBABILITIES: json.dumps(msb_measurement_probabilities),
            },
        }
        
        processed_data.append(processed_item)
    
    return processed_data


def main():
    parser = argparse.ArgumentParser(description="Process quantum circuit dataset to parquet format")
    parser.add_argument("--input_file", required=True, help="Path to input JSON lines file")
    parser.add_argument("--local_dir", default="~/data/state_pred", help="Local directory to save processed data")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory to copy data to (optional)")
    parser.add_argument("--train_split", type=float, default=1.0, help="Fraction of data for training (default: 1.0)")
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
        print(f"Using train split: {args.train_split*100:.1f}% for training, {(1-args.train_split)*100:.1f}% for testing")

    
        # Split into train and test
        split_idx = int(len(data) * args.train_split)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        print(f"Split data: {len(train_data)} train, {len(test_data)} test")

        # Process the datasets
        print("Processing training data...")
        processed_train = process_dataset(train_data, "train", args.data_source)


        # Shuffle the training data if specified
        seed = 42
        if args.shuffle:
            random.seed(seed)
            random.shuffle(processed_train)
            print("Shuffled the training data")

        print("Processing test data...")
        processed_test = process_dataset(test_data, "test", args.data_source)



        # Convert to Hugging Face datasets
        train_dataset = datasets.Dataset.from_list(processed_train)
        test_dataset = datasets.Dataset.from_list(processed_test)
        
        # Save to parquet files
        train_path = os.path.join(local_dir, "train.parquet")
        test_path = os.path.join(local_dir, "test.parquet")
        
        print(f"Saving training data to {train_path}...")
        train_dataset.to_parquet(train_path)
        
        print(f"Saving test data to {test_path}...")
        test_dataset.to_parquet(test_path)
    else:
        print("Processing entire dataset as training data...")
        processed_train = process_dataset(data, "train", args.data_source)

        # Shuffle the training data if specified
        seed = 42
        if args.shuffle:
            random.seed(seed)
            random.shuffle(processed_train)
            print("Shuffled the training data")

        train_dataset = datasets.Dataset.from_list(processed_train)
        train_path = os.path.join(local_dir, "train.parquet")
        
        print(f"Saving training data to {train_path}...")
        train_dataset.to_parquet(train_path)
    
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