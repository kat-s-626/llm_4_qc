import random
from itertools import combinations
import argparse
import os
import json

def _get_all_bitstrings(num_qubits):
    return [bin(i)[2:].zfill(num_qubits) for i in range(2 ** num_qubits)]

def _random_bitstring(n):
    return ''.join(random.choice('01') for _ in range(n))

def _save_sampled_marked_states(sampled_marked_states, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'w') as f:
        json.dump(sampled_marked_states, f)

def sample_marked_states(
    min_qubits,
    max_qubits,
    min_marked_states,
    max_marked_states,
    num_samples_per_qubits,
    max_attempts=1000,
):
    if min_qubits > max_qubits:
        raise ValueError(f"min_qubits ({min_qubits}) cannot be greater than max_qubits ({max_qubits})")
    if min_marked_states > max_marked_states:
        raise ValueError(f"min_marked_states ({min_marked_states}) cannot be greater than max_marked_states ({max_marked_states})")
    
    if min_marked_states < 1:
        raise ValueError("min_marked_states must be at least 1 to ensure valid marked states.")
    if max_marked_states < 1:
        raise ValueError("max_marked_states must be at least 1 to ensure valid marked states.")
    if min_qubits < 1:
        raise ValueError("min_qubits must be at least 1 to have valid qubits.")
    if max_qubits < 1:
        raise ValueError("max_qubits must be at least 1 to have valid qubits.")

    sampled_marked_states = set()

    for num_qubits in range(min_qubits, max_qubits + 1):
        # Only consider valid marked states that are less than half of the total states - 1
        max_possible_marked_states = 2 ** (num_qubits - 1) - 1

        curr_min_marked_states = min_marked_states
        curr_max_marked_states = min(max_marked_states, max_possible_marked_states)

        if curr_min_marked_states > curr_max_marked_states:
            Warning(
                f"Skipping {num_qubits} qubits: min_marked_states ({curr_min_marked_states}) is greater than max_marked_states ({curr_max_marked_states})"
            )
            continue

        for num_marked_states in range(curr_min_marked_states, curr_max_marked_states + 1):
            if num_marked_states > max_possible_marked_states:
                break

            max_sampled_per_qubits = min(num_samples_per_qubits, max_possible_marked_states)
            if num_qubits < 7 and num_marked_states < 5:
                # For instances in smaller scale, include all possible combinations of marked states to ensure diversity
                all_bitstrings = _get_all_bitstrings(num_qubits)
                all_combinations = list(combinations(all_bitstrings, num_marked_states))

                sampled_combinations = random.sample(all_combinations, max_sampled_per_qubits)

                for combination in sampled_combinations:
                    sampled_marked_states.add(tuple(sorted(combination)))
            else:
                sampled_marked_states = set()
                attempts = 0
                while attempts < max_attempts and len(sampled_marked_states) < max_sampled_per_qubits:
                    marked_states = tuple(sorted(_random_bitstring(num_qubits) for _ in range(num_marked_states)))

                    prev_sampled_count = len(sampled_marked_states)
                    sampled_marked_states.add(marked_states)

                    if len(sampled_marked_states) > prev_sampled_count:
                        attempts = 0
                    else:
                        attempts += 1

    # Transform to list of lists for JSON serialization
    sampled_marked_states = [list(combination) for combination in sampled_marked_states]

    return sampled_marked_states

def main():
    parser = argparse.ArgumentParser(description="Sample marked states for Grover's algorithm circuits.")
    parser.add_argument("--min_qubits", type=int, default=2, help="Minimum number of qubits.")
    parser.add_argument("--max_qubits", type=int, default=5, help="Maximum number of qubits.")
    parser.add_argument("--min_marked_states", type=int, default=1, help="Minimum number of marked states.")
    parser.add_argument("--max_marked_states", type=int, default=3, help="Maximum number of marked states.")
    parser.add_argument("--num_samples_per_qubits", type=int, default=10, help="Number of samples per qubit count.")
    parser.add_argument("--output_file", type=str, default="sampled_marked_states.json", help="Output file for sampled marked states.")

    args = parser.parse_args()

    sampled_marked_states = sample_marked_states(
        args.min_qubits,
        args.max_qubits,
        args.min_marked_states,
        args.max_marked_states,
        args.num_samples_per_qubits
    )

    _save_sampled_marked_states(sampled_marked_states, args.output_file)

if __name__ == "__main__":
    main()