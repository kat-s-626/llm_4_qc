#!/bin/bash

# Script for generating set of marked states using marked_state_sampler.py

# Configuration
source "constants.sh"

cd "$project_root" || {
    echo "Error: Failed to change directory to $project_root."
    exit 1
}

source "$venv" || {
	echo "Error: Failed to activate virtual environment at $venv."
	exit 1
}

# Generate marked states lists
MIN_QUBITS="${MIN_QUBITS:-2}"
MAX_QUBITS="${MAX_QUBITS:-5}"
MIN_MARKED_STATES="${MIN_MARKED_STATES:-1}"
MAX_MARKED_STATES="${MAX_MARKED_STATES:-3}"
NUM_SAMPLES_PER_QUBITS="${NUM_SAMPLES_PER_QUBITS:-1}"
OUTPUT_FILE="${OUTPUT_FILE:-$grover_gpt_replication_dir/sampled_marked_states.json}"

mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "Generating marked states list..."
echo "  qubits: $MIN_QUBITS-$MAX_QUBITS"
echo "  marked states per instance: $MIN_MARKED_STATES-$MAX_MARKED_STATES"
echo "  samples per qubit count: $NUM_SAMPLES_PER_QUBITS"
echo "  output: $OUTPUT_FILE"

python -m dataset_generator.src.marked_state_sampler \
	--min_qubits "$MIN_QUBITS" \
	--max_qubits "$MAX_QUBITS" \
	--min_marked_states "$MIN_MARKED_STATES" \
	--max_marked_states "$MAX_MARKED_STATES" \
	--num_samples_per_qubits "$NUM_SAMPLES_PER_QUBITS" \
	--output_file "$OUTPUT_FILE"

echo "Done: marked states saved to $OUTPUT_FILE"

