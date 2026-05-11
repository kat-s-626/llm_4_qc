#!/bin/bash
source constants.sh
set -euo pipefail

# Generate jsonl datasets of random circuits
cd "$project_root" || {
    echo "Error: Failed to change directory to $project_root."
    exit 1
}

source $venv || {
    echo "Error: Failed to activate virtual environment at $venv."
    exit 1
}

MIN_QUBITS=3
MAX_QUBITS=3
MIN_MARKED_STATES="${MIN_MARKED_STATES:-1}"
MAX_MARKED_STATES="${MAX_MARKED_STATES:-3}"
NUM_SAMPLES_PER_QUBITS="${NUM_SAMPLES_PER_QUBITS:-10}"
OUTPUT_FILE="${OUTPUT_FILE:-$grover_gpt_replication_dir/sampled_marked_states_3q.json}"

mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "Sampling marked states for 3 qubits..."
echo "  marked states per instance: $MIN_MARKED_STATES-$MAX_MARKED_STATES"
echo "  samples per qubit count: $NUM_SAMPLES_PER_QUBITS"
echo "  output: $OUTPUT_FILE"

if [ -s "$OUTPUT_FILE" ]; then
    echo "Marked states file already exists, skipping generation: $OUTPUT_FILE"
else
    python -m dataset_generator.src.marked_state_sampler \
        --min_qubits "$MIN_QUBITS" \
        --max_qubits "$MAX_QUBITS" \
        --min_marked_states "$MIN_MARKED_STATES" \
        --max_marked_states "$MAX_MARKED_STATES" \
        --num_samples_per_qubits "$NUM_SAMPLES_PER_QUBITS" \
        --output_file "$OUTPUT_FILE"
fi

echo "Done: marked states saved to $OUTPUT_FILE"

GROVER_OUTPUT_FILE="${GROVER_OUTPUT_FILE:-$grover_gpt_replication_dir/sampled_grover_circuits_3q.json}"
MULTI_CONTROLLED_Z="${MULTI_CONTROLLED_Z:-true}"
CIRCUITS_JSONL_FILE="${CIRCUITS_JSONL_FILE:-$grover_gpt_replication_dir/sampled_grover_circuits_3q.jsonl}"
REASONING_OUTPUT_FILE="${REASONING_OUTPUT_FILE:-$grover_gpt_replication_dir/sampled_grover_circuits_3q_reasoning.jsonl}"
SFT_LOCAL_DIR="${SFT_LOCAL_DIR:-$grover_gpt_replication_dir/sft_datasets}"
SFT_OUTPUT_FILE="${SFT_OUTPUT_FILE:-grover_circuits_3q.parquet}"
MAX_WORKERS="${MAX_WORKERS:-18}"
data_source="${data_source:-quantum_circuits/grover_3q_reasoning}"

mkdir -p "$(dirname "$GROVER_OUTPUT_FILE")"

echo "Generating Grover circuit dataset from sampled marked states..."
echo "  marked states file: $OUTPUT_FILE"
echo "  output: $GROVER_OUTPUT_FILE"
echo "  multi_controlled_z: $MULTI_CONTROLLED_Z"

if [ -s "$GROVER_OUTPUT_FILE" ]; then
    echo "Grover circuit file already exists, skipping generation: $GROVER_OUTPUT_FILE"
else
    if [ "$MULTI_CONTROLLED_Z" = "true" ]; then
        PYTHONPATH="$project_root:$project_root/dataset_generator/src" python dataset_generator/src/generate_grover_circuit.py \
            --marked_states_file "$OUTPUT_FILE" \
            --output_file "$GROVER_OUTPUT_FILE" \
            --multi_controlled_z
    else
        PYTHONPATH="$project_root:$project_root/dataset_generator/src" python dataset_generator/src/generate_grover_circuit.py \
            --marked_states_file "$OUTPUT_FILE" \
            --output_file "$GROVER_OUTPUT_FILE"
    fi
fi

echo "Done: Grover circuit dataset saved to $GROVER_OUTPUT_FILE"

if [ -s "$CIRCUITS_JSONL_FILE" ]; then
    echo "Circuit JSONL already exists, skipping conversion: $CIRCUITS_JSONL_FILE"
else
    python - "$GROVER_OUTPUT_FILE" "$CIRCUITS_JSONL_FILE" <<'PY'
import json
import os
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
    raise FileNotFoundError(f"Missing or empty Grover circuit JSON file: {input_path}")

with open(input_path, "r") as f:
    data = json.load(f)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    for entry in data:
        f.write(json.dumps(entry) + "\n")

print(f"Converted {len(data)} entries to JSONL: {output_path}")
PY
fi

generate_reasoning_traces() {
    local data_path=$1
    local new_data_path=$2
    local max_workers=${3:-18}

    echo "Generating reasoning traces: $data_path -> $new_data_path"

    python -m dataset_generator.src.simplify_reasoning \
        --data_path $data_path \
        --new_data_path $new_data_path \
        --max_workers $max_workers

    echo "Done"
}

generate_sft_parquet() {
    local input_file=$1
    local output_file=$2
    local local_dir=$3

    python -m verl.experiments.preprocess_sft_set \
        --input_file "$input_file" \
        --local_dir "$local_dir" \
        --data_source "$data_source" \
        --shuffle

    mv "$local_dir"/train.parquet "$local_dir"/$output_file
}

if [ -s "$REASONING_OUTPUT_FILE" ]; then
    echo "Reasoning traces file already exists, skipping generation: $REASONING_OUTPUT_FILE"
else
    generate_reasoning_traces "$CIRCUITS_JSONL_FILE" "$REASONING_OUTPUT_FILE" "$MAX_WORKERS"
fi

mkdir -p "$SFT_LOCAL_DIR"
if [ -s "$SFT_LOCAL_DIR/$SFT_OUTPUT_FILE" ]; then
    echo "Parquet file already exists, skipping generation: $SFT_LOCAL_DIR/$SFT_OUTPUT_FILE"
else
    generate_sft_parquet "$REASONING_OUTPUT_FILE" "$SFT_OUTPUT_FILE" "$SFT_LOCAL_DIR"
fi

echo "Done: SFT parquet saved to $SFT_LOCAL_DIR/$SFT_OUTPUT_FILE"

