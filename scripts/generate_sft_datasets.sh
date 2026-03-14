#!/bin/bash
source constants.sh

# Generate jsonl datasets of random circuits
cd "$project_root" || {
    echo "Error: Failed to change directory to $project_root."
    exit 1
}

source $venv || {
    echo "Error: Failed to activate virtual environment at $venv."
    exit 1
}

RANDOM_SET_PATH="$data/random_sets"

# Configuration
TOTAL_SIZE=22000
TRAINING_SET_SIZE=20000
TEST_SET_SIZE=2000
TEMP_OUTPUT_FILE="$RANDOM_SET_PATH/temp_circuits.jsonl"
TRAINING_OUTPUT_FILE="$RANDOM_SET_PATH/train.jsonl"
TEST_OUTPUT_FILE_1="$RANDOM_SET_PATH/test_1.jsonl"
TEST_OUTPUT_FILE_2="$RANDOM_SET_PATH/test_2.jsonl"
TEST_OUTPUT_FILE_3="$RANDOM_SET_PATH/test_3.jsonl"

# make sure output directory exists
mkdir -p "$RANDOM_SET_PATH"

# Function to generate and split circuits
generate_circuits() {
    local num_circuits=$1
    local min_qubits=$2
    local max_qubits=$3
    local min_gates=$4
    local max_gates=$5
    local test_file=$6
    local description=$7
    
    echo "Generating: qubits [$min_qubits-$max_qubits], gates [$min_gates-$max_gates] -> $description"

    
    python -m dataset_generator.src.random_set \
        --num_circuits $num_circuits \
        --min_num_qubits $min_qubits \
        --max_num_qubits $max_qubits \
        --min_num_gates $min_gates \
        --max_num_gates $max_gates \
        --output_file $TEMP_OUTPUT_FILE
    
    head -n $TRAINING_SET_SIZE $TEMP_OUTPUT_FILE >> $TRAINING_OUTPUT_FILE
    tail -n $TEST_SET_SIZE $TEMP_OUTPUT_FILE >> $test_file
    
    rm $TEMP_OUTPUT_FILE
    echo "Done"
}

generate_circuits_test_only() {
    local num_circuits=$1
    local min_qubits=$2
    local max_qubits=$3
    local min_gates=$4
    local max_gates=$5
    local test_file=$6
    local description=$7
    
    echo "Generating: qubits [$min_qubits-$max_qubits], gates [$min_gates-$max_gates] -> $description"

    
    python -m dataset_generator.src.random_set \
        --num_circuits $num_circuits \
        --min_num_qubits $min_qubits \
        --max_num_qubits $max_qubits \
        --min_num_gates $min_gates \
        --max_num_gates $max_gates \
        --output_file $TEMP_OUTPUT_FILE
    
    cat $TEMP_OUTPUT_FILE >> $test_file
    
    rm $TEMP_OUTPUT_FILE
    echo "Done"
}

# # Clear output files
> $TRAINING_OUTPUT_FILE
> $TEST_OUTPUT_FILE_1
> $TEST_OUTPUT_FILE_2
> $TEST_OUTPUT_FILE_3

echo "===== In-Distribution Test Set (test_1) ====="
generate_circuits $TOTAL_SIZE 1 5 1 10 $TEST_OUTPUT_FILE_1 "test_1.jsonl"
generate_circuits $TOTAL_SIZE 1 5 11 20 $TEST_OUTPUT_FILE_1 "test_1.jsonl"
generate_circuits $TOTAL_SIZE 1 5 21 30 $TEST_OUTPUT_FILE_1 "test_1.jsonl"
generate_circuits $TOTAL_SIZE 1 5 31 40 $TEST_OUTPUT_FILE_1 "test_1.jsonl"
generate_circuits $TOTAL_SIZE 1 5 41 50 $TEST_OUTPUT_FILE_1 "test_1.jsonl"

echo ""
echo "===== Extended Gates Test Set (test_2) ====="
generate_circuits_test_only $TEST_SET_SIZE 1 5 51 60 $TEST_OUTPUT_FILE_2 "test_2.jsonl"
generate_circuits_test_only $TEST_SET_SIZE 1 5 61 70 $TEST_OUTPUT_FILE_2 "test_2.jsonl"

echo ""
echo "===== More Qubits Test Set (test_3) ====="
generate_circuits_test_only $TEST_SET_SIZE 6 6 1 20 $TEST_OUTPUT_FILE_3 "test_3.jsonl"
generate_circuits_test_only $TEST_SET_SIZE 7 7 1 20 $TEST_OUTPUT_FILE_3 "test_3.jsonl"

# Final summary
TRAIN_COUNT=$(wc -l < $TRAINING_OUTPUT_FILE)
TEST_1_COUNT=$(wc -l < $TEST_OUTPUT_FILE_1)
TEST_2_COUNT=$(wc -l < $TEST_OUTPUT_FILE_2)
TEST_3_COUNT=$(wc -l < $TEST_OUTPUT_FILE_3)

echo ""
echo "=========================================="
echo "Generation complete!"
echo "Training set: $TRAIN_COUNT circuits"
echo "Test set 1 (in-distribution): $TEST_1_COUNT circuits"
echo "Test set 2 (extended gates): $TEST_2_COUNT circuits"
echo "Test set 3 (more qubits): $TEST_3_COUNT circuits"
echo "=========================================="

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

# Generate reasoning traces for each entry
generate_reasoning_traces $RANDOM_SET_PATH/train.jsonl $RANDOM_SET_PATH/train_updated.jsonl
generate_reasoning_traces $RANDOM_SET_PATH/test_1.jsonl $RANDOM_SET_PATH/test_1_updated.jsonl
generate_reasoning_traces $RANDOM_SET_PATH/test_2.jsonl $RANDOM_SET_PATH/test_2_updated.jsonl
generate_reasoning_traces $RANDOM_SET_PATH/test_3.jsonl $RANDOM_SET_PATH/test_3_updated.jsonl

# Arguments:
#   --input_file       (required) Path to input JSON lines file containing raw training data
#   --local_dir        (optional) Local directory path for saving processed data (default: ~/data/state_pred)
#   --hdfs_dir         (optional) HDFS directory path for remote data backup/distribution (default: None)
#   --train_split      (optional) Fraction of data to allocate for training set, range [0.0-1.0] (default: 1.0)
#   --data_source      (optional) Identifier/tag for data source tracking and organization (default: quantum_circuits/state_pred_reasoning)
#   --shuffle          (optional) Boolean flag to enable random shuffling of training split (default: disabled)
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

local_dir="$RANDOM_SET_PATH/sft_datasets"

generate_sft_parquet "$RANDOM_SET_PATH/train_updated.jsonl" "train_updated.parquet" "$local_dir"
generate_sft_parquet "$RANDOM_SET_PATH/test_1_updated.jsonl" "test_1.parquet" "$local_dir"
generate_sft_parquet "$RANDOM_SET_PATH/test_2_updated.jsonl" "test_2.parquet" "$local_dir"
generate_sft_parquet "$RANDOM_SET_PATH/test_3_updated.jsonl" "test_3.parquet" "$local_dir"

# Rename the training set back to train.parquet at the end
mv "$local_dir/train_updated.parquet" "$local_dir/train.parquet"