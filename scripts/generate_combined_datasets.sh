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

python -m dataset_generator.src.generate_combined_set 

# Generate the parquet files 
input_file="$random_set_dir/final_combined_dataset.jsonl"
local_dir="$random_set_dir/sft_grpo_datasets"
mkdir -p "$local_dir"


python -m verl.experiments.preprocess_combined_set \
    --input_file "$input_file" \
    --local_dir "$local_dir" \
    --shuffle

cp $random_set_dir/sft_datasets/test_1.parquet $local_dir/test_1.parquet
