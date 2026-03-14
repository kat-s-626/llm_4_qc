import json
import random
import os
from transformers import AutoTokenizer
from datasets import load_dataset
import tqdm
import pandas as pd
from config.paths import DATA_DIR, QWEN_8B_DIR


qwen_model_path = QWEN_8B_DIR
MAX_TOKENS = 17500
CIRCUIT_SETS = "random_sets"

# combine the random gate circuits dataset with the openMath and openCode datasets
# Read from json lines files
SFT_PATH = os.path.join(DATA_DIR, f"{CIRCUIT_SETS}/train_updated.jsonl")

data_files = [
    os.path.join(DATA_DIR, "OpenCodeReasoning/dataset.json"),
    os.path.join(DATA_DIR, "OpenMathReasoning/dataset.json")
]

# And then only random shuffle OpenMath and OpenCode entries
# And random from every other 1 to 5 entries insert random gate circuit entries between them

random.seed(42)  # For reproducibility

def get_tokenizer():
    return AutoTokenizer.from_pretrained(qwen_model_path)

# dict_keys(['id', 'input', 'output', 'source', 'license', 'dataset', 'split', 'difficulty', 'solution', 'text_length'])
def filter_open_code_entries(tokenizer=None, max_tokens=MAX_TOKENS):
    if tokenizer is None:
        tokenizer = get_tokenizer()
    open_code_dir = os.path.join(DATA_DIR, "OpenCodeReasoning")
    open_code = []
    os.makedirs(open_code_dir, exist_ok=True)

    candidate_paths = [
        os.path.join(open_code_dir, "dataset.jsonl")    ]
    input_path = next((path for path in candidate_paths if os.path.exists(path)), None)
    if input_path is None:
        open_code_ds = load_dataset("nvidia/OpenCodeReasoning", "split_0")
        open_code_ds = open_code_ds["split_0"]
        print(f"Loaded OpenCode from Hugging Face dataset with {len(open_code_ds)} entries. Saving to local jsonl...")
        print(f"First entry from Hugging Face OpenCode dataset: {open_code_ds[0]}")
        # read to jsonl
        with open(os.path.join(open_code_dir, "dataset.jsonl"), "w") as f:
            for entry in tqdm.tqdm(open_code_ds, desc="Saving OpenCode to jsonl"):
                json.dump(entry, f)
                f.write("\n")
                open_code.append(entry)
        input_path = os.path.join(open_code_dir, "dataset.jsonl")
        print(f"Loaded OpenCode from Hugging Face and saved to {input_path}")
    else:
        with open(input_path, "r") as f:
            first_char = ""
            while True:
                char = f.read(1)
                if not char:
                    break
                if not char.isspace():
                    first_char = char
                    break
            f.seek(0)

            if first_char == "[":
                loaded = json.load(f)
                if isinstance(loaded, list):
                    open_code = loaded
                else:
                    raise ValueError(f"Expected list in JSON array file: {input_path}")
            else:
                skipped_lines = 0
                for line in tqdm.tqdm(f, desc="Loading OpenCode"):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        entry = json.loads(stripped)
                        open_code.append(entry)
                    except json.JSONDecodeError:
                        skipped_lines += 1
                if skipped_lines:
                    print(f"Skipped {skipped_lines} malformed OpenCode lines from {input_path}")

    open_code = [entry for entry in open_code if isinstance(entry, dict)]
    if not open_code:
        print("No valid OpenCode records found in local file. Reloading from Hugging Face...")
        open_code_ds = load_dataset("nvidia/OpenCodeReasoning", "split_0")
        input_path = os.path.join(open_code_dir, "dataset.jsonl")
        with open(input_path, "w") as f:
            for entry in tqdm.tqdm(open_code_ds, desc="Saving OpenCode to jsonl"):
                json.dump(entry, f)
                f.write("\n")
                open_code.append(entry)

    filtered_open_code = []
    filtered_output_path = os.path.join(open_code_dir, "filtered_open_code.jsonl")
    with open(filtered_output_path, "w"):
        pass

    for entry in tqdm.tqdm(open_code, total=len(open_code), desc="Filtering OpenCode"):
        if len(filtered_open_code) >= 150000:
            break
        problem_text = entry.get("input", "")
        solution_text = entry.get("output", "")
        token_count = len(tokenizer(f"{problem_text}\n{solution_text}")["input_ids"])
        if token_count <= max_tokens:
            filtered_open_code.append(entry)

            with open(filtered_output_path, "a") as f:
                json.dump(entry, f)
                f.write("\n")

    print(f"Filtered OpenCode size: {len(filtered_open_code)} entries")
    return filtered_open_code


# Filter OpenMath entries:
# (1) keep only DeepSeek-R1 generation model samples
# (2) exclude entries above 17,500 tokens for context-window compatibility
def filter_open_math_entries(tokenizer=None, max_tokens=MAX_TOKENS):
    if tokenizer is None:
        tokenizer = get_tokenizer()
    open_math_dir = os.path.join(DATA_DIR, "OpenMathReasoning")

    
    filtered_open_math = []
    open_math = []

    candidate_paths = [
        os.path.join(open_math_dir, "dataset.jsonl"),
        os.path.join(open_math_dir, "dataset.json"),
    ]
    input_path = next((path for path in candidate_paths if os.path.exists(path)), None)
    if input_path is None:
        raise FileNotFoundError(f"No OpenMath dataset file found in {open_math_dir}")

    with open(input_path, "r") as f:
        first_char = ""
        while True:
            char = f.read(1)
            if not char:
                break
            if not char.isspace():
                first_char = char
                break
        f.seek(0)

        if first_char == "[":
            loaded = json.load(f)
            if isinstance(loaded, list):
                open_math = loaded
            else:
                raise ValueError(f"Expected list in JSON array file: {input_path}")
        else:
            skipped_lines = 0
            for line in tqdm.tqdm(f, desc="Loading OpenMath"):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                    open_math.append(entry)
                except json.JSONDecodeError:
                    skipped_lines += 1
            if skipped_lines:
                print(f"Skipped {skipped_lines} malformed OpenMath lines from {input_path}")
    
    for entry in tqdm.tqdm(open_math, total=len(open_math), desc="Filtering OpenMath"):
        if len(filtered_open_math) >= 150000:
            break
        if entry.get("generation_model") != "DeepSeek-R1":
            continue
        if entry.get("inference_mode", "").strip().lower() != "cot":
            continue
        problem_text = entry.get("problem", "")
        solution_text = entry.get("generated_solution", "")
        token_count = len(tokenizer(f"{problem_text}\n{solution_text}")["input_ids"])
        if token_count <= max_tokens:
            filtered_open_math.append(entry)
            with open(os.path.join(open_math_dir, "filtered_open_math.jsonl"), "a") as f:
                json.dump(entry, f)
                f.write("\n")

    print(f"Filtered OpenMath size: {len(filtered_open_math)} entries")
    return filtered_open_math


if __name__ == "__main__":
    filtered_open_math = os.path.join(DATA_DIR, "OpenMathReasoning/filtered_open_math.jsonl")
    if not os.path.exists(filtered_open_math):
        print("Create filtered OpenMath dataset")
        filter_open_math_entries()
    else:
        print(f"Filtered OpenMath dataset already exists at {filtered_open_math}, skipping filtering.")

    filtered_open_code = os.path.join(DATA_DIR, "OpenCodeReasoning/filtered_open_code.jsonl")
    if not os.path.exists(filtered_open_code):
        print("Create filtered OpenCode dataset")
        filter_open_code_entries()
    else:
        print(f"Filtered OpenCode dataset already exists at {filtered_open_code}, skipping filtering.")
    
    print("Loading and filtering circuit dataset...")
    circuit_set = pd.read_json(SFT_PATH, lines=True)   

    print(f"Circuit Sets size: {len(circuit_set)} entries")

    gate_lengths = circuit_set["gates_list"].apply(len)

    sft_1_10  = circuit_set[gate_lengths.between(1,  10)].sample(n=10000, random_state=42)
    sft_11_20 = circuit_set[gate_lengths.between(11, 20)].sample(n=10000, random_state=42)
    sft_21_30 = circuit_set[gate_lengths.between(21, 30)].sample(n=10000, random_state=42)
    sft_31_40 = circuit_set[gate_lengths.between(31, 40)].sample(n=10000, random_state=42)
    sft_41_50 = circuit_set[gate_lengths.between(41, 50)].sample(n=10000, random_state=42)

    print(f"SFT 1-10 size:  {len(sft_1_10)} entries")
    print(f"SFT 11-20 size: {len(sft_11_20)} entries")
    print(f"SFT 21-30 size: {len(sft_21_30)} entries")
    print(f"SFT 31-40 size: {len(sft_31_40)} entries")
    print(f"SFT 41-50 size: {len(sft_41_50)} entries")

    # Combine all datasets
    combined = pd.concat([sft_1_10, sft_11_20, sft_21_30, sft_31_40, sft_41_50], ignore_index=True)
    print(f"Combined dataset size: {len(combined)} entries")

    # Save combined dataset to jsonl
    combined_output_path = os.path.join(DATA_DIR, f"{CIRCUIT_SETS}/combined_{CIRCUIT_SETS}.jsonl")
    combined.to_json(combined_output_path, orient="records", lines=True)
    print(f"Combined dataset saved to {combined_output_path}")

    # Randomly select 50,000 entries from open math and 50,000 entries from open code
    filtered_open_math_entries = pd.read_json(filtered_open_math, lines=True).sample(n=50000, random_state=42)
    filtered_open_code_entries = pd.read_json(filtered_open_code, lines=True).sample(n=50000, random_state=42)

    # Concatenate all datasets together and shuffle
    final_combined = pd.concat([combined, filtered_open_math_entries, filtered_open_code_entries], ignore_index=True).sample(frac=1, random_state=42)
    print(f"Final combined dataset size: {len(final_combined)} entries")

    # Save final combined dataset to jsonl
    final_combined_output_path = os.path.join(DATA_DIR, f"{CIRCUIT_SETS}/final_combined_dataset.jsonl")
    final_combined.to_json(final_combined_output_path, orient="records", lines=True)
    print(f"Final combined dataset saved to {final_combined_output_path}")
