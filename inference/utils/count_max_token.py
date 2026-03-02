from transformers import AutoTokenizer
import tqdm
import json
import pandas as pd
import argparse

TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

def count_max_tokens(parquet_path):
    
    max_tokens = 0
    avg_tokens = 0
    count = 0

    df = pd.read_parquet(parquet_path)

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        circuit_str = row["extra_info"]["formatted_prompt"] + row["extra_info"]["formatted_completion"]
        inputs = TOKENIZER(circuit_str, return_tensors="pt")
        input_ids = inputs["input_ids"]
        max_tokens = max(max_tokens, len(input_ids[0]))
        avg_tokens += len(input_ids[0])
        count += 1

    avg_tokens = avg_tokens / count if count > 0 else 0
    print(f"Max tokens in {parquet_path}: {max_tokens}")
    print(f"Average tokens in {parquet_path}: {avg_tokens}")
    return max_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count max tokens in a parquet file")
    parser.add_argument("parquet_path", type=str, help="Path to the parquet file")
    args = parser.parse_args()

    count_max_tokens(args.parquet_path)