from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def inference(tokenizer, model_path, lora_path, data_path, output_path, 
            n_samples=1,
            tensor_parallel_size=1, 
            enable_lora=True, 
            max_lora_rank=16,
            max_num_seqs=256,
            enable_chunked_prefill=True,
            pipeline_parallel_size=1,
            temperature=0.6,
            top_p=0.95,
            max_tokens=10000,
            start_idx=0,
            end_idx=10000,
            enable_circuit_reasoning_format=False):
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Initialize model
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,  # Use 1 GPU
        enable_lora=enable_lora,
        max_lora_rank=max_lora_rank,
            # vLLM automatically batches requests for efficiency
        max_num_seqs=max_num_seqs,  # Max sequences processed concurrently
        enable_chunked_prefill=enable_chunked_prefill,
        pipeline_parallel_size=pipeline_parallel_size
    )

    # Define sampling parameters
    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    ds = pd.read_parquet(data_path)
    ds = ds.iloc[start_idx:end_idx]
    print(f"Total records in the dataset: {len(ds)}")

    filter_ds = []
    for idx in range(len(ds)):
        prompt = ds.iloc[idx]['prompt']

        if type(prompt) is not str:
            if "content" in prompt[0]:
                prompt = prompt[0]["content"]
            else:
                print(f"Prompt at index {idx} is not a string and does not contain 'content' key, skipping.")
                continue
        

        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            prompt_len = len(inputs["input_ids"][0])
        except Exception as e:
            print(f"Error tokenizing prompt at index {idx}: {e}")
            continue
        # Filter prompts based on max_tokens
        if prompt_len > max_tokens:
            print(f"Prompt at index {idx} exceeds max_tokens ({max_tokens}), skipping.")
            continue
        filter_ds.append(ds.iloc[idx])
    
    ds = pd.DataFrame(filter_ds)

    # print columns
    print(f"Columns in dataset: {ds.columns}")

    # ds = ds.reset_index(drop=True)  # Reset index to avoid mismatch in DataFrame creation

    print(f"Processing data from index {start_idx} to {end_idx}, total {len(ds)} records.")

    if 'prompt' not in ds.columns:
        prompts = [p['formatted_prompt'] for p in ds['extra_info']]  # Extract strings from list of dicts
    else:
        try:
            prompts = [p[0]['content'] for p in ds['prompt']]  # Extract strings from list of dicts
        except Exception as e:
            prompts = ds['prompt']
    messages = [{"role": "user", "content": p} for p in prompts]
    prompts = [tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True, reasoning_effort="high") for msg in messages]
    
    completions = ds['completion'] if 'completion' in ds.columns else ["" for _ in range(len(ds))]
    
    # add thinking token <think> after the prompt
    if enable_circuit_reasoning_format:
        prompts = [p + "<circuit_reasoning>" for p in prompts]

    if lora_path and enable_lora:
        lora_adapter = lora_path if enable_lora else None
        lora_path = LoRARequest("adapter", 1, lora_adapter)
        outputs= llm.generate(prompts, sampling_params, lora_request=lora_path)
    else:
        outputs= llm.generate(prompts, sampling_params)

    records = []
    
    # Process outputs (keep all n samples per prompt)
    for prompt_text, ground_truth, extra, output in zip(prompts, completions, ds['extra_info'], outputs):
        for sample_id, candidate in enumerate(output.outputs):
            records.append(
                {
                    'prompt': prompt_text,
                    'completion': ground_truth,
                    'responses': candidate.text,
                    'extra_info': extra,
                    'sample_id': sample_id,
                }
            )

    # save to parquet
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"Length of DataFrame: {len(df)}")
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate per prompt")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to the LoRA adapter")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data in parquet format")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output data in parquet format")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--enable_lora", action='store_true', help="Whether to enable LoRA")
    parser.add_argument("--max_lora_rank", type=int, default=16, help="Max LoRA rank")
    parser.add_argument("--max_num_seqs", type=int, default=256, help="Max number of sequences processed concurrently")
    parser.add_argument("--enable_chunked_prefill", action='store_true', help="Whether to enable chunked prefill")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=10000, help="Maximum tokens to generate")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of the dataset to process")
    parser.add_argument("--end_idx", type=int, default=10000, help="End index of the dataset to process")
    parser.add_argument("--enable_circuit_reasoning_format", action='store_true', help="Whether to append <circuit_reasoning> token to the prompt for circuit reasoning format")

    args = parser.parse_args()

    inference(
        tokenizer=None,
        model_path=args.model_path,
        n_samples=args.n_samples,
        lora_path=args.lora_path,
        data_path=args.data_path,
        output_path=args.output_path,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_lora=args.enable_lora,
        max_lora_rank=args.max_lora_rank,
        max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=args.enable_chunked_prefill,
        pipeline_parallel_size=args.pipeline_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        enable_circuit_reasoning_format=args.enable_circuit_reasoning_format
    )