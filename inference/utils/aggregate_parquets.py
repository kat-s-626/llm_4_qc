# aggregate parquet file with the same prefix
import os
import pandas as pd
import argparse

def aggregate_parquet_files(input_dir, output_file, prefix=''):
    # List all parquet files in the input directory with the specified prefix
    parquet_files = [f for f in os.listdir(input_dir) 
                     if f.endswith('.parquet') and f.startswith(prefix)
    ]
    
    # Initialize an empty list to hold dataframes
    dataframes = []
    
    # Read each parquet file and append to the list
    for file in parquet_files:
        file_path = os.path.join(input_dir, file)
        try:
            if os.path.getsize(file_path) == 0:
                print(f"Skipping zero-byte file {file_path}")
                continue
        except OSError as e:
            print(f"Could not stat {file_path}: {e}; skipping")
            continue

        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}; skipping")
            continue

        dataframes.append(df)
        print(f"Read {file_path} with {len(df)} records.")
    
    if not dataframes:
        print("No valid parquet files found to aggregate.")
        return
    
    # Concatenate all dataframes
    aggregated_df = pd.concat(dataframes, ignore_index=True)
    print(f"Aggregated dataframe has {len(aggregated_df)} total records.")
    
    # Write the aggregated dataframe to a new parquet file
    aggregated_df.to_parquet(output_file)
    print(f"Aggregated parquet file saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate parquet files with the same prefix')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing parquet files')
    parser.add_argument('--output_file', type=str, required=True, 
                        help='Output aggregated parquet file path')
    parser.add_argument('--prefix', type=str, default='', 
                        help='Prefix of parquet files to aggregate (default: all files)')
    
    args = parser.parse_args()
    aggregate_parquet_files(args.input_dir, args.output_file, args.prefix)