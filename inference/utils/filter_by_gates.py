import pandas as pd
import os
import argparse

def filter_circuit_by_gates(num_tokens_threshold: int, csv_paths: list[str], jsonl_file: str) -> pd.DataFrame:
    filtered_data_set = pd.DataFrame()
    data_set = pd.read_json(jsonl_file, lines=True)
    for csv in csv_paths:
        df = pd.read_csv(csv)

        means = (
            df[df['num_tokens'] <= num_tokens_threshold]
            .groupby('circuit_hash')['tvd_top15']
            .mean()
        )
        result = means[(means > 0) & (means < 1)]



        # filter the dataset with the circuit hash in result and add to filtered_data_set
        temp_filtered_data_set = data_set[data_set['circuit_hash'].isin(result.index)]
        filtered_data_set = pd.concat([filtered_data_set, temp_filtered_data_set], ignore_index=True)
        
        # count the distribution of the filtered dataset in terms of the number of qubits
        distribution = filtered_data_set['num_qubits'].value_counts().sort_index()

        print("Distribution of number of qubits in the filtered dataset:")
        print(distribution)

        filtered_data_set['num_gates'] = filtered_data_set['gates_list'].apply(lambda x: len(x))

        # count the distribution of the filtered dataset in terms of the number of gates
        distribution_gates = filtered_data_set['num_gates'].value_counts().sort_index()
        print("Distribution of number of gates in the filtered dataset:")
        print(distribution_gates)

        # make this a cross table
        cross_table = pd.crosstab(
            filtered_data_set['num_qubits'],
            filtered_data_set['num_gates']
        )
        print("Cross table of number of qubits and number of gates in the filtered dataset:")
        print(cross_table)

    return filtered_data_set.drop_duplicates(subset='circuit_hash')


def main():
    parser = argparse.ArgumentParser(description='Filter Grover set by token threshold and circuit quality.')
    parser.add_argument(
        '--num-tokens-threshold',
        type=int,
        required=True,
        help='Maximum allowed num_tokens in CSV rows.',
    )
    parser.add_argument(
        '--csv-paths',
        nargs='+',
        required=True,
        help='One or more CSV file paths to aggregate from.',
    )
    parser.add_argument(
        '--jsonl-file',
        type=str,
        required=True,
        help='Path to source dataset JSONL file.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory where filtered JSONL will be saved.',
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Output JSONL filename (e.g., filtered_train_stage_1.jsonl).',
    )
    args = parser.parse_args()

    filtered_data_set = filter_circuit_by_gates(
        num_tokens_threshold=args.num_tokens_threshold,
        csv_paths=args.csv_paths,
        jsonl_file=args.jsonl_file,
    )

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = f'{output_dir}/{args.output_file}'
    print(f"Saving filtered dataset with {len(filtered_data_set)} records to {output_path}")
    filtered_data_set.to_json(output_path, orient='records', lines=True)


if __name__ == '__main__':
    main()
