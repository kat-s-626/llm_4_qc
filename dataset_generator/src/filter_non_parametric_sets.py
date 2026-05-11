import argparse
import json
from pathlib import Path


def extract_hash(record: dict) -> str | None:
    if "circuit_hash" in record and record["circuit_hash"] is not None:
        return str(record["circuit_hash"])
    return None


def load_hashes(path: Path) -> tuple[set[str], int, int]:
    hashes: set[str] = set()
    missing_hash = 0
    invalid_json = 0

    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue

            hash_value = extract_hash(record)
            if hash_value is None:
                missing_hash += 1
                continue

            hashes.add(hash_value)

    return hashes, missing_hash, invalid_json


def filter_missing_hashes(train_path: Path, known_hashes: set[str], output_path: Path) -> tuple[int, int, int]:
    total = 0
    kept = 0
    missing_hash = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with train_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue

            total += 1
            record = json.loads(raw)
            hash_value = extract_hash(record)

            if hash_value is None:
                missing_hash += 1
                continue

            if hash_value not in known_hashes:
                outfile.write(raw + "\n")
                kept += 1

    return total, kept, missing_hash


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter train_updated.jsonl rows whose circuit hash does not exist in combined_non_parametric_sets.jsonl"
    )
    parser.add_argument(
        "--train",
        default="non_parametric_sets/train_updated.jsonl",
        help="Input train JSONL file",
    )
    parser.add_argument(
        "--combined",
        default="non_parametric_sets/combined_non_parametric_sets.jsonl",
        help="Reference combined JSONL file",
    )
    parser.add_argument(
        "--output",
        default="filtered_non_parametric_sets.jsonl",
        help="Output filtered JSONL file",
    )

    args = parser.parse_args()

    train_path = Path(args.train)
    combined_path = Path(args.combined)
    output_path = Path(args.output)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined file not found: {combined_path}")

    known_hashes, combined_missing_hash, combined_invalid_json = load_hashes(combined_path)
    total_train, kept, train_missing_hash = filter_missing_hashes(train_path, known_hashes, output_path)

    print(f"Combined hashes loaded: {len(known_hashes)}")
    print(f"Combined rows missing hash: {combined_missing_hash}")
    print(f"Combined invalid JSON rows: {combined_invalid_json}")
    print(f"Train rows processed: {total_train}")
    print(f"Train rows missing hash: {train_missing_hash}")
    print(f"Rows written to {output_path}: {kept}")


if __name__ == "__main__":
    main()