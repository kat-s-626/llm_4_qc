#!/usr/bin/env python3
"""
Extract marked states from GroverGPT-plus data_MMS filenames and write
sampled_marked_states_{N}q.json files in the same format used by the pipeline.

File naming convention:
  grover_n{N}_k{K}_m{s1}_{s2}_{...}.qasm
  e.g.  grover_n3_k2_m000_010.qasm  →  marked_states = ['000', '010']

Usage:
  python dataset_generator/src/extract_marked_states_mms.py \
      --data_mms_dir /scratch3/ip004/GroverGPT-plus/data_MMS \
      --output_dir   data/grover_gpt_replication \
      --nq 2 3 4 5 \
      --max_k 3
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_marked_states(filename: str) -> list[str] | None:
    """
    Parse the marked states from a filename like:
      grover_n4_k2_m0001_0110.qasm  →  ['0001', '0110']
      grover_n3_k1_m010.qasm        →  ['010']
    Returns None if the filename does not match the expected pattern.
    """
    stem = Path(filename).stem  # strip .qasm
    # Pattern: grover_n<N>_k<K>_m<states_separated_by_underscores>
    m = re.match(r"grover_n(\d+)_k(\d+)_m(.+)$", stem)
    if not m:
        return None
    k = int(m.group(2))
    rest = m.group(3)  # e.g. "0001_0110" or "000"
    states = rest.split("_")
    if len(states) != k:
        return None
    return states


def extract_for_nq(data_mms_dir: Path, nq: int, max_k: int) -> list[list[str]]:
    subdir = data_mms_dir / f"grover_n{nq}"
    if not subdir.exists():
        print(f"  WARNING: directory not found: {subdir}")
        return []

    all_entries: list[list[str]] = []
    skipped = 0
    for f in sorted(subdir.iterdir()):
        if not f.name.endswith(".qasm"):
            continue
        # Filter by k
        m = re.match(r"grover_n\d+_k(\d+)_", f.name)
        if not m:
            continue
        k = int(m.group(1))
        if k > max_k:
            skipped += 1
            continue
        states = parse_marked_states(f.name)
        if states is None:
            print(f"  WARNING: could not parse: {f.name}")
            continue
        all_entries.append(states)

    print(f"  n{nq}q: {len(all_entries)} entries  (skipped k>{max_k}: {skipped})")
    return all_entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract marked states from data_MMS filenames."
    )
    parser.add_argument(
        "--data_mms_dir",
        type=Path,
        default=Path("/scratch3/ip004/GroverGPT-plus/data_MMS"),
        help="Root directory of data_MMS",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write sampled_marked_states_{N}q.json files",
    )
    parser.add_argument(
        "--nq",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5],
        help="Qubit counts to process (default: 2 3 4 5)",
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=3,
        help="Maximum number of marked states to include (default: 3)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for nq in args.nq:
        print(f"Processing n{nq}q ...")
        entries = extract_for_nq(args.data_mms_dir, nq, args.max_k)
        out_file = args.output_dir / f"sampled_marked_states_{nq}q.json"
        with out_file.open("w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2)
        print(f"  Saved {len(entries)} entries → {out_file}")


if __name__ == "__main__":
    main()
