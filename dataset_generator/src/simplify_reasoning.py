"""
Generates step-by-step circuit reasoning strings for quantum circuit datasets.

Convention: MSB ordering throughout.
  - Qubit 0 is the LEFTMOST (most significant) bit in all bitstrings.
  - State index i corresponds to bitstring where qubit 0 = bit (n_qubits-1) of i.
  - Example (3 qubits): index 4 = binary 100 = qubit0=1, qubit1=0, qubit2=0 → |100⟩

All state vectors are produced by StateVectorProcessor which natively uses MSB.
"""

import numpy as np
import os
import json
import argparse
import multiprocessing
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
from python_code_generator import (
    python_code_from_qiskit_circuit,
    hash_circuit,
)
from nl_generator import natural_language_from_qiskit_circuit
from config.constants import COMMON_SQRT_VALUES
from state_vector import StateVectorProcessor




def complex_to_symbolic(z, tolerance=1e-10, significant_figure=2):
    """
    Convert a complex number to a human-readable Cartesian string (a + bi).

    MSB convention has no effect here — this is purely a numeric formatter.
    """
    if abs(z) < tolerance:
        return "0"

    real_part = z.real if abs(z.real) > tolerance else 0.0
    imag_part = z.imag if abs(z.imag) > tolerance else 0.0
    z_clean = complex(real_part, imag_part)

    # Pure real
    if abs(z_clean.imag) < tolerance:
        for val, symbolic in COMMON_SQRT_VALUES.items():
            if abs(z_clean.real - val) < tolerance:
                return symbolic
        return f"{z_clean.real:.{significant_figure}f}"

    # Pure imaginary
    if abs(z_clean.real) < tolerance:
        for val, symbolic in COMMON_SQRT_VALUES.items():
            if abs(z_clean.imag - val) < tolerance:
                return f"{symbolic}i"
            if abs(z_clean.imag + val) < tolerance:
                return f"-{symbolic}i"
        if abs(z_clean.imag - 1.0) < tolerance:
            return "i"
        elif abs(z_clean.imag + 1.0) < tolerance:
            return "-i"
        return f"{z_clean.imag:.{significant_figure}f}i"

    # General complex
    real_str = f"{z_clean.real:.{significant_figure}f}"
    imag_str = f"{abs(z_clean.imag):.{significant_figure}f}"
    sign = "+" if z_clean.imag >= 0 else "-"
    return f"{real_str} {sign} {imag_str}i"


def array_to_symbolic(array, significant_figure=2):
    """Convert a state vector to a list of symbolic strings."""
    return [
        str(complex_to_symbolic(amplitude, significant_figure=significant_figure))
        for amplitude in array
    ]


def format_symbolic_array(symbolic_array):
    """Format a symbolic array for inline display (no surrounding quotes)."""
    return "[" + ", ".join(str(e) for e in symbolic_array) + "]"


def _n_qubits_from_state_vector(state_vector):
    """
    Derive the number of qubits from a state vector.

    Asserts that the length is a power of 2, so the bit-width calculation
    is always exact.
    """
    n = len(state_vector)
    assert n > 0 and (n & (n - 1)) == 0, (
        f"State vector length {n} is not a power of 2."
    )
    return n.bit_length() - 1


def get_answer_str(state_vector, significant_figure=2):
    """
    Format the final probability distribution as a human-readable string.

    MSB convention: index i is displayed as an n-bit binary string where
    the leftmost bit corresponds to qubit 0.

    Example (3 qubits):
      index 4 → binary '100' → |100⟩  (qubit0=1, qubit1=0, qubit2=0)
    """
    n_qubits = _n_qubits_from_state_vector(state_vector)

    lines = ["The probability distribution of measurement outcome is:\n"]
    for idx, amplitude in enumerate(state_vector):
        if abs(amplitude) > 1e-6:
            probability = abs(amplitude) ** 2
            amplitude_str = complex_to_symbolic(amplitude, significant_figure=significant_figure)
            # MSB: format idx as n_qubits-wide binary — leftmost bit = qubit 0
            bitstring = format(idx, f"0{n_qubits}b")
            lines.append(
                f"|{bitstring}>: |{amplitude_str}|^2 = {probability:.3f}\n"
            )
    return "".join(lines)



def _make_single_gate_entries(gate_list):
    """
    Yield one-gate sub-lists from a full gate list, skipping 'initial_state'.

    Each yielded item is just the gate dict (not wrapped in an entry dict),
    since n_qubits is now always taken from the parent entry — never inferred
    from target_qubits alone (which can undercount).
    """
    for gate in gate_list:
        if gate["type"] == "initial_state":
            continue
        yield gate



def _build_intermediate_str(step_idx, gate, n_qubits, state_vector, significant_figure=2):
    """
    Build the reasoning string for a single gate step.

    Args:
        step_idx:      1-based step counter
        gate:          gate dict (type, target_qubits, params, …)
        n_qubits:      total qubit count for the circuit
        state_vector:  MSB-ordered state vector AFTER applying this gate
        significant_figure: decimal places for symbolic output
    """
    symbolic = array_to_symbolic(state_vector, significant_figure=significant_figure)

    py_str = python_code_from_qiskit_circuit(
        num_qubits=n_qubits,
        gates_list=[gate],
        circuit_name="circuit",
        significant_figure=3,
        top_probabilities=15,
        include_import=False,
        include_execution=False,
    )
    # Drop the first line (circuit initialisation boilerplate)
    py_str = py_str.split("\n")[1].strip()

    nl_str = natural_language_from_qiskit_circuit(n_qubits, [gate])
    # Drop the first line (initial state sentence) and leading step number
    nl_str = nl_str.split("\n", 2)[-1].strip()
    nl_str = nl_str.split(".", 1)[-1].strip()

    # Ensure exactly one period between segments
    py_str = py_str.rstrip(".")
    nl_str = nl_str.rstrip(".")

    return (
        f"{step_idx}. {py_str}. {nl_str}. "
        f"<quantum_state>{format_symbolic_array(symbolic)}</quantum_state>\n"
    )


def process_single_entry_base_only(entry, lock, output_queue):
    """
    Process one dataset entry using StateVectorProcessor (MSB, no Qiskit).

    State vector evolution:
      - Initialised to |0…0⟩  (index 0 = 1.0)
      - Each gate matrix is applied via left-multiplication: state = M @ state
      - StateVectorProcessor.get_gate_matrix uses MSB convention throughout

    Output bitstrings in the reasoning text and final answer all follow MSB:
      leftmost character = qubit 0.
    """
    SV = StateVectorProcessor()
    try:
        gl = entry.get("gates_list", entry.get("gate_list", []))
        n_qubits = entry.get("num_qubits", entry.get("n_qubits", None))

        if n_qubits is None:
            raise ValueError("Entry is missing 'num_qubits' / 'n_qubits'.")
        if not gl:
            raise ValueError("Entry has an empty gate list.")

        # ── Initialise |0…0⟩ ────────────────────────────────────────────────
        state = np.zeros(2 ** n_qubits, dtype=complex)
        state[0] = 1.0

        initial_state_list = [1 if i == 0 else 0 for i in range(2 ** n_qubits)]

        lines = [
            f"<circuit_reasoning>Initialize a {n_qubits}-qubit quantum circuit "
            f"in the state {initial_state_list}."
        ]

        # ── Step through gates ───────────────────────────────────────────────
        for step_idx, gate in enumerate(_make_single_gate_entries(gl), start=1):
            # MSB matrix multiply — StateVectorProcessor convention
            gate_matrix = SV.get_gate_matrix(gate, n_qubits)
            state = gate_matrix @ state

            lines.append(
                _build_intermediate_str(step_idx, gate, n_qubits, state)
            )

        # ── Final answer ─────────────────────────────────────────────────────
        # get_answer_str also uses MSB, so indices and bitstrings are consistent
        lines.append(get_answer_str(state, significant_figure=2))
        reasoning = "\n".join(lines) + "</circuit_reasoning>"

        new_entry = dict(entry)
        new_entry["nl_description"] = reasoning
        output_queue.put(new_entry)

    except Exception as e:
        print(f"Error processing entry: {e}")
        output_queue.put(None)




def write_results_to_file(output_queue, new_data_path, total_entries, lock):
    """Consume results from the queue and write them to a JSONL file."""
    os.makedirs(os.path.dirname(new_data_path), exist_ok=True)
    written_count = 0

    with open(new_data_path, "w") as f:
        while written_count < total_entries:
            try:
                result = output_queue.get(timeout=30)
                if result is not None:
                    with lock:
                        f.write(json.dumps(result) + "\n")
                        f.flush()
                written_count += 1

                if written_count % 100 == 0:
                    print(f"Written {written_count}/{total_entries} entries")

            except queue.Empty:
                print("Timeout waiting for results, continuing…")


def generate_datasets(data_path, new_data_path=None, max_workers=None):
    """
    Generate reasoning-annotated datasets.

    Args:
        data_path:     Path to input JSONL file.
        new_data_path: Path to output JSONL file.  If None, returns list.
        max_workers:   Worker process count (default: CPU count).
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    with open(data_path, "r") as f:
        data_set = [json.loads(line) for line in f if line.strip()]

    print(f"Processing {len(data_set)} entries with {max_workers} workers")

    manager = Manager()
    output_queue = manager.Queue()
    lock = manager.Lock()

    if new_data_path:
        writer = multiprocessing.Process(
            target=write_results_to_file,
            args=(output_queue, new_data_path, len(data_set), lock),
        )
        writer.start()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_entry_base_only, entry, lock, output_queue)
            for entry in data_set
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Task failed: {e}")

    if new_data_path:
        writer.join()
        print(f"All entries written to {new_data_path}")
        return None

    results = []
    while not output_queue.empty():
        result = output_queue.get()
        if result is not None:
            results.append(result)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate simplified reasoning datasets for quantum circuits."
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to input dataset file (JSONL).",
    )
    parser.add_argument(
        "--new_data_path", type=str, default=None,
        help="Path to save output dataset (JSONL). If omitted, data is returned.",
    )
    parser.add_argument(
        "--max_workers", type=int, default=None,
        help="Number of worker processes (default: CPU count).",
    )
    args = parser.parse_args()

    generate_datasets(args.data_path, args.new_data_path, max_workers=args.max_workers)