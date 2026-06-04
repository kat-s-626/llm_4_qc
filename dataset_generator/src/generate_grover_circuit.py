from qiskit import QuantumCircuit
import math
import os
import json
import argparse
from circuit_runner import run_circuit_statevector
from python_code_generator import python_code_from_qiskit_circuit
from nl_generator import natural_language_from_qiskit_circuit
from dataset_generator.src.parameterized_set import _get_circuit_hash
from qiskit.circuit.library import MCMTGate, ZGate
from config.constants import (
    GATELIST_TYPE,
    GATELIST_PARAMS,
    GATELIST_TARGET_QUBITS,
    GATELIST_TARGET_GATE,
    GATELIST_NUM_CONTROLS,
    GATELIST_NUM_TARGETS,
    DATASET_NUM_QUBITS,
    DATASET_CIRCUIT_DEPTH,
    DATASET_GATES_LIST,
    DATASET_CIRCUIT_HASH,
    DATASET_LSB_MEASUREMENT_PROBABILITIES,
    DATASET_MSB_MEASUREMENT_PROBABILITIES,
    DATASET_PYTHON_CODE,
    DATASET_NL_DESCRIPTION,
    DATASET_EXTRA_INFO,
    EXTRA_INFO_MARKED_STATES,
    EXTRA_INFO_OPTIMAL_ITERATIONS,
    EXTRA_INFO_NUM_ITERATIONS,
    EXTRA_INFO_ITERATIONS_OFFSET,
)

# Grover circuit (high level):
# |0...0> --H^n--> |s> --[ Oracle Uf ]--[ Diffuser D ]-- ... (R rounds) ... --> measure
#                    where D = H^n X^n (CZ / MCX) X^n H^n


def _build_gate_entry(gate_name, params, target_qubits, target_gate=None, num_controls=0, num_targets=1):
    return {
        GATELIST_TYPE: gate_name,
        GATELIST_PARAMS: params,
        GATELIST_TARGET_QUBITS: target_qubits,
        GATELIST_TARGET_GATE: target_gate,
        GATELIST_NUM_CONTROLS: num_controls,
        GATELIST_NUM_TARGETS: num_targets,
    }

def _load_sampled_marked_states(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def _generate_grover_circuit(marked_states, num_qubits, multi_controlled_z=False, iterations_offset=0):
    marked_states = list(marked_states)
    optimal = max(1, math.floor(math.pi / (4 * math.asin(math.sqrt(len(marked_states) / (2 ** num_qubits))))))
    num_iterations = optimal + iterations_offset
    if num_iterations < 1:
        return None, None  # caller should skip
    circuit, gates_list = _circuit_generation_helper(
        num_qubits, marked_states,
        multi_controlled_z=multi_controlled_z,
        num_iterations=num_iterations,
    )
    lsb_measurement_probabilities = run_circuit_statevector(circuit)
    msb_measurement_probabilities = {key[::-1]: value for key, value in lsb_measurement_probabilities.items()}
    python_code = python_code_from_qiskit_circuit(num_qubits, gates_list)
    natural_language = natural_language_from_qiskit_circuit(num_qubits, gates_list)
    circuit_hash = _get_circuit_hash(circuit)
    circuit_data = {
            DATASET_NUM_QUBITS: num_qubits,
            DATASET_CIRCUIT_DEPTH: circuit.depth(),
            DATASET_GATES_LIST: gates_list,
            DATASET_CIRCUIT_HASH: circuit_hash,
            DATASET_LSB_MEASUREMENT_PROBABILITIES: lsb_measurement_probabilities,
            DATASET_MSB_MEASUREMENT_PROBABILITIES: msb_measurement_probabilities,
            DATASET_PYTHON_CODE: python_code,
            DATASET_NL_DESCRIPTION: natural_language,
            DATASET_EXTRA_INFO: {
                EXTRA_INFO_MARKED_STATES: marked_states,
                EXTRA_INFO_OPTIMAL_ITERATIONS: optimal,
                EXTRA_INFO_NUM_ITERATIONS: num_iterations,
                EXTRA_INFO_ITERATIONS_OFFSET: iterations_offset,
            }
        }  
    
    return circuit, circuit_data

def _circuit_generation_helper(num_qubits, marked_states, multi_controlled_z=False, num_iterations=None):
    if num_iterations is None:
        num_iterations = max(1, math.floor(math.pi / (4 * math.asin(math.sqrt(len(marked_states) / (2 ** num_qubits))))))
    
    circuit = QuantumCircuit(num_qubits)
    gates_list = []
    
    # Step 1: Create superposition
    circuit.h(range(num_qubits))
    for qubit in range(num_qubits):
        gates_list.append(_build_gate_entry("h", [], [qubit]))
    
    for _ in range(num_iterations):
        # Step 2: Apply oracle
        oracle_circuit, oracle_gates = _grover_oracle(marked_states, multi_controlled_z=multi_controlled_z)
        circuit.compose(oracle_circuit, inplace=True)
        gates_list.extend(oracle_gates)

        # Step 3: Apply diffuser
        diffuser_circuit, diffuser_gates = _grover_diffuser(num_qubits, multi_controlled_z=multi_controlled_z)
        circuit.compose(diffuser_circuit, inplace=True)
        gates_list.extend(diffuser_gates)
    
    return circuit, gates_list

def _grover_oracle(marked_states, multi_controlled_z=False):

    if not isinstance(marked_states, list):
        marked_states = [marked_states]
    # Compute the number of qubits in circuit
    num_qubits = len(marked_states[0])

    gate_list = []

    circuit = QuantumCircuit(num_qubits)
    # Mark each target state in the input list
    for target in marked_states:
        # Flip target bit-string to match Qiskit bit-ordering
        rev_target = target[::-1]
        
        # Find the indices of all the '0' elements in bit-string
        zero_inds = [
            ind
            for ind in range(num_qubits)
            if rev_target.startswith("0", ind)
        ]
        # Add a multi-controlled Z-gate with pre- and post-applied X-gates (open-controls)
        # where the target bit-string has a '0' entry
        if zero_inds:
            circuit.x(zero_inds)
        for zero_ind in zero_inds:
            gate_list.append(_build_gate_entry("x", [], [zero_ind]))
        
        
        if not multi_controlled_z:
            circuit.h(num_qubits - 1)
            gate_list.append(_build_gate_entry("h", [], [num_qubits - 1]))
            if num_qubits == 2:
                circuit.cx(0, 1)
                gate_list.append(_build_gate_entry("cx", [], [0, 1]))
            elif num_qubits == 3:
                circuit.ccx(0, 1, 2)
                gate_list.append(_build_gate_entry("ccx", [], [0, 1, 2], num_controls=2, num_targets=1))
            else:
                circuit.mcx(list(range(num_qubits - 1)), num_qubits - 1)
                gate_list.append(
                    _build_gate_entry(
                        "mcx",
                        [],
                        [*range(num_qubits)],
                        target_gate="x",
                        num_controls=num_qubits - 1,
                        num_targets=1,
                    )
                )
            
            circuit.h(num_qubits - 1)
            gate_list.append(_build_gate_entry("h", [], [num_qubits - 1]))
        else:
            if num_qubits == 2:
                circuit.cz(0, 1)
                gate_list.append(_build_gate_entry("cz", [], [0, 1]))
            elif num_qubits == 3:
                circuit.ccz(0, 1, 2)
                gate_list.append(_build_gate_entry("ccz", [], [0, 1, 2], num_controls=2, num_targets=1))
            else:
                circuit.append(MCMTGate(ZGate(), num_qubits - 1, 1), list(range(num_qubits)))
                gate_list.append(
                    _build_gate_entry(
                        "mcmt",
                        [],
                        [*range(num_qubits)],
                        target_gate="z",
                        num_controls=num_qubits - 1,
                        num_targets=1,
                    )
                )
        if zero_inds:
            circuit.x(zero_inds)
        for zero_ind in zero_inds:
            gate_list.append(_build_gate_entry("x", [], [zero_ind]))
    return circuit, gate_list

def _grover_diffuser(num_qubits, multi_controlled_z=False):
    circuit = QuantumCircuit(num_qubits)
    gate_list = []
    # Apply Hadamard gates to all qubits
    circuit.h(range(num_qubits))
    for qubit in range(num_qubits):
        gate_list.append(_build_gate_entry("h", [], [qubit]))
    # Apply X gates to all qubits
    circuit.x(range(num_qubits))
    for qubit in range(num_qubits):
        gate_list.append(_build_gate_entry("x", [], [qubit]))
    
        # multi_controlled_z = False: Apply multi-controlled Z gate (with Hadamard gates to convert to multi-controlled X)
    if not multi_controlled_z:
        circuit.h(num_qubits - 1)
        gate_list.append(_build_gate_entry("h", [], [num_qubits - 1]))
        # Apply multi-controlled X gate
        if num_qubits == 2:
            circuit.cx(0, 1)
            gate_list.append(_build_gate_entry("cx", [], [0, 1]))
        elif num_qubits == 3:
                circuit.ccx(0, 1, 2)
                gate_list.append(_build_gate_entry("ccx", [], [0, 1, 2], num_controls=2, num_targets=1))
        else:
            circuit.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            gate_list.append(
                _build_gate_entry(
                    "mcx",
                    [],
                    [*range(num_qubits)],
                    target_gate="x",
                    num_controls=num_qubits - 1,
                    num_targets=1,
                )
            )

        circuit.h(num_qubits - 1)        
        gate_list.append(_build_gate_entry("h", [], [num_qubits - 1]))
        # multi_controlled_z = True: Apply multi-controlled Z gate directly
    else:
        if num_qubits == 2:
            circuit.cz(0, 1)
            gate_list.append(_build_gate_entry("cz", [], [0, 1]))
        elif num_qubits == 3:
            circuit.ccz(0, 1, 2)
            gate_list.append(_build_gate_entry("ccz", [], [0, 1, 2], num_controls=2, num_targets=1))
        else:
            # For more than 3 qubits, we use the MCMT (multi-controlled multi-target) gate to implement the multi-controlled Z gate
            circuit.append(MCMTGate(ZGate(), num_qubits - 1, 1), list(range(num_qubits)))
            gate_list.append(
                _build_gate_entry(
                    "mcmt",
                    [],
                    [*range(num_qubits)],
                    target_gate="z",
                    num_controls=num_qubits - 1,
                    num_targets=1,
                )
            )

    # Apply X gates to all qubits
    circuit.x(range(num_qubits))
    for qubit in range(num_qubits):
        gate_list.append(_build_gate_entry("x", [], [qubit]))
    # Apply Hadamard gates to all qubits
    circuit.h(range(num_qubits))
    for qubit in range(num_qubits):
        gate_list.append(_build_gate_entry("h", [], [qubit]))
    
    return circuit, gate_list    

def main():
    parser = argparse.ArgumentParser(description="Generate and save random Grover's algorithm circuits with specified parameters.")
    parser.add_argument("--min_qubits", type=int, default=2, help="Minimum number of qubits in the generated circuits.")
    parser.add_argument("--max_qubits", type=int, default=5, help="Maximum number of qubits in the generated circuits.")
    parser.add_argument("--min_marked_states", type=int, default=1, help="Minimum number of marked states in the generated circuits.")
    parser.add_argument("--max_marked_states", type=int, default=3, help="Maximum number of marked states in the generated circuits.")
    parser.add_argument("--num_samples_per_qubits", type=int, default=10, help="Number of unique combinations of marked states to sample for each qubit count.")
    parser.add_argument("--multi_controlled_z", action="store_true", help="Whether to use multi-controlled Z gates directly in the oracle and diffuser, instead of converting to multi-controlled X gates with Hadamard transformations.")
    parser.add_argument("--iterations_offset", type=int, default=0, help="Offset added to optimal Grover iterations. 0 = optimal, +1 = over-rotated, -1 = under-rotated. Circuits where optimal+offset < 1 are skipped.")
    parser.add_argument("--marked_states_file", type=str, default="sampled_marked_states.json", help="Path to the file containing sampled marked states.")
    parser.add_argument("--output_file", type=str, default="sampled_grover_circuits.json", help="Path to save the generated circuit data.")
    
    args = parser.parse_args()

    sampled_marked_states = _load_sampled_marked_states(args.marked_states_file)
    if sampled_marked_states is None:
        raise FileNotFoundError(f"Marked states file not found: {args.marked_states_file}")
    
    all_circuit_data = []
    skipped = 0
    for marked_states in sampled_marked_states:
        num_qubits = len(marked_states[0])
        circuit, circuit_data = _generate_grover_circuit(
            marked_states, num_qubits,
            multi_controlled_z=args.multi_controlled_z,
            iterations_offset=args.iterations_offset,
        )
        if circuit_data is None:
            skipped += 1
            continue
        all_circuit_data.append(circuit_data)

    if skipped:
        print(f"Skipped {skipped} circuits (optimal + offset < 1)")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(all_circuit_data, f)

    print(f"Saved {len(all_circuit_data)} circuits to {args.output_file}")

if __name__ == "__main__":
    main()