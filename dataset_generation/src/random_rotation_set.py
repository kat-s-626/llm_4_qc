# Random Parameterized set utilities
#
# Special conditions:
# 1. For multi-controlled gates we limit the variety by enforcing that the controls
#    are the first (n-1) qubits and the target is the last qubit.
# 2. To ensure the dataset includes examples where multi-controlled gates take effect,
#    with 30% probability the generator will add H gates on all qubits to create
#    superposition. This increases the chance that multi-controlled gates act
#    non-trivially and improves training data for models learning their behavior.
#
# Qubit ordering:
# Language models often assume most-significant-qubit ordering, we convert the qiskit LSB to MSB ordering by reversing the qubit indices when adding gates. For example, a gate targeting qubit 0 in qiskit will target qubit (num_qubits - 1) in the generated circuit, and so on.

from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import HGate, RXGate, RYGate, RZGate, CXGate
from .python_code_generator import python_code_from_qiskit_circuit
from .nl_generator import natural_language_from_qiskit_circuit
import pandas as pd
import random
import argparse
import os
import json
import hashlib
from tqdm import tqdm
import math
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
)

# Rotation Set
ROTATION_SET = {
    "rx": RXGate, "ry": RYGate, "rz": RZGate, "h": HGate, "cx": CXGate
}

PARAMS = [
    0,
    math.pi/8,
    math.pi/6,
    math.pi/4,
    math.pi/3,
    math.pi/2,
    2*math.pi/3,
    3*math.pi/4,
    math.pi,
    -math.pi/8,
    -math.pi/6,
    -math.pi/4,
    -math.pi/3,
    -math.pi/2,
    -2*math.pi/3,
    -3*math.pi/4,
    -math.pi,
]

# [single-qubit gates, two-qubit gates]
PROBABILITY_DISTRIBUTIONS = [0.85, 0.15]
AMPLITUDE_THRESHOLD = 1e-10

def _select_gate(num_qubits):
    if num_qubits < 2:
        return 0  # Only single-qubit gates
    else:
        return random.choices(
            population=[0, 1], weights=[0.85, 0.15], k=1
        )[0]  # Use defined probabilities for single and two-qubit gates
    

def _get_measurement_probabilities(circuit):
    state = Statevector.from_instruction(circuit)
    probabilities = state.probabilities_dict()
    cleaned_probabilities = {str(k): float(v) for k, v in probabilities.items() if v > AMPLITUDE_THRESHOLD}
        
    return cleaned_probabilities

def _add_gate(gates_list, gate_name, params, target_qubits, target_gate, num_controls, num_targets):
    gate_list = {GATELIST_TYPE: gate_name,
                 GATELIST_PARAMS: [params] if params is not None else None,
                 GATELIST_TARGET_QUBITS: target_qubits,
                 GATELIST_TARGET_GATE: target_gate,
                 GATELIST_NUM_CONTROLS: num_controls,
                 GATELIST_NUM_TARGETS: num_targets}
    gates_list.append(gate_list)

def _get_circuit_hash(circuit):
    return hashlib.sha256(qasm3.dumps(circuit).encode('utf-8')).hexdigest()

def _reverse_qubit_ordering(measurement_dict):
    reversed_dict = {}
    for key, value in measurement_dict.items():
        reversed_key = key[::-1]  # Reverse the bitstring
        reversed_dict[reversed_key] = value
    return reversed_dict

def _get_python_code(num_qubits, gates_list):
    return python_code_from_qiskit_circuit(num_qubits, gates_list)

def _get_natural_language_description(num_qubits, gates_list):
    return natural_language_from_qiskit_circuit(num_qubits, gates_list)

def generate_random_circuit(num_qubits, max_num_gates):
    gates_list = []
    
    circuit = QuantumCircuit(num_qubits)
    curr_num_gates = 0

    while curr_num_gates < max_num_gates:
        gate_type = _select_gate(num_qubits)
       
        if gate_type == 0:  # Single-qubit gate
            gate = random.choice(list(ROTATION_SET.keys())[:4])  # RX, RY, RZ, H
            if gate in ["rx", "ry", "rz"]:
                params = random.choice(PARAMS)
            else:
                params = None
            qubit = random.randint(0, num_qubits - 1)
            _add_gate(gates_list, gate, params, [qubit], None, 0, 1)
            if params is not None:
                circuit.append(ROTATION_SET[gate](params), [qubit])
            else:
                circuit.append(ROTATION_SET[gate](), [qubit])
            curr_num_gates += 1
        elif gate_type == 1:  # Two-qubit gate
            gate = list(ROTATION_SET.keys())[-1]  # CNOT
            control_qubit = random.randint(0, num_qubits - 1)
            target_qubit = random.choice(
                [q for q in range(num_qubits) if q != control_qubit]
            )
            _add_gate(gates_list, gate, None, [control_qubit, target_qubit], None, 1, 1)
            circuit.append(ROTATION_SET[gate](), [control_qubit, target_qubit])
            curr_num_gates += 1
        
    circuit_hash = _get_circuit_hash(circuit)
    measurement_probabilities = _get_measurement_probabilities(circuit)  # Ensure probabilities are calculated and stored in the circuit object
    return circuit, gates_list, circuit_hash, measurement_probabilities

def generate_random_rotation_set(num_circuits, min_num_qubits, max_num_qubits, min_num_gates, max_num_gates, output_file=None):
    with open(output_file, "w") as f:
        circuit_hashes = set()
        num_generated = 0

        pbar = tqdm(total=num_circuits, desc="Generating random Rotation circuits")
        while num_generated < num_circuits:
            num_qubits = random.randint(min_num_qubits, max_num_qubits)
            num_gates = random.randint(min_num_gates, max_num_gates)
            circuit, gates_list, circuit_hash, measurement_probabilities = generate_random_circuit(num_qubits, num_gates)

            if circuit_hash in circuit_hashes:
                continue  # Skip duplicate circuits
            circuit_hashes.add(circuit_hash)
            msb_measurement_probabilities = _reverse_qubit_ordering(measurement_probabilities)
            python_code = _get_python_code(num_qubits, gates_list)
            nl_description = _get_natural_language_description(num_qubits, gates_list)
            extra_info = {}

            f.write(json.dumps({
                DATASET_NUM_QUBITS: num_qubits,
                DATASET_CIRCUIT_DEPTH: circuit.depth(),
                DATASET_GATES_LIST: gates_list,
                DATASET_CIRCUIT_HASH: circuit_hash,
                DATASET_LSB_MEASUREMENT_PROBABILITIES: measurement_probabilities,
                DATASET_MSB_MEASUREMENT_PROBABILITIES: msb_measurement_probabilities,
                DATASET_PYTHON_CODE: python_code,
                DATASET_NL_DESCRIPTION: nl_description,
                DATASET_EXTRA_INFO: extra_info,
            }) + "\n")

            num_generated += 1
            pbar.update(1)
            
        
    print(f"Generated {num_generated} unique random Grover circuits and saved to {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random Grover circuits and save to JSON.")
    parser.add_argument("--num_circuits", type=int, default=10, help="Number of random circuits to generate.")
    parser.add_argument("--min_num_qubits", type=int, default=2, help="Minimum number of qubits in the circuits.")
    parser.add_argument("--max_num_qubits", type=int, default=5, help="Maximum number of qubits in the circuits.")
    parser.add_argument("--min_num_gates", type=int, default=5, help="Minimum number of gates in the circuits.")
    parser.add_argument("--max_num_gates", type=int, default=20, help="Maximum number of gates in the circuits.")
    parser.add_argument("--output_file", type=str, default="random_grover_set.json", help="Output JSON file to save the generated circuits.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()  

    random.seed(args.random_seed)

    data = []
    generate_random_rotation_set(num_circuits=args.num_circuits,
                              min_num_qubits=args.min_num_qubits,
                              max_num_qubits=args.max_num_qubits,
                              min_num_gates=args.min_num_gates,
                              max_num_gates=args.max_num_gates,
                              output_file=args.output_file) 
