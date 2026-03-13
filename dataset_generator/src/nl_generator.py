import numpy as np
from math import pi
from qiskit.circuit.library import *
from qiskit import qasm3

from config.constants import SYMBOLIC_PARAMS_TO_STR

def natural_language_from_qiskit_circuit(num_qubits, gates_list):
    nl_str = []

    # Find initial state if present
    initial_state = f"|0⟩^⊗{num_qubits}"  # Default initial state
    for gate in gates_list:
        if gate["type"] == "initial_state":
            initial_state = gate["state"]
            break

    nl_str.append(f"Given the initial quantum state {initial_state}, apply the following quantum gates in sequence:")
    nl_str.append("")

    gate_number = 1
    for gate in gates_list:
        if gate["type"] == "initial_state":
            # Skip initial state gates as they're already handled in the introduction
            continue
        else:
            gate_type = gate["type"]
            target_qubits = gate["target_qubits"]
            params = gate.get("params", [])
            target_gate = gate.get("target_gate", None)
            num_controls = gate.get("num_controls", None)
            num_targets = gate.get("num_targets", None)

            if gate_type == "mcmt":
                if not target_gate or not num_controls or not num_targets or not target_qubits:
                    raise ValueError("MCMT gate requires 'target_gate', 'num_controls', 'num_targets', and 'target_qubits'.")
                
                # Partition the qubits based on num_targets
                total_qubits = len(target_qubits)
                # Sorted control qubits to ensure consistent control/target assignment
                control_qubits = sorted(target_qubits[0:total_qubits - num_targets])
                target_gate_qubits = sorted(target_qubits[total_qubits - num_targets:])
                
                # Format base gate name
                base_gate = target_gate.upper()
                
                # Build the natural language description
                if params:
                    params_str = ", ".join(map(_param_to_string, params))
                    if num_targets == 1:
                        nl_desc = f"{gate_number}. Multi-controlled {base_gate}({params_str}) gate controls on qubits {control_qubits}, targets on qubit {target_gate_qubits[0]}."
                    else:
                        nl_desc = f"{gate_number}. Multi-controlled {base_gate}({params_str}) gate controls on qubits {control_qubits}, targets on qubits {target_gate_qubits}."
                else:
                    if num_targets == 1:
                        nl_desc = f"{gate_number}. Multi-controlled {base_gate} gate controls on qubits {control_qubits}, targets on qubit {target_gate_qubits[0]}."
                    else:
                        nl_desc = f"{gate_number}. Multi-controlled {base_gate} gate controls on qubits {control_qubits}, targets on qubits {target_gate_qubits}."
                
                nl_str.append(nl_desc)
            elif gate_type =="mcx":
                if not num_controls or not target_qubits:
                    raise ValueError("MCX gate requires 'num_controls' and 'target_qubits'.")
                
                # Partition qubits: first n-1 are controls, last is target
                control_qubits = sorted(target_qubits[0:-1])
                target_qubit = target_qubits[-1]
                
                # Build the natural language description
                nl_desc = f"{gate_number}. Multi-controlled X gate controls on qubits {control_qubits}, targets on qubit {target_qubit}."
                
                nl_str.append(nl_desc)
            elif gate_type == "mcp":
                if not num_controls or not target_qubits:
                    raise ValueError("MCP gate requires 'num_controls' and 'target_qubits'.")
                
                # Partition qubits: first n-1 are controls, last is target
                control_qubits = sorted(target_qubits[0:-1])
                target_qubit = target_qubits[-1]
                
                # Build the natural language description
                if params:
                    params_str = ", ".join(map(_param_to_string, params))
                    nl_desc = f"{gate_number}. Multi-controlled P({params_str}) gate controls on qubits {control_qubits}, targets on qubit {target_qubit}."
                else:
                    raise ValueError("MCP gate requires 'params'.")
                
                nl_str.append(nl_desc)
            else:
                params_str = ", ".join(map(_param_to_string, params)) if params else ""

                gate_name = gate_type.upper()

                # Only parameterized gates should render as GATE(param)
                if params_str:
                    gate_label = f"{gate_name}({params_str})"
                else:
                    gate_label = gate_name

                if len(target_qubits) == 1:
                    nl_desc = f"{gate_number}. {gate_label} gate on qubit {target_qubits[0]}."
                else:
                    nl_desc = f"{gate_number}. {gate_label} gate on qubits {target_qubits}."

                nl_str.append(nl_desc)
            
            gate_number += 1
        
    return "\n".join(nl_str)


def qasm_code_from_qiskit_circuit(qc):
    """Convert a Qiskit QuantumCircuit to a QASM3.0."""
    return qasm3.dumps(qc)


def _get_import_string():
    return [
        "from qiskit import QuantumCircuit, transpile",
        "from qiskit.circuit.library import *",
        "from qiskit_aer import AerSimulator",
        "from qiskit.quantum_info import Statevector",
        "from math import pi",
        "from itertools import islice",
        "import numpy as np",
    ]

def _get_execution_string(circuit_name, significant_figures=3, top_n=15):
    return [
        "",
        f"{circuit_name}.save_statevector()",
        f"simulator = AerSimulator(method='statevector')",
        f"tqc = transpile({circuit_name}, simulator)",
        f"result = simulator.run(tqc).result()",
        f"statevector = result.get_statevector()",
        f"probabilities = Statevector(statevector).probabilities_dict()",
        f"formatted_probabilities = {{str(k): round(float(v), {significant_figures}) for k, v in probabilities.items() if round(float(v), {significant_figures}) > 0}}",
        f"sorted_probabilities = dict(sorted(formatted_probabilities.items(), key=lambda item: item[1], reverse=True))",
        f"top_probabilities = dict(list(sorted_probabilities.items())[:{top_n}])",
    ]



def _format_qubits(target_qubits):
    """Return a grammatically correct phrase for the target qubits list."""
    if not target_qubits:
        return "no qubits"  # Fallback; ideally shouldn't happen for standard gates
    qubits = [str(q) for q in target_qubits]
    if len(qubits) == 1:
        return f"qubit {qubits[0]}"
    if len(qubits) == 2:
        return f"qubits {qubits[0]} and {qubits[1]}"
    return "qubits " + ", ".join(qubits[:-1]) + f", and {qubits[-1]}"

def _format_parameters(params_str):
    """Return a phrase with correct singular/plural for parameter(s)."""
    if not params_str:
        return "no parameters"
    # Count parameters by splitting on commas that separate arguments
    count = len([p for p in params_str.split(',') if p.strip()])
    if count == 1:
        return f"parameter {params_str.strip()}"
    return f"parameters {params_str}"

def _format_controls_and_target(target_qubits):
    """Return a phrase describing controls and target from a qubit list.

    Assumes the first n-1 qubits are controls and the last is the target.
    Examples:
      [0, 3] -> "control qubit 0 and target qubit 3"
      [0, 1, 3] -> "control qubits 0 and 1, and target qubit 3"
    """
    if not target_qubits:
        return "no qubits"
    if len(target_qubits) == 1:
        return _format_qubits(target_qubits)
    controls = [str(q) for q in target_qubits[:-1]]
    target = str(target_qubits[-1])
    if len(controls) == 1:
        return f"control qubit {controls[0]} and target qubit {target}"
    if len(controls) == 2:
        controls_str = f"{controls[0]} and {controls[1]}"
    else:
        controls_str = ", ".join(controls[:-1]) + f", and {controls[-1]}"
    return f"control qubits {controls_str}, and target qubit {target}"

def _param_to_string(param, tolerance=1e-10):
    """Convert parameter to string with tolerance for floating point comparison."""
    for symbolic_val, str_repr in SYMBOLIC_PARAMS_TO_STR.items():
        if abs(param - symbolic_val) < tolerance:
            return str_repr
    return str(param)

def hash_circuit(qc):
    """Generate a SHA256 hash of the circuit's QASM3 representation."""
    import hashlib
    qasm_str = qasm3.dumps(qc)
    return hashlib.sha256(qasm_str.encode('utf-8')).hexdigest()