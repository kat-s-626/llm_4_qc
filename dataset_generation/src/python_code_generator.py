"""Convert quantum circuits to python code string and qasm code string from list of gates."""
import numpy as np
from math import pi
from qiskit.circuit.library import *
from qiskit import QuantumCircuit, qasm3

from config.constants import QISKIT_GATES_STR, SYMBOLIC_PARAMS_TO_STR_PY

def python_code_from_qiskit_circuit(num_qubits, gates_list, circuit_name="circuit", significant_figure=3, top_probabilities=15, include_import=False, include_execution=False):
    python_str = []
    if include_import:
        python_str.extend(_get_import_string())
        python_str.append("")

    python_str.append(f"{circuit_name} = QuantumCircuit({num_qubits})")

    for gate in gates_list:
        if gate["type"] == "initial_state":
            state = gate["state"]
            python_str.append(f"{circuit_name}.initialize({state})")
        else:
            gate_type = gate["type"]
            target_qubits = gate["target_qubits"]
            params = gate.get("params", [])
            target_gate = gate.get("target_gate", None)
            num_controls = gate.get("num_controls", None)
            num_targets = gate.get("num_targets", None)

            if gate_type == "mcmt":
                # qc.append(MCMTGate(ZGate(), 2, 1), target_qubits)
                # partition the qubits based on num_controls and num_targets and then sort them separately and merge back
                total_qubits = len(target_qubits)
                control_qubits = sorted(target_qubits[0:total_qubits - num_targets])
                target_gate_qubits = sorted(target_qubits[total_qubits - num_targets:])
                target_qubits = control_qubits + target_gate_qubits

                mcmt_gate_str = ""
                if not target_gate:
                    raise ValueError("MCMT gate requires 'target_gate'.")
                if not num_controls:
                    raise ValueError("MCMT gate requires 'num_controls'.")
                if not num_targets:
                    raise ValueError("MCMT gate requires 'num_targets'.")
                if not target_qubits:
                    raise ValueError("MCMT gate requires 'target_qubits'.")
                params_str = ", ".join(map(_param_to_string, params)) if params else ""
                if params_str:
                    target_gate_str = f"{QISKIT_GATES_STR.get(target_gate, None)}({params_str})"
                else:
                    target_gate_str = f"{QISKIT_GATES_STR.get(target_gate, None)}()"
                
                mcmt_gate_str = f"MCMTGate({target_gate_str}, {num_controls}, {num_targets})"
                
                if target_qubits: 
                    python_str.append(f"{circuit_name}.append({mcmt_gate_str}, {target_qubits})")
                else:
                    python_str.append(f"{circuit_name}.append({mcmt_gate_str})")
            elif gate_type =="mcx":
                mcx_gate_str = ""
                if not num_controls:
                    raise ValueError("MCX gate requires 'num_controls'.")
                
                mcx_gate_str = f"MCXGate({num_controls})"
                
                if target_qubits:
                    python_str.append(f"{circuit_name}.append({mcx_gate_str}, {target_qubits})")
                else:
                    python_str.append(f"{circuit_name}.append({mcx_gate_str})")
            elif gate_type == "mcp":
                mcp_gate_str = ""
                if not num_controls:
                    raise ValueError("MCP gate requires 'num_controls'.")
                params_str = ", ".join(map(_param_to_string, params)) if params else ""
                mcp_gate_str = f"MCPhaseGate({params_str}, {num_controls})"
                
                if target_qubits:
                    python_str.append(f"{circuit_name}.append({mcp_gate_str}, {target_qubits})")
                else:
                    python_str.append(f"{circuit_name}.append({mcp_gate_str})")
            else:
                params_str = ", ".join(map(_param_to_string, params)) if params else ""
                
                # Format target qubits properly - expand list if more than one qubit
                if len(target_qubits) == 1:
                    target_qubits_str = str(target_qubits[0])
                else:
                    target_qubits_str = ", ".join(map(str, target_qubits))

                if params_str:
                    python_str.append(f"{circuit_name}.{gate_type}({params_str}, {target_qubits_str})")
                else:
                    python_str.append(f"{circuit_name}.{gate_type}({target_qubits_str})")
    
    if include_execution:
        python_str.extend(_get_execution_string(circuit_name, significant_figure, top_probabilities))
        
    return "\n".join(python_str)
    

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

def _param_to_string(param, tolerance=1e-10):
    """Convert parameter to string with tolerance for floating point comparison."""
    for symbolic_val, str_repr in SYMBOLIC_PARAMS_TO_STR_PY.items():
        if abs(param - symbolic_val) < tolerance:
            return str_repr
    return str(param)

def hash_circuit(qc):
    """Generate a SHA256 hash of the circuit's QASM3 representation."""
    import hashlib
    qasm_str = qasm3.dumps(qc)
    return hashlib.sha256(qasm_str.encode('utf-8')).hexdigest()