from config.constants import (
    QISKIT_GATES_STR,
    SYMBOLIC_PARAMS_TO_STR,
    SYMBOLIC_PARAMS_TO_STR_PY)

for gate in ["h", "id", "x", "y", "z", "s", "sdg", "t", "tdg"]:
    print(f"{gate}: {QISKIT_GATES_STR.get(gate, None)}")