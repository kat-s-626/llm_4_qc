from math import pi, sqrt
from qiskit.circuit.library import standard_gates
from config.paths import GROVER_SFT_DIR, ROTATION_SFT_DIR


def _qiskit_gate(name: str):
    if standard_gates is None:
        return None
    return getattr(standard_gates, name, None)

# Gatelist field name constants
GATELIST_TYPE = "type"
GATELIST_PARAMS = "params"
GATELIST_TARGET_QUBITS = "target_qubits"
GATELIST_TARGET_GATE = "target_gate"
GATELIST_NUM_CONTROLS = "num_controls"
GATELIST_NUM_TARGETS = "num_targets"

# Dataset JSONL field name constants
DATASET_NUM_QUBITS = "num_qubits"
DATASET_CIRCUIT_DEPTH = "circuit_depth"
DATASET_GATES_LIST = "gates_list"
DATASET_CIRCUIT_HASH = "circuit_hash"
DATASET_LSB_MEASUREMENT_PROBABILITIES = "lsb_measurement_probabilities"
DATASET_MSB_MEASUREMENT_PROBABILITIES = "msb_measurement_probabilities"
DATASET_PYTHON_CODE = "python_code"
DATASET_NL_DESCRIPTION = "nl_description"
DATASET_EXTRA_INFO = "extra_info"

DATASET_REQUIRED_FIELDS = (
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

# Common extra_info field name constants
EXTRA_INFO_SPLIT = "split"
EXTRA_INFO_INDEX = "index"
EXTRA_INFO_NUM_GATES = "num_gates"
EXTRA_INFO_MARKED_STATES = "marked_states"

SYMBOLIC_PARAMS_TO_STR = {
    0: "0",
    pi / 8: "π/8",
    pi / 6: "π/6",
    pi / 4: "π/4",
    pi / 3: "π/3",
    pi / 2: "π/2",
    2 * pi / 3: "2π/3",
    3 * pi / 4: "3π/4",
    pi: "π",
    -pi / 8: "-π/8",
    -pi / 6: "-π/6",
    -pi / 4: "-π/4",
    -pi / 3: "-π/3",
    -pi / 2: "-π/2",
    -2 * pi / 3: "-2π/3",
    -3 * pi / 4: "-3π/4",
    -pi: "-π",
}

SYMBOLIC_PARAMS_TO_STR_PY = {
    0: "0",
    pi / 8: "pi/8",
    pi / 6: "pi/6",
    pi / 4: "pi/4",
    pi / 3: "pi/3",
    pi / 2: "pi/2",
    2 * pi / 3: "2*pi/3",
    3 * pi / 4: "3*pi/4",
    pi: "pi",
    -pi / 8: "-pi/8",
    -pi / 6: "-pi/6",
    -pi / 4: "-pi/4",
    -pi / 3: "-pi/3",
    -pi / 2: "-pi/2",
    -2 * pi / 3: "-2*pi/3",
    -3 * pi / 4: "-3*pi/4",
    -pi: "-pi",
}

QISKIT_GATES_STR = {
    "h": "HGate",
    "id": "IGate",
    "x": "XGate",
    "y": "YGate",
    "z": "ZGate",
    "s": "SGate",
    "sdg": "SdgGate",
    "t": "TGate",
    "tdg": "TdgGate",
    "p": "PhaseGate",
    "r": "RGate",
    "rx": "RXGate",
    "ry": "RYGate",
    "rz": "RZGate",
    "u": "UGate",
    "u1": "U1Gate",
    "u2": "U2Gate",
    "u3": "U3Gate",
    "cx": "CXGate",
    "ch": "CHGate",
    "cy": "CYGate",
    "cz": "CZGate",
    "cs": "CSGate",
    "csdg": "CSdgGate",
    "cp": "CPhaseGate",
    "csx": "CSXGate",
    "swap": "SwapGate",
    "iswap": "iSwapGate",
    "rxx": "RXXGate",
    "ryy": "RYYGate",
    "rzz": "RZZGate",
    "crx": "CRXGate",
    "cry": "CRYGate",
    "crz": "CRZGate",
    "cu": "CUGate",
    "cu1": "CU1Gate",
    "cu3": "CU3Gate",
    "ccx": "CCXGate",
    "cswap": "CSwapGate",
    "ccz": "CCZGate",
    "xxminusyy": "XXMinusYYGate",
    "xxplusyy": "XXPlusYYGate",
    "ecr": "ECRGate",
    "mcmt": "MCMTGate",
    "mcx": "MCXGate",
    "mcp": "MCPhaseGate",
}

SYMBOLIC_PARAMS = [
    0,
    pi / 8,
    pi / 6,
    pi / 4,
    pi / 3,
    pi / 2,
    2 * pi / 3,
    3 * pi / 4,
    pi,
    -3 * pi / 4,
    -2 * pi / 3,
    -pi / 2,
    -pi / 3,
    -pi / 4,
    -pi / 6,
    -pi / 8,
]

COMMON_SQRT_VALUES = {
    1 / sqrt(2): "1/√2",
    0.5: "1/2",
    1 / (2 * sqrt(2)): "1/(2√2)",
    0.25: "1/4",
    1 / (4 * sqrt(2)): "1/(4√2)",
    sqrt(3) / 2: "√3/2",
    0.0: "0",
    1.0: "1",
}

QISKIT_GATES_1Q = {
    "id": (_qiskit_gate("IGate"), 1, 0),
    "h": (_qiskit_gate("HGate"), 1, 0),
    "x": (_qiskit_gate("XGate"), 1, 0),
    "y": (_qiskit_gate("YGate"), 1, 0),
    "z": (_qiskit_gate("ZGate"), 1, 0),
    "s": (_qiskit_gate("SGate"), 1, 0),
    "sdg": (_qiskit_gate("SdgGate"), 1, 0),
    "t": (_qiskit_gate("TGate"), 1, 0),
    "tdg": (_qiskit_gate("TdgGate"), 1, 0),
    "p": (_qiskit_gate("PhaseGate"), 1, 1),
    "r": (_qiskit_gate("RGate"), 1, 2),
    "rx": (_qiskit_gate("RXGate"), 1, 1),
    "ry": (_qiskit_gate("RYGate"), 1, 1),
    "rz": (_qiskit_gate("RZGate"), 1, 1),
    "u": (_qiskit_gate("UGate"), 1, 3),
    "u1": (_qiskit_gate("U1Gate"), 1, 1),
    "u2": (_qiskit_gate("U2Gate"), 1, 2),
    "u3": (_qiskit_gate("U3Gate"), 1, 3),
}

QISKIT_GATES_2Q = {
    "cx": (_qiskit_gate("CXGate"), 2, 0),
    "ch": (_qiskit_gate("CHGate"), 2, 0),
    "cy": (_qiskit_gate("CYGate"), 2, 0),
    "cz": (_qiskit_gate("CZGate"), 2, 0),
    "cs": (_qiskit_gate("CSGate"), 2, 0),
    "csdg": (_qiskit_gate("CSdgGate"), 2, 0),
    "cp": (_qiskit_gate("CPhaseGate"), 2, 1),
    "csx": (_qiskit_gate("CSXGate"), 2, 0),
    "xxminusyy": (_qiskit_gate("XXMinusYYGate"), 2, 2),
    "xxplusyy": (_qiskit_gate("XXPlusYYGate"), 2, 2),
    "ecr": (_qiskit_gate("ECRGate"), 2, 0),
    "swap": (_qiskit_gate("SwapGate"), 2, 0),
    "iswap": (_qiskit_gate("iSwapGate"), 2, 0),
    "rxx": (_qiskit_gate("RXXGate"), 2, 1),
    "ryy": (_qiskit_gate("RYYGate"), 2, 1),
    "rzz": (_qiskit_gate("RZZGate"), 2, 1),
    "crx": (_qiskit_gate("CRXGate"), 2, 1),
    "cry": (_qiskit_gate("CRYGate"), 2, 1),
    "crz": (_qiskit_gate("CRZGate"), 2, 1),
    "cu": (_qiskit_gate("CUGate"), 2, 4),
    "cu1": (_qiskit_gate("CU1Gate"), 2, 1),
    "cu3": (_qiskit_gate("CU3Gate"), 2, 3),
}

QISKIT_GATES_3Q = {
    "ccx": (_qiskit_gate("CCXGate"), 3, 0),
    "cswap": (_qiskit_gate("CSwapGate"), 3, 0),
    "ccz": (_qiskit_gate("CCZGate"), 3, 0),
}

QISKIT_MULTI_CONTROL_GATES = {
    "mcmt": ("mcmt", None, None),
    "mcx": ("mcx", None, None),
    "mcp": ("mcp", None, None),
}

QISKIT_GATES = {
    **QISKIT_GATES_1Q,
    **QISKIT_GATES_2Q,
    **QISKIT_GATES_3Q,
    **QISKIT_MULTI_CONTROL_GATES,
}
