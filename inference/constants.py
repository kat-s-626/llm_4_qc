from math import pi, sqrt
from qiskit.circuit.library import *

SYMBOLIC_PARAMS_TO_STR = {
    # Positive phases
    0: "0",
    pi/8: "π/8",
    pi/6: "π/6",
    pi/4: "π/4",
    pi/3: "π/3",
    pi/2: "π/2",
    2*pi/3: "2π/3",
    3*pi/4: "3π/4",
    pi: "π",
    
    # Negative phases
    -pi/8: "-π/8",
    -pi/6: "-π/6",
    -pi/4: "-π/4",
    -pi/3: "-π/3",
    -pi/2: "-π/2",
    -2*pi/3: "-2π/3",
    -3*pi/4: "-3π/4",
    -pi: "-π",
}

SYMBOLIC_PARAMS_TO_STR_PY = {
    # Positive phases
    0: "0",
    pi/8: "pi/8",
    pi/6: "pi/6",
    pi/4: "pi/4",
    pi/3: "pi/3",
    pi/2: "pi/2",
    2*pi/3: "2*pi/3",
    3*pi/4: "3*pi/4",
    pi: "pi",
    
    # Negative phases
    -pi/8: "-pi/8",
    -pi/6: "-pi/6",
    -pi/4: "-pi/4",
    -pi/3: "-pi/3",
    -pi/2: "-pi/2",
    -2*pi/3: "-2*pi/3",
    -3*pi/4: "-3*pi/4",
    -pi: "-pi",
}


QISKIT_GATES_STR = {
    'h' : "HGate",
    'id' : "IGate",
    'x' : "XGate",
    'y' : "YGate",
    'z' : "ZGate",
    's' : "SGate",
    'sdg' : "SdgGate",
    't' : "TGate",
    'tdg' : "TdgGate",
    'p' : "PhaseGate",
    'r' : "RGate",      # Rotation θ around the cos(φ)x
    'rx' : "RXGate",
    'ry' : "RYGate",
    'rz' : "RZGate",
    'u' : "UGate",
    'u1' : "U1Gate",
    'u2' : "U2Gate",
    'u3' : "U3Gate",
    'cx' : "CXGate",
    'ch' : "CHGate",
    'cy' : "CYGate",
    'cz' : "CZGate",
    'cs' : "CSGate",
    'csdg' : "CSdgGate",
    'cp' : "CPhaseGate",
    'csx' : "CSXGate",
    'swap' : "SwapGate",
    'iswap' : "iSwapGate",
    'rxx' : "RXXGate",
    'ryy' : "RYYGate",
    'rzz' : "RZZGate",
    'crx' : "CRXGate",
    'cry' : "CRYGate",
    'crz' : "CRZGate",
    'cu' : "CUGate",
    'cu1' : "CU1Gate",
    'cu3' : "CU3Gate",
    'ccx' : "CCXGate",  # Toffoli gate
    'cswap' : "CSwapGate",  # Fredkin gate
    'ccz' : "CCZGate",
    'xxminusyy' : "XXMinusYYGate",
    'xxplusyy' : "XXPlusYYGate",
    'ecr' : "ECRGate",
    'mcmt' : "MCMTGate",
    'mcx' : "MCXGate",
    'mcp' : "MCPhaseGate"
}

SYMBOLIC_PARAMS = [
    0,          # 0°
    pi/8,       # 22.5° - T gate (π/4 rotation)
    pi/6,       # 30°
    pi/4,       # 45°
    pi/3,       # 60°
    pi/2,       # 90°
    2*pi/3,     # 120°
    3*pi/4,     # 135°
    pi,         # 180°
    -3*pi/4,    # -135° (equivalent to 225°)
    -2*pi/3,    # -120°
    -pi/2,      # -90°
    -pi/3,      # -60°
    -pi/4,      # -45°
    -pi/6,      # -30°
    -pi/8,      # -22.5°
]

COMMON_SQRT_VALUES = {
    # Hadamard family (uniform superpositions)
    1/sqrt(2): "1/√2",              # 0.707... (1-qubit uniform)
    0.5: "1/2",                      # 0.500... (2-qubit uniform)
    1/(2*sqrt(2)): "1/(2√2)",       # 0.354... (3-qubit uniform)
    0.25: "1/4",                     # 0.250... (4-qubit uniform)
    1/(4*sqrt(2)): "1/(4√2)",       # 0.177... (5-qubit uniform)
    
    # Rotation-induced
    sqrt(3)/2: "√3/2",               # 0.866... (RY(π/3), RY(2π/3))
    
    # Boundary cases
    0.0: "0",                        # Exact zero
    1.0: "1",                        # Basis states
}

# (Gate class, number of qubits, number of parameters)
QISKIT_GATES_1Q = {
    'id': (standard_gates.IGate, 1, 0),
    'h': (standard_gates.HGate, 1, 0),
    'x': (standard_gates.XGate, 1, 0),
    'y': (standard_gates.YGate, 1, 0),
    'z': (standard_gates.ZGate, 1, 0),
    's': (standard_gates.SGate, 1, 0),
    'sdg': (standard_gates.SdgGate, 1, 0),
    't': (standard_gates.TGate, 1, 0),
    'tdg': (standard_gates.TdgGate, 1, 0),
    'p': (standard_gates.PhaseGate, 1, 1),
    'r': (standard_gates.RGate, 1, 2),      # Rotation θ around the cos(φ)x + sin(φ)y axis.
    'rx': (standard_gates.RXGate, 1, 1),
    'ry': (standard_gates.RYGate, 1, 1),
    'rz': (standard_gates.RZGate, 1, 1),
    'u': (standard_gates.UGate, 1, 3),
    'u1': (standard_gates.U1Gate, 1, 1),
    'u2': (standard_gates.U2Gate, 1, 2),
    'u3': (standard_gates.U3Gate, 1, 3),
}

QISKIT_GATES_2Q = {
    'cx': (standard_gates.CXGate, 2, 0),
    'ch': (standard_gates.CHGate, 2, 0),
    'cy': (standard_gates.CYGate, 2, 0),
    'cz': (standard_gates.CZGate, 2, 0),
    'cs': (standard_gates.CSGate, 2, 0),
    'csdg': (standard_gates.CSdgGate, 2, 0),
    'cp': (standard_gates.CPhaseGate, 2, 1),
    'csx': (standard_gates.CSXGate, 2, 0),
    'xxminusyy': (standard_gates.XXMinusYYGate, 2, 2),
    'xxplusyy': (standard_gates.XXPlusYYGate, 2, 2),
    'ecr': (standard_gates.ECRGate, 2, 0),
    'swap': (standard_gates.SwapGate, 2, 0),
    'iswap': (standard_gates.iSwapGate, 2, 0),
    'rxx': (standard_gates.RXXGate, 2, 1),
    'ryy': (standard_gates.RYYGate, 2, 1),
    'rzz': (standard_gates.RZZGate, 2, 1),
    'crx': (standard_gates.CRXGate, 2, 1),
    'cry': (standard_gates.CRYGate, 2, 1),
    'crz': (standard_gates.CRZGate, 2, 1),
    'cu': (standard_gates.CUGate, 2, 4),
    'cu1': (standard_gates.CU1Gate, 2, 1),
    'cu3': (standard_gates.CU3Gate, 2, 3),
}

QISKIT_GATES_3Q = {
    'ccx': (standard_gates.CCXGate, 3, 0),  # Toffoli gate
    'cswap': (standard_gates.CSwapGate, 3, 0),  # Fredkin gate
    'ccz': (standard_gates.CCZGate, 3, 0),  
}

QISKIT_MULTI_CONTROL_GATES = {
    'mcmt': ("mcmt", None, None),
    'mcx': ("mcx", None, None),
    'mcp': ("mcp", None, None),
}

QISKIT_GATES = {**QISKIT_GATES_1Q, **QISKIT_GATES_2Q, **QISKIT_GATES_3Q, **QISKIT_MULTI_CONTROL_GATES}