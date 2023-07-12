from qiskit import QuantumCircuit


def add_all_hadamards(reg: str, num: [int] = None) -> str:
    """
        Apply an H-gate to all 'qubits' in reg
    """
    if num:
        return "\n".join([f"h {reg}[{i}];\n" for i in num])
    else:
        return f"h {reg};\n"


def superposition_circuit(nqubits: int) -> QuantumCircuit:
    """
        Create a new quantum circuit where all qubits are initialized to |+>
    """
    qc_0 = QuantumCircuit(nqubits)
    return add_all_hadamards(qc_0, range(nqubits))