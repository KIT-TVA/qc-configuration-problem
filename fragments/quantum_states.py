from qiskit import QuantumCircuit


def add_all_hadamards(qc, nqubits):
    """
        Apply an H-gate to all 'qubits' in qc
    """
    for q in nqubits:
        qc.h(q)
    return qc


def superposition_circuit(nqubits: int) -> QuantumCircuit:
    """
        Create a new quantum circuit where all qubits are initialized to |+>
    """
    qc_0 = QuantumCircuit(nqubits)
    return add_all_hadamards(qc_0, range(nqubits))