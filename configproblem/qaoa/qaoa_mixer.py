from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import MCMT, RZGate


def standard_mixer(nqubits: int, beta: Parameter) -> QuantumCircuit:
    """
        Creates a standard mixer circuit for the given number of qubits and beta

        :param nqubits: The number of qubits to create the circuit for
        :param beta: The parameter to use for the circuit
    """
    qc_mix = QuantumCircuit(nqubits)
    for i in range(0, nqubits):
        qc_mix.rx(2 * beta, i)
    return qc_mix


def grover_mixer(nqubits: int, beta: Parameter) -> QuantumCircuit:
    """
        Creates a grover mixer circuit for the given number of qubits and beta

        :param nqubits: The number of qubits to create the circuit for
        :param beta: The parameter to use for the circuit
    """
    qc_mix = QuantumCircuit(nqubits)
    for i in range(0, nqubits):
        qc_mix.h(i)

    for i in range(0, nqubits):
        qc_mix.x(i)

    qc_mix = qc_mix.compose(MCMT(RZGate(2 * beta), nqubits - 1, 1, label=None))

    for i in range(0, nqubits):
        qc_mix.x(i)

    for i in range(0, nqubits):
        qc_mix.h(i)

    return qc_mix.decompose()
