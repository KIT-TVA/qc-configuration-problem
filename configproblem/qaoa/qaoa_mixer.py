from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def standard_mixer(nqubits: int, beta: Parameter) -> QuantumCircuit:
    """
        Creates a mixer circuit for the given number of qubits

        :param nqubits: The number of qubits to create the circuit for
        :param beta: The parameter to use for the circuit
    """
    qc_mix = QuantumCircuit(nqubits)
    for i in range(0, nqubits):
        qc_mix.rx(2 * beta, i)
    return qc_mix
