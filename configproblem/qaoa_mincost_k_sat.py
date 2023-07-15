from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import math

from qubovert.utils import DictArithmetic


def k_Rz_gate(qc: QuantumCircuit, qubits: list, gate_parameter: float) -> QuantumCircuit:
    """
        Applies a Z gate across the given qubits with the given parameter

        :param qc: The quantum circuit to apply the gate to
        :param qubits: The qubits to apply the gate to
        :param gate_parameter: The parameter to apply the gate with
    """
    if len(qubits) == 0:
        return qc
    for index, qubit in enumerate(qubits):
        if index < len(qubits) - 1:
            qc.cnot(qubit, qubits[index + 1])

    qc.rz(gate_parameter, qubits[-1])

    qubits.reverse()

    for index, qubit in enumerate(qubits):
        if index < len(qubits) - 1:
            qc.cnot(qubits[index + 1], qubit)

    return qc


def problem_circuit(hamiltonian: DictArithmetic, nqubits: int) -> tuple[QuantumCircuit, Parameter]:
    """
        Creates a quantum circuit for the given hamiltonian

        :param hamiltonian: The hamiltonian to create the circuit for
        :param nqubits: The number of qubits to create the circuit for
    """
    gamma = Parameter("$\\gamma$")
    qc_p = QuantumCircuit(nqubits)
    normalization = math.pi/max(hamiltonian.values())

    for key, factor in hamiltonian.items():
        qc_p.compose(k_Rz_gate(qc_p, list(key), 2 * normalization * factor * gamma))

    return qc_p, gamma


def mixer_circuit(nqubits: int) -> tuple[QuantumCircuit, Parameter]:
    """
        Creates a mixer circuit for the given number of qubits

        :param nqubits: The number of qubits to create the circuit for
    """
    beta = Parameter("$\\beta$")
    qc_mix = QuantumCircuit(nqubits)
    for i in range(0, nqubits):
        qc_mix.rx(2 * beta, i)
    return qc_mix, beta


def qaoa_circuit(hamiltonian: DictArithmetic, nqubits: int, nlayers: int, amplitude_vector: list = None,
                 measure: bool = True) -> tuple[QuantumCircuit, Parameter, Parameter]:
    """
        Creates a QAOA circuit for the given hamiltonian

        :param hamiltonian: The hamiltonian to create the circuit for
        :param nqubits: The number of qubits to create the circuit for
        :param nlayers: The number of layers to create the circuit for
        :param amplitude_vector: The amplitude vector to use for the circuit
        :param measure: Whether to measure the circuit
    """
    if amplitude_vector is not None:
        # warm start
        qc = QuantumCircuit(nqubits)
        qc.initialize(amplitude_vector)
    else:
        # equal superposition
        qc = QuantumCircuit(nqubits)

    qg_mixer, beta = mixer_circuit(nqubits)
    qg_problem, gamma = problem_circuit(hamiltonian, nqubits)

    for l in range(nlayers):
        qc.barrier()
        qc = qc.compose(qg_problem)
        qc.barrier()
        qc = qc.compose(qg_mixer)

    if measure:
        qc.measure_all()
    return qc, gamma, beta
