from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import MCMT, RZGate


def get_state_preparation_circuit(nqubits: int, statevector: list[float]) -> QuantumCircuit:
    """
        Creates a state preparation circuit for the given number of qubits and statevector

        :param nqubits: The number of qubits to create the circuit for
        :param statevector: The statevector to use for the circuit
    """
    qc_state_preperation = QuantumCircuit(nqubits)
    qc_state_preperation.initialize(statevector, [i for i in range(nqubits)])

    # Remove reset gates at the beginning of the circuit, so that it only contains the state preparation unitary
    qc_state_preperation = qc_state_preperation.decompose()
    for i in range(nqubits):
        qc_state_preperation.data.pop(0)
    # Decompose the state preparation unitary into standard gates
    return qc_state_preperation.decompose(reps=nqubits + 2)


def standard_mixer(nqubits: int, beta: Parameter, statevector: list[float] = None) -> QuantumCircuit:
    """
        Creates a standard mixer circuit for the given number of qubits and beta

        :param nqubits: The number of qubits to create the circuit for
        :param beta: The parameter to use for the circuit
        :param statevector: The statevector to use for the circuit (not used for this mixer)
    """
    qc_mix = QuantumCircuit(nqubits)
    for i in range(0, nqubits):
        qc_mix.rx(2 * beta, i)
    return qc_mix


def grover_mixer(nqubits: int, beta: Parameter, statevector: list[float] = None) -> QuantumCircuit:
    """
        Creates a grover mixer circuit for the given number of qubits and beta

        :param nqubits: The number of qubits to create the circuit for
        :param beta: The parameter to use for the circuit
        :param statevector: The statevector to use for the circuit
    """
    qc_mix = QuantumCircuit(nqubits)

    if statevector is None:
        qc_state_preparation = QuantumCircuit(nqubits)
        for i in range(nqubits):
            qc_state_preparation.h(i)
    else:
        qc_state_preparation = get_state_preparation_circuit(nqubits, statevector)

    qc_mix = qc_mix.compose(qc_state_preparation.decompose(reps=2).inverse())

    for i in range(0, nqubits):
        qc_mix.x(i)

    qc_mix = qc_mix.compose(MCMT(RZGate(2 * beta), nqubits - 1, 1, label=None))

    for i in range(0, nqubits):
        qc_mix.x(i)

    qc_mix = qc_mix.compose(qc_state_preparation)

    return qc_mix.decompose()


def warmstart_mixer(nqubits: int, beta: Parameter, statevector: list[float] = None) -> QuantumCircuit:
    """
        Creates a warmstart mixer circuit for the given number of qubits, beta and statevector according to the paper by
        Egger et al. (https://quantum-journal.org/papers/q-2021-06-17-479/)

        :param nqubits: The number of qubits to create the circuit for
        :param beta: The parameter to use for the circuit
        :param statevector: The statevector to use for the circuit
    """
    if statevector is None:
        raise ValueError("Statevector cannot be None for warmstart mixer")
    qc_mix = QuantumCircuit(nqubits)

    qc_state_preparation = get_state_preparation_circuit(nqubits, statevector)

    qc_mix = qc_mix.compose(qc_state_preparation.inverse())

    for i in range(0, nqubits):
        qc_mix.rz(-2 * beta, i)

    qc_mix = qc_mix.compose(qc_state_preparation)

    return qc_mix
