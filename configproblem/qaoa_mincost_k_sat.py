from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit import Parameter
from scipy.optimize import minimize
import math

from qubovert.utils import DictArithmetic


def k_rz_gate(qc: QuantumCircuit, qubits: list, gate_parameter: float) -> QuantumCircuit:
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
        qc_p.compose(k_rz_gate(qc_p, list(key), 2 * normalization * factor * gamma))

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

    for layer in range(nlayers):
        qc.barrier()
        qc = qc.compose(qg_problem)
        qc.barrier()
        qc = qc.compose(qg_mixer)

    if measure:
        qc.measure_all()
    return qc, gamma, beta


def compute_config_energy(hamiltonian: DictArithmetic, config: list) -> float:
    """
        Computes the energy of a configuration for a given hamiltonian

        :param hamiltonian: The hamiltonian to compute the energy for
        :param config: The configuration to compute the energy for
    """
    energy = 0
    for key, factor in hamiltonian.items():
        term_energy = factor
        for qubit in key:
            term_energy *= config[qubit]
        energy += term_energy
    return energy


def hamiltonian_strategy_top(hamiltonian, counts):
    """
        Computes the energy of the configuration that was measured the most for the given hamiltonian
    """
    # take the config that was measured most often
    config_str = max(counts, key=counts.get)
    # convert the 0/1 feature string to ising integers
    config = [-1 if s == "0" else 1 for s in config_str]
    return compute_config_energy(hamiltonian, config)


def hamiltonian_strategy_average(hamiltonian, counts):
    """
        Computes the average energy across the entire measurement for the given hamiltonian
    """
    average_energy = 0
    for config_str in counts.keys():
        # convert the 0/1 feature string to ising integers
        config = [-1 if s == "0" else 1 for s in config_str]
        energy = compute_config_energy(hamiltonian, config)*counts.get(config_str)
        average_energy += energy
    average_energy /= counts.shots()
    return average_energy


def hamiltonian_strategy_min(hamiltonian, counts):
    """
        Finds the _measured_ configuration with the least energy and returns its value.
    """
    min_energy = float('inf')
    for config_str in counts.keys():
        # convert the 0/1 feature string to ising integers
        config = [-1 if s == "0" else 1 for s in config_str]
        energy = compute_config_energy(hamiltonian, config)
        min_energy = energy if energy < min_energy else min_energy
    return min_energy


def compute_hamiltonian_energy(hamiltonian: DictArithmetic, counts: dict, strategy: str = 'top') -> float:
    """
        Compute the energy state of a hamiltonian from measurements.

        :param hamiltonian: the hamiltonian (QUBO) describing the system
        :param counts: measurement results for a quantum system for the hamiltonian
        :param strategy: method for actually evaluating the hamiltonian. Available: 'avg', 'top', 'min'
    """
    if strategy == 'min':
        return hamiltonian_strategy_min(hamiltonian, counts)
    elif strategy == 'avg':
        return hamiltonian_strategy_average(hamiltonian, counts)
    elif strategy == 'top':
        return hamiltonian_strategy_top(hamiltonian, counts)
    else:
        raise RuntimeError(f"Unsupported strategy: {strategy}")


def quantum(hamiltonian, nqubits, layers, beta_val, gamma_val, shots=512, amplitude_vector=None):
    qc, beta, gamma = qaoa_circuit(hamiltonian, nqubits, layers, amplitude_vector)

    # Set parameters for qc
    qc = qc.bind_parameters({
        beta: beta_val,
        gamma: gamma_val
    })

    # run and measure qc
    qasm_sim = Aer.get_backend('qasm_simulator')
    transpiled_qaoa = transpile(qc, qasm_sim)
    results = qasm_sim.run(transpiled_qaoa, shots=shots).result()
    counts = results.get_counts()

    return counts, qc


def get_expectation(hamiltonian, nqubits, nlayers, shots=128, amplitude_vector=None):
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = shots

    def execute_circ(theta):
        qc, beta, gamma = qaoa_circuit(hamiltonian, nqubits, nlayers, amplitude_vector)

        # Set parameters for qc
        qc = qc.bind_parameters({
            beta: theta[0],
            gamma: theta[1]
        })

        counts = backend.run(qc, nshots=shots).result().get_counts()
        return compute_hamiltonian_energy(hamiltonian, counts, strategy='min')

    return execute_circ


def apply_qaoa(hamiltonian, layers=60, n_features=6, shots=256, theta={"beta": 0.01, "gamma": -0.01},
               warmstart_statevector=None, use_optimizer=True):
    """
        Applies the QAOA Algorithm for the given problem hamiltonian in QUBO form.

        :param int layers: the hyperparameter p of QAOA defining how many cost-mixer-layers will be in the circuit
        :param int n_features: the number of independent variables in the input hamiltonian
        :param int shots: the number of shots used in a simulator run of the QAOA quantum circuit
        :param dict theta: dictionary with keys "beta" and "gamma" that parameterize the QAOA circuit, used as start value when optimizing
        :param list warmstart_statevector: statevector to warmstart to, instead of creating an equal superposition
        :param bool use_optimizer: indicates whether to optimize theta using classical optimizers
    """
    # define expectation function for optimizers
    expectation = get_expectation(hamiltonian, n_features, layers, shots, warmstart_statevector)

    # optimize beta and gamma
    if use_optimizer:
        res = minimize(expectation, [theta["beta"], theta["gamma"]], method='COBYLA', tol=1e-12)
        print(res)
        theta = {"beta": res.x[0], "gamma": res.x[1]}

        # run qaoa circuit with parameters in theta
    counts, qc = quantum(hamiltonian, n_features, layers, theta["beta"], theta["gamma"], shots, warmstart_statevector)
    return counts, qc