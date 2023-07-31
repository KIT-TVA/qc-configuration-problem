from typing import Callable

from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit import Parameter
from qiskit.result import Counts
from qiskit_aer import StatevectorSimulator
from qubovert.utils import DictArithmetic
from scipy.optimize import minimize

from configproblem.fragments.quantum_states import superposition_circuit
from configproblem.util.hamiltonian_math import compute_hamiltonian_energy, compute_hamiltonian_energy_from_statevector


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


def qaoa_circuit(problem_circuit: Callable, hamiltonian: DictArithmetic, nqubits: int, nlayers: int,
                 amplitude_vector: list[float] = None, measure: bool = True) -> tuple[QuantumCircuit, Parameter, Parameter]:
    """
        Creates a QAOA circuit for the given hamiltonian

        :param problem_circuit: The method for creating the corresponding problem quantum circuit
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
        qc = superposition_circuit(nqubits)

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


def quantum(problem_circuit: Callable, hamiltonian: DictArithmetic, nqubits: int, layers: int, beta_val: float,
            gamma_val: float, shots: int = 512, amplitude_vector: list[float] = None) -> tuple[Counts, QuantumCircuit]:
    qc, beta, gamma = qaoa_circuit(problem_circuit, hamiltonian, nqubits, layers, amplitude_vector)

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


def get_expectation(problem_circuit: Callable, hamiltonian: DictArithmetic, nqubits: int, nlayers: int,
                    shots: int = 128, amplitude_vector: list[float] = None) -> Callable:
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = shots

    def execute_circ(theta):
        qc, beta, gamma = qaoa_circuit(problem_circuit, hamiltonian, nqubits, nlayers, amplitude_vector)

        # Set parameters for qc
        qc = qc.bind_parameters({
            beta: theta[0],
            gamma: theta[1]
        })

        counts = backend.run(qc, nshots=shots).result().get_counts()
        return compute_hamiltonian_energy(hamiltonian, counts, strategy='min')

    return execute_circ


def apply_qaoa(problem_circuit: Callable, hamiltonian: DictArithmetic, layers: int = 60, n_features: int = 6,
               shots: int = 256, theta={"beta": 0.01, "gamma": -0.01}, warmstart_statevector: list[float] = None,
               use_optimizer: bool = True, print_res: bool = True) -> tuple[Counts, QuantumCircuit]:
    """
        Applies the QAOA Algorithm for the given problem hamiltonian in QUSO form.

        :param problem_circuit: The function for creating the corresponding problem quantum circuit
        :param hamiltonian: the hamiltonian used for creating the quantum circuit
                            and determining the expected config cost
        :param int layers: the hyperparameter p of QAOA defining how many cost-mixer-layers will be in the circuit
        :param int n_features: the number of independent variables in the input hamiltonian
        :param int shots: the number of shots used in a simulator run of the QAOA quantum circuit
        :param dict theta: dictionary with keys "beta" and "gamma" that parameterize the QAOA circuit,
                           used as start value when optimizing
        :param list warmstart_statevector: statevector to warmstart to, instead of creating an equal superposition
        :param bool use_optimizer: indicates whether to optimize theta using classical optimizers
        :param bool print_res: indicates whether the results of the optimization should be printed
    """
    # define expectation function for optimizers
    expectation = get_expectation(problem_circuit, hamiltonian, n_features, layers, shots, warmstart_statevector)

    # optimize beta and gamma
    if use_optimizer:
        res = minimize(expectation, [theta["beta"], theta["gamma"]], method='COBYLA', tol=1e-12)
        if print_res:
            print(res)
        theta = {"beta": res.x[0], "gamma": res.x[1]}

        # run qaoa circuit with parameters in theta
    counts, qc = quantum(problem_circuit, hamiltonian, n_features, layers, theta["beta"], theta["gamma"], shots,
                         warmstart_statevector)
    return counts, qc


def quantum_statevector(problem_circuit: Callable, hamiltonian: DictArithmetic, nqubits: int, layers: int,
                        beta_val: float, gamma_val: float, amplitude_vector: list[float] = None)\
        -> tuple[list[float], QuantumCircuit]:
    qc, beta, gamma = qaoa_circuit(problem_circuit, hamiltonian, nqubits, layers, amplitude_vector, measure=False)

    # Set parameters for qc
    qc = qc.bind_parameters({
        beta: beta_val,
        gamma: gamma_val
    })

    # run and measure qc
    statevector_sim = StatevectorSimulator()
    transpiled_qaoa = transpile(qc, statevector_sim)
    result = statevector_sim.run(transpiled_qaoa).result()
    probabilities = result.get_statevector().probabilities()

    return probabilities, qc


def get_expectation_statevector(problem_circuit: Callable, hamiltonian: DictArithmetic, nqubits: int, nlayers: int,
                                amplitude_vector: list[float] = None, strategy: str = 'min') -> Callable:
    backend = StatevectorSimulator()

    def execute_circ(theta):
        qc, beta, gamma = qaoa_circuit(problem_circuit, hamiltonian, nqubits, nlayers, amplitude_vector, measure=False)

        # Set parameters for qc
        qc = qc.bind_parameters({
            beta: theta[0],
            gamma: theta[1]
        })

        statevector = backend.run(qc).result().get_statevector()

        return compute_hamiltonian_energy_from_statevector(hamiltonian, statevector, nqubits, strategy=strategy)

    return execute_circ


def apply_qaoa_statevector(problem_circuit: Callable, hamiltonian: DictArithmetic, layers: int = 60,
                           n_features: int = 6, theta={"beta": 0.01, "gamma": -0.01},
                           warmstart_statevector: list[float] = None, use_optimizer: bool = True, print_res: bool = True)\
        -> tuple[list[float], QuantumCircuit]:
    """
        Applies the QAOA Algorithm for the given hamiltonian in QUSO form.

        :param problem_circuit: The function for creating the corresponding problem quantum circuit
        :param hamiltonian: the hamiltonian used for creating the quantum circuit
                            and determining the expected config cost
        :param int layers: the hyperparameter p of QAOA defining how many cost-mixer-layers will be in the circuit
        :param int n_features: the number of independent variables in the input hamiltonian
        :param dict theta: dictionary with keys "beta" and "gamma" that parameterize the QAOA circuit,
                           used as start value when optimizing
        :param list warmstart_statevector: statevector to warmstart to, instead of creating an equal superposition
        :param bool use_optimizer: indicates whether to optimize theta using classical optimizers
        :param bool print_res: indicates whether the results of the optimization should be printed
    """
    # define expectation function for optimizers
    expectation = get_expectation_statevector(problem_circuit, hamiltonian, n_features, layers, warmstart_statevector)

    # optimize beta and gamma
    if use_optimizer:
        res = minimize(expectation, [theta["beta"], theta["gamma"]], method='COBYLA', tol=1e-12)
        if print_res:
            print(res)
        theta = {"beta": res.x[0], "gamma": res.x[1]}

    probabilities, qc = quantum_statevector(problem_circuit, hamiltonian, n_features, layers, theta["beta"],
                                            theta["gamma"], warmstart_statevector)
    return probabilities, qc
