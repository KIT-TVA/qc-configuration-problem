import math
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit import Parameter
from qiskit.result import Counts
from qiskit_aer import StatevectorSimulator
from qubovert.utils import DictArithmetic
from scipy.optimize import minimize

from configproblem.fragments.quantum_states import superposition_circuit
from configproblem.util.hamiltonian_math import compute_hamiltonian_energy, compute_hamiltonian_energy_from_statevector


def mixer_circuit(nqubits: int, beta: Parameter) -> QuantumCircuit:
    """
        Creates a mixer circuit for the given number of qubits

        :param nqubits: The number of qubits to create the circuit for
        :param beta: The parameter to use for the circuit
    """
    qc_mix = QuantumCircuit(nqubits)
    for i in range(0, nqubits):
        qc_mix.rx(2 * beta, i)
    return qc_mix


def get_expectation_params_per_layer(problem_circuit: Callable, hamiltonian: DictArithmetic, nqubits: int,
                                     nlayers: int, shots: int = 128, amplitude_vector: list[float] = None,
                                     strategy: str = 'avg') -> Callable:
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = shots

    def execute_circ(theta: list[float]):
        qc, beta_list, gamma_list = qaoa_circuit(problem_circuit, hamiltonian, nqubits, nlayers,
                                                 amplitude_vector=amplitude_vector, params_per_layer=True)
        # Set parameters for qc
        for i in range(math.floor(len(theta) / 2)):
            qc = qc.bind_parameters({
                beta_list[i]: theta[2 * i],
                gamma_list[i]: theta[(2 * i) + 1]
            })
        counts = backend.run(qc, nshots=shots).result().get_counts()
        return compute_hamiltonian_energy(hamiltonian, counts, strategy=strategy)

    return execute_circ


def apply_qaoa_params_per_layer(problem_circuit: Callable, hamiltonian: DictArithmetic, layers: int = 60,
                                n_features: int = 6, shots: int = 256, theta: dict = {"beta": 0.01, "gamma": -0.01},
                                warmstart_statevector: list[float] = None, strategy: str = 'avg',
                                use_optimizer: bool = True, print_res: bool = True) -> tuple[Counts, QuantumCircuit]:
    expectation = get_expectation_params_per_layer(problem_circuit, hamiltonian, n_features, layers, shots,
                                                   warmstart_statevector, strategy=strategy)

    beta_val_list, gamma_val_list = determine_params_per_layer(layers, theta, expectation, use_optimizer, print_res)

    counts, qc = quantum(problem_circuit, hamiltonian, n_features, layers, beta_val_list,
                         gamma_val_list, shots, amplitude_vector=warmstart_statevector, params_per_layer=True)
    return counts, qc


def get_expectation_statevector_params_per_layer(problem_circuit: Callable, hamiltonian: DictArithmetic,
                                                 nqubits: int, nlayers: int, amplitude_vector: list[float] = None,
                                                 strategy: str = 'avg') -> Callable:
    backend = StatevectorSimulator()

    def execute_circ(theta: list[float]):
        qc, beta_list, gamma_list = qaoa_circuit(problem_circuit, hamiltonian, nqubits, nlayers,
                                                 amplitude_vector=amplitude_vector, params_per_layer=True)
        # Set parameters for qc
        for i in range(math.floor(len(theta) / 2)):
            qc = qc.bind_parameters({
                beta_list[i]: theta[2 * i],
                gamma_list[i]: theta[(2 * i) + 1]
            })
        statevector = backend.run(qc).result().get_statevector()
        return compute_hamiltonian_energy_from_statevector(hamiltonian, statevector, nqubits, strategy=strategy)

    return execute_circ


def apply_qaoa_statevector_params_per_layer(problem_circuit: Callable, hamiltonian: DictArithmetic, layers: int = 60,
                                            n_features: int = 6, theta={"beta": 0.01, "gamma": -0.01},
                                            warmstart_statevector: list[float] = None, strategy: str = 'avg',
                                            use_optimizer: bool = True, print_res: bool = True)\
        -> tuple[list[float], QuantumCircuit]:
    """
        Applies the QAOA Algorithm for the given hamiltonian in QUSO/PUSO form.

        :param problem_circuit: The function for creating the corresponding problem quantum circuit
        :param hamiltonian: the hamiltonian used for creating the quantum circuit
                            and determining the expected config cost
        :param layers: the hyperparameter p of QAOA defining how many cost-mixer-layers will be in the circuit
        :param n_features: the number of independent variables in the input hamiltonian
        :param theta: dictionary with keys "beta" and "gamma" that parameterize the QAOA circuit,
                      used as start value when optimizing
        :param warmstart_statevector: statevector to warmstart to, instead of creating an equal superposition
        :param strategy: the strategy used to compute the expected config cost
        :param use_optimizer: indicates whether to optimize theta using classical optimizers
        :param print_res: indicates whether the results of the optimization should be printed
    """
    # define expectation function for optimizers
    expectation = get_expectation_statevector_params_per_layer(problem_circuit, hamiltonian, n_features, layers,
                                                               warmstart_statevector, strategy)
    beta_val_list, gamma_val_list = determine_params_per_layer(layers, theta, expectation, use_optimizer, print_res)

    probabilities, qc = quantum_statevector(problem_circuit, hamiltonian, n_features, layers, beta_val_list,
                                            gamma_val_list, warmstart_statevector, params_per_layer=True)
    return probabilities, qc


def determine_params_per_layer(layers: int, theta: dict, expectation: Callable, use_optimizer: bool, print_res: bool)\
        -> tuple[list[float], list[float]]:
    # Prepare initial parameter list
    parameters = []
    for i in range(layers):
        parameters.append(theta)

    # optimize parameters
    if use_optimizer:
        parameters = optimize_parameters_per_layer(parameters, expectation, print_res)

    # get list of beta and gamma values from parameters
    beta_val_list = []
    gamma_val_list = []
    for parameter in parameters:
        beta_val_list.append(parameter["beta"])
        gamma_val_list.append(parameter["gamma"])

    return beta_val_list, gamma_val_list


def qaoa_circuit(problem_circuit: Callable, hamiltonian: DictArithmetic, nqubits: int, nlayers: int,
                 amplitude_vector: list[float] = None, params_per_layer: bool = True, measure: bool = True)\
        -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    """
        Creates a QAOA circuit for the given hamiltonian

        :param problem_circuit: The method for creating the corresponding problem quantum circuit
        :param hamiltonian: The hamiltonian to create the circuit for
        :param nqubits: The number of qubits to create the circuit for
        :param nlayers: The number of layers to create the circuit for
        :param amplitude_vector: The amplitude vector to use for the circuit
        :param params_per_layer: indicates whether a unique parameter pair should be used for each layer
        :param measure: Whether to measure the circuit
    """
    if amplitude_vector is not None:
        # warm start
        qc = QuantumCircuit(nqubits)
        qc.initialize(amplitude_vector)
    else:
        # equal superposition
        qc = superposition_circuit(nqubits)

    # define parameters
    if params_per_layer:
        beta_list = [Parameter("$\\beta_{}$".format(i)) for i in range(nlayers)]
        gamma_list = [Parameter("$\\gamma_{}$".format(i)) for i in range(nlayers)]
    else:
        beta_list = ["$\\beta$"]
        gamma_list = ["$\\gamma$"]

    for i in range(nlayers):
        qc.barrier()
        qg_problem = problem_circuit(hamiltonian, nqubits, gamma_list[i]) if params_per_layer \
            else problem_circuit(hamiltonian, nqubits, gamma_list[0])
        qc = qc.compose(qg_problem)
        qc.barrier()
        qg_mixer = mixer_circuit(nqubits, beta_list[i]) if params_per_layer \
            else mixer_circuit(nqubits, beta_list[0])
        qc = qc.compose(qg_mixer)

    if measure:
        qc.measure_all()
    return qc, beta_list, gamma_list


def quantum(problem_circuit: Callable, hamiltonian: DictArithmetic, nqubits: int, layers: int,
            beta_val_list: list[float], gamma_val_list: list[float], shots: int = 512,
            amplitude_vector: list[float] = None, params_per_layer: bool = True) -> tuple[Counts, QuantumCircuit]:
    qc, beta_list, gamma_list = qaoa_circuit(problem_circuit, hamiltonian, nqubits, layers,
                                             amplitude_vector=amplitude_vector, params_per_layer=params_per_layer)

    # Set parameters for qc
    for i in range(len(beta_val_list)):
        qc = qc.bind_parameters({
            beta_list[i]: beta_val_list[i],
            gamma_list[i]: gamma_val_list[i]
        })

    # run and measure qc
    qasm_sim = Aer.get_backend('qasm_simulator')
    transpiled_qaoa = transpile(qc, qasm_sim)
    results = qasm_sim.run(transpiled_qaoa, shots=shots).result()
    counts = results.get_counts()

    return counts, qc


def get_expectation(problem_circuit: Callable, hamiltonian: DictArithmetic, nqubits: int, nlayers: int,
                    shots: int = 128, amplitude_vector: list[float] = None, strategy: str = 'avg') -> Callable:
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = shots

    def execute_circ(theta):
        qc, beta, gamma = qaoa_circuit(problem_circuit, hamiltonian, nqubits, nlayers,
                                       amplitude_vector=amplitude_vector, params_per_layer=False)

        # Set parameters for qc
        qc = qc.bind_parameters({
            beta: theta[0],
            gamma: theta[1]
        })

        counts = backend.run(qc, nshots=shots).result().get_counts()
        return compute_hamiltonian_energy(hamiltonian, counts, strategy=strategy)

    return execute_circ


def apply_qaoa(problem_circuit: Callable, hamiltonian: DictArithmetic, layers: int = 60, n_features: int = 6,
               shots: int = 256, theta={"beta": 0.01, "gamma": -0.01}, warmstart_statevector: list[float] = None,
               strategy: str = 'avg', use_optimizer: bool = True, params_per_layer: bool = True,
               print_res: bool = True) -> tuple[Counts, QuantumCircuit]:
    """
        Applies the QAOA Algorithm for the given problem hamiltonian in QUSO/PUSO form.

        :param problem_circuit: The function for creating the corresponding problem quantum circuit
        :param hamiltonian: the hamiltonian used for creating the quantum circuit
                            and determining the expected config cost
        :param layers: the hyperparameter p of QAOA defining how many cost-mixer-layers will be in the circuit
        :param n_features: the number of independent variables in the input hamiltonian
        :param shots: the number of shots used in a simulator run of the QAOA quantum circuit
        :param theta: dictionary with keys "beta" and "gamma" that parameterize the QAOA circuit,
                      used as start value when optimizing
        :param warmstart_statevector: statevector to warmstart to, instead of creating an equal superposition
        :param strategy: the strategy used to compute the expected config cost
        :param use_optimizer: indicates whether to optimize theta using classical optimizers
        :param params_per_layer: indicates whether a unique parameter pair should be used for each layer
        :param print_res: indicates whether the results of the optimization should be printed
    """
    if params_per_layer:
        return apply_qaoa_params_per_layer(problem_circuit, hamiltonian, layers, n_features, shots, theta,
                                           warmstart_statevector, strategy=strategy, use_optimizer=use_optimizer,
                                           print_res=print_res)

    # define expectation function for optimizers
    expectation = get_expectation(problem_circuit, hamiltonian, n_features, layers, shots, warmstart_statevector,
                                  strategy=strategy)

    # optimize beta and gamma
    if use_optimizer:
        theta = optimize_parameters(theta, expectation, print_res)

        # run qaoa circuit with parameters in theta
    counts, qc = quantum(problem_circuit, hamiltonian, n_features, layers, [theta["beta"]], [theta["gamma"]], shots,
                         amplitude_vector=warmstart_statevector, params_per_layer=False)
    return counts, qc


def quantum_statevector(problem_circuit: Callable, hamiltonian: DictArithmetic, nqubits: int, layers: int,
                        beta_val_list: list[float], gamma_val_list: list[float], amplitude_vector: list[float] = None,
                        params_per_layer: bool = True) -> tuple[list[float], QuantumCircuit]:
    qc, beta_list, gamma_list = qaoa_circuit(problem_circuit, hamiltonian, nqubits, layers,
                                             amplitude_vector=amplitude_vector, params_per_layer=params_per_layer,
                                             measure=False)

    # Set parameters for qc
    for i in range(len(beta_val_list)):
        qc = qc.bind_parameters({
            beta_list[i]: beta_val_list[i],
            gamma_list[i]: gamma_val_list[i]
        })

    # run and measure qc
    statevector_sim = StatevectorSimulator()
    transpiled_qaoa = transpile(qc, statevector_sim)
    result = statevector_sim.run(transpiled_qaoa).result()
    probabilities = result.get_statevector().probabilities()

    return probabilities, qc


def get_expectation_statevector(problem_circuit: Callable, hamiltonian: DictArithmetic, nqubits: int, nlayers: int,
                                amplitude_vector: list[float] = None, strategy: str = 'avg') -> Callable:
    backend = StatevectorSimulator()

    def execute_circ(theta):
        qc, beta, gamma = qaoa_circuit(problem_circuit, hamiltonian, nqubits, nlayers,
                                       amplitude_vector=amplitude_vector, params_per_layer=False, measure=False)

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
                           warmstart_statevector: list[float] = None, strategy: str = 'avg', use_optimizer: bool = True,
                           params_per_layer: bool = True, print_res: bool = True) -> tuple[list[float], QuantumCircuit]:
    """
        Applies the QAOA Algorithm for the given hamiltonian in QUSO/PUSO form.

        :param problem_circuit: The function for creating the corresponding problem quantum circuit
        :param hamiltonian: the hamiltonian used for creating the quantum circuit
                            and determining the expected config cost
        :param layers: the hyperparameter p of QAOA defining how many cost-mixer-layers will be in the circuit
        :param n_features: the number of independent variables in the input hamiltonian
        :param theta: dictionary with keys "beta" and "gamma" that parameterize the QAOA circuit,
                      used as start value when optimizing
        :param warmstart_statevector: statevector to warmstart to, instead of creating an equal superposition
        :param strategy: the strategy used to compute the expected config cost
        :param use_optimizer: indicates whether to optimize theta using classical optimizers
        :param params_per_layer: indicates whether a unique parameter pair should be used for each layer
        :param print_res: indicates whether the results of the optimization should be printed
    """
    if params_per_layer:
        return apply_qaoa_statevector_params_per_layer(problem_circuit, hamiltonian, layers, n_features, theta,
                                                       warmstart_statevector, strategy=strategy,
                                                       use_optimizer=use_optimizer, print_res=print_res)

    # define expectation function for optimizers
    expectation = get_expectation_statevector(problem_circuit, hamiltonian, n_features, layers, warmstart_statevector,
                                              strategy=strategy)

    # optimize beta and gamma
    if use_optimizer:
        theta = optimize_parameters(theta, expectation, print_res)

    probabilities, qc = quantum_statevector(problem_circuit, hamiltonian, n_features, layers, [theta["beta"]],
                                            [theta["gamma"]], warmstart_statevector, params_per_layer=False)
    return probabilities, qc


def optimize_parameters_per_layer(theta: list[dict], expectation: Callable, print_res: bool) -> list[dict]:
    parameter_list = []
    for parameter_pair in theta:
        parameter_list.append(parameter_pair["beta"])
        parameter_list.append(parameter_pair["gamma"])

    res = minimize(expectation, np.array(parameter_list), method='COBYLA',
                   options={'maxiter': 1000, 'disp': False}, tol=1e-12)
    if print_res:
        print(res)
    optimized_theta = []
    for i in range(len(theta)):
        optimized_theta.append({"beta": res.x[2 * i], "gamma": res.x[(2 * i) + 1]})
    return optimized_theta


def optimize_parameters(theta: dict, expectation: Callable, print_res: bool) -> dict:
    """
        Optimizes the parameters beta and gamma of the QAOA circuit using classical optimizers.

        :param theta: dictionary with keys "beta" and "gamma" that parameterize the QAOA circuit,
                      used as start value when optimizing
        :param expectation: the expectation function used to determine the expected config cost
        :param print_res: indicates whether the results of the optimization should be printed
    """
    res = minimize(expectation, np.array([theta["beta"], theta["gamma"]]), method='COBYLA',
                   options={'maxiter': 1000, 'disp': False}, tol=1e-12)
    if print_res:
        print(res)
    return {"beta": res.x[0], "gamma": res.x[1]}
