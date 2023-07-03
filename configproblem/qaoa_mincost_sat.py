from qiskit import Aer, transpile, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, basinhopping
from qubovert.utils import DictArithmetic

from pprint import pprint
import math

from fragments.quantum_states import superposition_circuit
from util.hamiltonian_math import compute_hamiltonian_energy

def mixer_circuit(nqubits: int) -> QuantumCircuit:
    beta = Parameter("$\\beta$")
    qc_mix = QuantumCircuit(nqubits)
    for i in range(0, nqubits):
        qc_mix.rx(2 * beta, i)
    return qc_mix, beta


def problem_circuit(hamiltonian, nqubits: int) -> QuantumCircuit:
    gamma = Parameter("$\\gamma$")
    qc_p = QuantumCircuit(nqubits)
    normalization = math.pi/max(hamiltonian.values())
    
    for key, factor in hamiltonian.items():
        acting_qubits = len(key)
        
        if acting_qubits == 0:
            pass # identity case
        elif acting_qubits == 1:
            # single qubit term
            q1 = key[0]
            qc_p.rz(2 * normalization * factor * gamma, q1)
        elif acting_qubits ==2:
            # quadratic qubit term
            q1 = key[0]
            q2 = key[1]
            qc_p.rzz(2 * normalization * factor * gamma, q1, q2)
        else:
            # non quadratic, error
            RuntimeError(f"Non quadratic term in hamiltonian: {key, factor}")
            
    return qc_p, gamma


def qaoa_circuit(hamiltonian, nqubits, nlayers, amplitude_vector=None, measure=True) -> QuantumCircuit:
    if amplitude_vector is not None:
        # warm start
        qc = QuantumCircuit(nqubits) 
        qc.initialize(amplitude_vector)
    else:
        # equal superposition
        qc = superposition_circuit(nqubits)
        
    qg_mixer, beta = mixer_circuit(nqubits)
#     qg_mixer.name = "Mixer"
    qg_problem, gamma = problem_circuit(hamiltonian, nqubits)
#     qg_problem.name = "Problem"
    
    for l in range(nlayers):
        qc.barrier()
        qc = qc.compose(qg_problem)
        qc.barrier()
        qc = qc.compose(qg_mixer)

    if measure:
        qc.measure_all()
    return qc, beta, gamma


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


def apply_qaoa(hamiltonian, layers=60, n_features=6, shots=256, theta={"beta": 0.01, "gamma": -0.01}, warmstart_statevector=None, use_optimizer=True):
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


def deflate_config(hamiltonian, config_str, deflation_factor=1000):
    # TODO add docstring

    config_array = np.array([[0 if s == "0" else 1 for s in config_str]])
    deflation_matrix = np.matmul(config_array.transpose(), config_array)

    deflation_dict_arithmetic = DictArithmetic()
    for i, i_val in enumerate(deflation_matrix):
        for j, j_val in enumerate(i_val):
            deflation_dict_arithmetic[(i, j)] = j_val * deflation_factor

    return hamiltonian + deflation_dict_arithmetic


def config_prioritization(hamiltonian, output_list_size):
    # TODO add docstring

    current_hamiltonian = hamiltonian
    output_list = []

    for i in range(0, output_list_size):
        counts, qc = apply_qaoa(current_hamiltonian)
        current_config = max(counts, key=counts.get)
        output_list.append(current_config)

        current_hamiltonian = deflate_config(hamiltonian, current_config)

    return output_list
    