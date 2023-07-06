import numpy as np
from qaoa_mincost_sat import apply_qaoa_statevector
from qubovert.utils import DictArithmetic
from util.visualization import plot_counts_histogram


def deflate_config(hamiltonian, config_str, deflation_factor=300):
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
        probabilities, qc = apply_qaoa_statevector(current_hamiltonian)
        probabilities_dict = {}
        for i in range(0, 2 ** 6):
            probabilities_dict[(np.binary_repr(i, width=6))] = round(probabilities[i], 4)

        current_config = max(probabilities_dict, key=probabilities_dict.get)
        output_list.append(current_config)

        current_hamiltonian = deflate_config(hamiltonian, current_config)

    return output_list
