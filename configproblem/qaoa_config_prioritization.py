import numpy as np
from qaoa_mincost_sat import apply_qaoa_statevector
from qubovert.utils import DictArithmetic
from util.hamiltonian_math import compute_config_energy
from util.visualization import plot_counts_histogram

best_config = "000110"  # 654321
valid_configs = ["101010", "101001", "101011", "100110", "100101", "100111", "001010", "001001", "001011", "000101",
                 "000111", "111010", "111001", "111011", "110110", "110101", "110111"]


def deflate_config(hamiltonian, config_str, deflation_factor=300):
    """
        Deflates the hamiltonian with the given config.

        :param hamiltonian: the hamiltonian of the optimization problem
        :param config_str: the config to deflate the hamiltonian with
        :param deflation_factor: the deflation factor to use
    """

    config_array = np.array([[0 if s == "0" else 1 for s in config_str]])
    deflation_matrix = np.matmul(config_array.transpose(), config_array)

    deflation_dict_arithmetic = DictArithmetic()
    for i, i_val in enumerate(deflation_matrix):
        for j, j_val in enumerate(i_val):
            deflation_dict_arithmetic[(i, j)] = j_val * deflation_factor

    return hamiltonian + deflation_dict_arithmetic


def config_prioritization(hamiltonian, output_list_size, deflation_factor_start_value):
    """
        Creates an ordered list of n configurations with minimal cost (ascending)
        by deflating the hamiltonian with the configs that have already been found.

        :param hamiltonian: the hamiltonian of the optimization problem
        :param output_list_size: the size of the output list
        :param deflation_factor_start_value: the deflation factor to start with
    """

    current_hamiltonian = hamiltonian
    output_list = []

    deflation_factor_value = deflation_factor_start_value
    i = 0
    while i < output_list_size:
        restart = False
        probabilities, qc = apply_qaoa_statevector(current_hamiltonian, print_res=False)
        probabilities_dict = {}
        for j in range(0, 2 ** 6):
            probabilities_dict[(np.binary_repr(j, width=6))] = round(probabilities[j], 4)

        plot_counts_histogram(probabilities_dict, best_config, valid_configs)

        current_config = max(probabilities_dict, key=probabilities_dict.get)
        current_config_energy = compute_config_energy(hamiltonian, [-1 if s == "0" else 1 for s in current_config])

        for config in output_list:
            energy = compute_config_energy(hamiltonian, [-1 if s == "0" else 1 for s in config])
            if deflation_factor_value - energy <= current_config_energy - energy or current_config in output_list:
                deflation_factor_value *= 2
                output_list.clear()
                current_hamiltonian = hamiltonian
                i = 0
                restart = True
                print("Restarting with deflation factor: " + str(deflation_factor_value))
                break
        if restart:
            continue
        output_list.append(current_config)
        current_hamiltonian = deflate_config(current_hamiltonian, current_config,
                                             deflation_factor=deflation_factor_value - current_config_energy)
        print("Current hamiltonian: " + str(current_hamiltonian))
        i += 1

    return output_list
