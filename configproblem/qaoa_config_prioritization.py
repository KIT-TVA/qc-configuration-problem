import numpy as np
from qaoa_mincost_sat import apply_qaoa_statevector
from qubovert.utils import QUSOMatrix
from util.hamiltonian_math import compute_config_energy
from util.visualization import plot_counts_histogram

best_config = "000110"  # 654321
valid_configs = ["101010", "101001", "101011", "100110", "100101", "100111", "001010", "001001", "001011", "000101",
                 "000111", "111010", "111001", "111011", "110110", "110101", "110111"]


def convert_quso_matrix_to_numpy_ndarray(quso_matrix: QUSOMatrix) -> np.ndarray:
    """
        Converts a QUSOMatrix to a numpy ndarray.

        :param quso_matrix: the QUSOMatrix to convert
    """
    matrix = np.zeros((6, 6))
    for key, value in quso_matrix.items():
        acting_qubits = len(key)
        if acting_qubits == 0:
            # identity case -> ignore
            pass
        elif acting_qubits == 1:
            # single qubit term
            q1 = key[0]
            matrix[q1, q1] = value
        elif acting_qubits == 2:
            # quadratic qubit term
            q1 = key[0]
            q2 = key[1]
            matrix[q1, q2] = value
            matrix[q2, q1] = value
        else:
            # non quadratic, error
            raise RuntimeError(f"Non quadratic term in hamiltonian: {key, value}")
    return matrix


def convert_numpy_ndarray_to_quso_matrix(matrix: np.ndarray) -> QUSOMatrix:
    """
        Converts a numpy ndarray to a QUSOMatrix.

        :param matrix: the numpy ndarray to convert
    """
    quso_matrix = QUSOMatrix()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            quso_matrix[i, j] += matrix[i, j]
    return quso_matrix


def get_deflation_quso_matrix(config_str: str, deflation_factor: float) -> QUSOMatrix:
    """
        Calculated the deflation matrix as DictArithmetic for a given config and deflation factor.

        :param config_str: the config to calculate the deflation matrix from
        :param deflation_factor: the deflation factor to multiply the matrix with
    """
    config_array = np.array([[0 if s == "0" else 1 for s in config_str]])
    deflation_matrix = np.matmul(config_array.transpose(), config_array)

    return convert_numpy_ndarray_to_quso_matrix(deflation_matrix * deflation_factor)


def deflate_config(hamiltonian: QUSOMatrix, config_str: str, deflation_factor: float) -> QUSOMatrix:
    """
        Deflates the hamiltonian with the given config.

        :param hamiltonian: the hamiltonian of the optimization problem
        :param config_str: the config to deflate the hamiltonian with
        :param deflation_factor: the deflation factor to use
    """
    return hamiltonian + get_deflation_quso_matrix(config_str, deflation_factor)


def get_probabilities_dict(hamiltonian: QUSOMatrix) -> dict[str, float]:
    """
        Calculates the probabilities for all configs of a given hamiltonian.

        :param hamiltonian: the hamiltonian of the optimization problem
    """
    probabilities, _ = apply_qaoa_statevector(hamiltonian, print_res=False)
    probabilities_dict = {}
    for j in range(0, 2 ** 6):
        probabilities_dict[(np.binary_repr(j, width=6))] = round(probabilities[j], 4)
    return probabilities_dict


def strategy_projection_deflation(hamiltonian: QUSOMatrix, output_list_size: int, deflation_factor_start_value: float)\
        -> list[str]:
    """
        Calculates the output list for the given hamiltonian using the projection deflation strategy.

        :param hamiltonian: the hamiltonian of the optimization problem
        :param output_list_size: the size of the output list
        :param deflation_factor_start_value: the start value for the deflation factor
    """
    current_hamiltonian = hamiltonian
    output_list = []
    config_energies = []

    deflation_factor_value = deflation_factor_start_value
    i = 0
    while i < output_list_size:
        print(current_hamiltonian)
        probabilities_dict = get_probabilities_dict(current_hamiltonian)

        plot_counts_histogram(probabilities_dict, best_config, valid_configs)

        output_list.append(max(probabilities_dict, key=probabilities_dict.get))
        config_energies.append(compute_config_energy(hamiltonian, [-1 if s == "0" else 1 for s in output_list[-1]]))

        # check if current config energy is maximum
        if not config_energies[-1] <= deflation_factor_value or output_list[-1] in output_list[:-1]:
            deflation_factor_value *= 2
            output_list.clear()
            config_energies.clear()
            current_hamiltonian = hamiltonian
            i = 0
            print("Restarting with deflation factor: " + str(deflation_factor_value))
            continue

        # Determine A_k+1
        deflation_matrix = np.identity(6)
        for config in output_list:
            config_array = np.array([[0 if s == "0" else 1 for s in config]])
            config_matrix = np.identity(6) - np.matmul(config_array.transpose(), config_array)

            deflation_matrix = np.matmul(deflation_matrix, config_matrix)

        # Determine H_k+1
        hamiltonian_ndarray = convert_quso_matrix_to_numpy_ndarray(current_hamiltonian)
        new_hamiltonian_ndarray = np.matmul(deflation_matrix.transpose(),
                                            np.matmul(hamiltonian_ndarray - deflation_factor_value * np.identity(6),
                                                      deflation_matrix))
        current_hamiltonian = convert_numpy_ndarray_to_quso_matrix(new_hamiltonian_ndarray)
        i += 1

    return output_list


def strategy_variational_quantum_deflation(hamiltonian: QUSOMatrix, output_list_size: int,
                                           deflation_factor_start_value: float) -> list[str]:
    """
        Calculates the output list for the given hamiltonian using the variational quantum deflation strategy.

        :param hamiltonian: the hamiltonian of the optimization problem
        :param output_list_size: the size of the output list
        :param deflation_factor_start_value: the start value for the deflation factor
    """
    current_hamiltonian = hamiltonian
    output_list = []

    deflation_factor_value = deflation_factor_start_value
    i = 0
    while i < output_list_size:
        print("Current hamiltonian: " + str(current_hamiltonian))
        restart = False
        probabilities_dict = get_probabilities_dict(current_hamiltonian)

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
        i += 1

    return output_list


def config_prioritization(hamiltonian: QUSOMatrix, output_list_size: int, deflation_factor_start_value: float,
                          strategy='vqd') -> list[str]:
    """
        Calculates the output list for the given hamiltonian using the given strategy.
        Available strategies are: 'vqd' (variational quantum deflation) and 'pd' (projection deflation).

        :param hamiltonian: the hamiltonian of the optimization problem
        :param output_list_size: the size of the output list
        :param deflation_factor_start_value: the start value for the deflation factor
        :param strategy: the strategy to use
    """
    if strategy == 'vqd':
        return strategy_variational_quantum_deflation(hamiltonian, output_list_size, deflation_factor_start_value)
    elif strategy == 'pd':
        return strategy_projection_deflation(hamiltonian, output_list_size, deflation_factor_start_value)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
