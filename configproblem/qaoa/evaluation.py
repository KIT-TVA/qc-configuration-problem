import sys
import pandas as pd
import numpy as np
from qiskit.circuit import Parameter

from ..qaoa.qaoa_application import apply_qaoa_statevector
from ..util.hamiltonian_math import get_hamiltonian_dimension
from ..util.problem_instance import ProblemInstance, generate_problem_instance_set
from .qaoa_mincost_k_sat import problem_circuit as puso_problem_circuit, convert_ancilla_bit_results
from .qaoa_mincost_sat import problem_circuit as quso_problem_circuit

theta = {"beta": 0.01, "gamma": -0.01}  # start values for optimization
use_warmstart = False
use_optimizer = True
print_res = False
warmstart_statevector = None


def run_puso_qaoa(instance: ProblemInstance, layers: int, strategy: str) -> dict:
    """
        Runs the QAOA algorithm for the given instance using its puso hamiltonian and returns the results

        :param instance: The instance to run the algorithm for
        :param layers: The number of layers to use for the algorithm
        :param strategy: The strategy to use for the algorithm
    """
    hamiltonian = instance.get_puso_combined_hamiltonian()

    probabilities, _ = apply_qaoa_statevector(puso_problem_circuit, hamiltonian, layers,
                                              get_hamiltonian_dimension(hamiltonian), theta, warmstart_statevector,
                                              strategy=strategy, use_optimizer=use_optimizer, print_res=print_res)
    probabilities_dict = {}
    for i in range(0, 2 ** get_hamiltonian_dimension(hamiltonian)):
        probabilities_dict[(np.binary_repr(i, width=get_hamiltonian_dimension(hamiltonian)))] = probabilities[i]

    probabilities_dict = instance.convert_solution_dict(probabilities_dict, 'puso_combined')

    return {'hamiltonian': hamiltonian,
            'success_probability': instance.get_success_probability(probabilities_dict),
            'best_config_probability': probabilities_dict[instance.get_best_config()],
            'result_quality': instance.get_result_quality(probabilities_dict),
            'problem_circuit_depth': puso_problem_circuit(hamiltonian, get_hamiltonian_dimension(hamiltonian),
                                                          Parameter("$\\gamma$")).depth(),
            'circuit_width': get_hamiltonian_dimension(hamiltonian)}


def run_quso_qaoa(instance: ProblemInstance, layers: int, strategy: str) -> dict:
    """
        Runs the QAOA algorithm for the given instance using its quso hamiltonian and returns the results

        :param instance: The instance to run the algorithm for
        :param layers: The number of layers to use for the algorithm
        :param strategy: The strategy to use for the algorithm
    """
    hamiltonian = instance.get_quso_combined_hamiltonian()

    probabilities, _ = apply_qaoa_statevector(quso_problem_circuit, hamiltonian, layers,
                                              get_hamiltonian_dimension(hamiltonian), theta, warmstart_statevector,
                                              strategy=strategy, use_optimizer=use_optimizer, print_res=print_res)
    probabilities_dict = {}
    for i in range(0, 2 ** get_hamiltonian_dimension(hamiltonian)):
        probabilities_dict[np.binary_repr(i, width=get_hamiltonian_dimension(hamiltonian))] = probabilities[i]

    probabilities_dict = convert_ancilla_bit_results(probabilities_dict, instance.get_num_features())

    probabilities_dict = instance.convert_solution_dict(probabilities_dict, 'quso_combined')

    return {'hamiltonian': hamiltonian,
            'success_probability': instance.get_success_probability(probabilities_dict),
            'best_config_probability': probabilities_dict[instance.get_best_config()],
            'result_quality': instance.get_result_quality(probabilities_dict),
            'problem_circuit_depth': puso_problem_circuit(hamiltonian, get_hamiltonian_dimension(hamiltonian),
                                                          Parameter("$\\gamma$")).depth(),
            'circuit_width': get_hamiltonian_dimension(hamiltonian)}


def run_instance(instance: ProblemInstance, layers: int, strategy: str) -> dict:
    """
        Runs the QAOA algorithm for the given instance and returns the results

        :param instance: The instance to run the algorithm for
        :param layers: The number of layers to use for the algorithm
        :param strategy: The strategy to use for the algorithm
    """
    puso_results = run_puso_qaoa(instance, layers, strategy)
    quso_results = run_quso_qaoa(instance, layers, strategy)

    min_literals_per_clause = sys.maxsize
    max_literals_per_clause = 0
    for clause in instance.get_sat_instance():
        if len(clause) < min_literals_per_clause:
            min_literals_per_clause = len(clause)
        if len(clause) > max_literals_per_clause:
            max_literals_per_clause = len(clause)

    return {'n_features': instance.get_num_features(),
            'n_clauses': len(instance.get_sat_instance()),
            'min_literals_per_clause': min_literals_per_clause,
            'max_literals_per_clause': max_literals_per_clause,
            'feature_cost': instance.get_feature_cost(),
            'n_valid_configs': len(instance.get_valid_configs()),
            'n_layers': layers,
            'strategy': strategy,
            'success_probability_puso': instance.get_success_probability(puso_results['probabilities']),
            'best_config_probability_puso': puso_results['probabilities'][instance.get_best_config()],
            'result_quality_puso': instance.get_result_quality(puso_results['probabilities']),
            'problem_circuit_depth_puso': puso_problem_circuit(puso_results['hamiltonian'],
                                                               get_hamiltonian_dimension(puso_results['hamiltonian']),
                                                               Parameter("$\\gamma$")).depth(),
            'circuit_width_puso': get_hamiltonian_dimension(puso_results['hamiltonian']),
            'success_probability_quso': instance.get_success_probability(quso_results['probabilities']),
            'best_config_probability_quso': quso_results['probabilities'][instance.get_best_config()],
            'result_quality_quso': instance.get_result_quality(quso_results['probabilities']),
            'problem_circuit_depth_quso': puso_problem_circuit(quso_results['hamiltonian'],
                                                               get_hamiltonian_dimension(quso_results['hamiltonian']),
                                                               Parameter("$\\gamma$")).depth(),
            'circuit_width_quso': get_hamiltonian_dimension(quso_results['hamiltonian'])}


def run_instances_and_create_dataframe(instances: list[ProblemInstance], layers: int, strategy: str) -> pd.DataFrame:
    """
        Runs the QAOA algorithm for the given instances and returns the results as a dataframe

        :param instances: The instances to run the algorithm for
        :param layers: The number of layers to use for the algorithm
        :param strategy: The strategy to use for the algorithm
    """
    results = []
    for instance in instances:
        results.append(run_instance(instance, layers, strategy))

    return pd.DataFrame(results)


def run_with_different_num_layers(instances: list[ProblemInstance], layers: list[int], strategy: str) -> pd.DataFrame:
    """
        Runs the QAOA algorithm for the given instances and returns the results as a dataframe

        :param instances: The instances to run the algorithm for
        :param layers: The number of layers to use for the algorithm
        :param strategy: The strategy to use for the algorithm
    """
    dataframes = []
    for layer in layers:
        dataframes.append(run_instances_and_create_dataframe(instances, layer, strategy))

    return pd.concat(dataframes)


def generate_instance_set_different_number_of_clauses(n_instances_per_n_clauses: int, n_features: int,
                                                      n_clauses_list: list[int], n_literals_per_clause: int,
                                                      min_feature_cost: int, max_feature_cost: int, alpha_sat: float,
                                                      seed: int, generation_type: str) \
        -> list[ProblemInstance]:
    """
        Generates a set of problem instances with different number of clauses

        :param n_instances_per_n_clauses: The number of instances to generate
        :param n_features: The number of features to use
        :param n_clauses_list: The list of number of clauses to use
        :param n_literals_per_clause: The number of literals per clause to use
        :param min_feature_cost: The minimum feature cost to use
        :param max_feature_cost: The maximum feature cost to use
        :param alpha_sat: The alpha_sat to use
        :param seed: The seed to use
        :param generation_type: The generation type to use
    """
    instances = []
    for n_clauses in n_clauses_list:
        instances.extend(generate_problem_instance_set(n_instances_per_n_clauses, n_features, n_clauses, n_clauses,
                                                       n_literals_per_clause, n_literals_per_clause, min_feature_cost,
                                                       max_feature_cost, alpha_sat, seed, generation_type, None))
    return instances


def filter_out_unfulfillable_instances(instances: list[ProblemInstance]) -> list[ProblemInstance]:
    """
        Filters out unfulfillable instances from the given list of instances

        :param instances: The instances to filter
    """
    for instance in instances:
        if not instance.get_valid_configs():
            instances.remove(instance)
    return instances
