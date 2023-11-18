import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
from qiskit.circuit import Parameter

from configproblem.qaoa.qaoa_application import apply_qaoa_statevector
from configproblem.util.hamiltonian_math import get_hamiltonian_dimension
from configproblem.util.problem_instance import ProblemInstance, get_problem_instance_from_dimacs
from configproblem.qaoa.qaoa_mincost_k_sat import problem_circuit as puso_problem_circuit, convert_ancilla_bit_results
from configproblem.qaoa.qaoa_mincost_sat import problem_circuit as quso_problem_circuit

theta = {"beta": 0.01, "gamma": -0.01}  # start values for optimization
use_warmstart = False
use_optimizer = True
print_res = False
warmstart_statevector = None


def timing_decorator(func):
    execution_times = []

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_times.append(time.time() - start_time)
        return result

    wrapper.execution_times = execution_times
    return wrapper


@timing_decorator
def run_puso_qaoa(instance: ProblemInstance, layers: int, strategy: str, skip=False) -> dict:
    """
        Runs the QAOA algorithm for the given instance using its puso hamiltonian and returns the results

        :param instance: The instance to run the algorithm for
        :param layers: The number of layers to use for the algorithm
        :param strategy: The strategy to use for the algorithm
    """
    hamiltonian = instance.get_puso_combined_hamiltonian()

    if skip:
        probabilities_dict = {}
    else:
        probabilities, _ = apply_qaoa_statevector(puso_problem_circuit, hamiltonian, layers,
                                                  get_hamiltonian_dimension(hamiltonian), theta, warmstart_statevector,
                                                  strategy=strategy, use_optimizer=use_optimizer, print_res=print_res)
        probabilities_dict = {}
        for i in range(0, 2 ** get_hamiltonian_dimension(hamiltonian)):
            probabilities_dict[(np.binary_repr(i, width=get_hamiltonian_dimension(hamiltonian)))] = probabilities[i]

        probabilities_dict = instance.convert_solution_dict(probabilities_dict, 'puso_combined')

    return {'hamiltonian': hamiltonian,
            'probabilities': probabilities_dict,
            'problem_circuit_depth': puso_problem_circuit(hamiltonian, get_hamiltonian_dimension(hamiltonian),
                                                          Parameter("$\\gamma$")).depth(),
            'circuit_width': get_hamiltonian_dimension(hamiltonian)}


@timing_decorator
def run_quso_qaoa(instance: ProblemInstance, layers: int, strategy: str, skip=False) -> dict:
    """
        Runs the QAOA algorithm for the given instance using its quso hamiltonian and returns the results

        :param instance: The instance to run the algorithm for
        :param layers: The number of layers to use for the algorithm
        :param strategy: The strategy to use for the algorithm
        :param skip: Whether to skip running the algorithm for the quso hamiltonian
    """
    hamiltonian = instance.get_quso_combined_hamiltonian()

    if skip:
        probabilities_dict = {}
    else:
        probabilities, _ = apply_qaoa_statevector(quso_problem_circuit, hamiltonian, layers,
                                                  get_hamiltonian_dimension(hamiltonian), theta, warmstart_statevector,
                                                  strategy=strategy, use_optimizer=use_optimizer, print_res=print_res)
        probabilities_dict = {}
        for i in range(0, 2 ** get_hamiltonian_dimension(hamiltonian)):
            probabilities_dict[np.binary_repr(i, width=get_hamiltonian_dimension(hamiltonian))] = probabilities[i]

        probabilities_dict = convert_ancilla_bit_results(probabilities_dict, instance.get_num_features())

        probabilities_dict = instance.convert_solution_dict(probabilities_dict, 'quso_combined')

    return {'hamiltonian': hamiltonian,
            'probabilities': probabilities_dict,
            'problem_circuit_depth': puso_problem_circuit(hamiltonian, get_hamiltonian_dimension(hamiltonian),
                                                          Parameter("$\\gamma$")).depth(),
            'circuit_width': get_hamiltonian_dimension(hamiltonian)}


def run_instance(instance: ProblemInstance, layers: int, strategy: str, skip_quso: bool = False,
                 skip_puso: bool = False) -> dict:
    """
        Runs the QAOA algorithm for the given instance and returns the results

        :param instance: The instance to run the algorithm for
        :param layers: The number of layers to use for the algorithm
        :param strategy: The strategy to use for the algorithm
        :param skip_quso: Whether to skip running the algorithm for the quso hamiltonian
        :param skip_puso: Whether to skip running the algorithm for the puso hamiltonian
    """
    puso_results = run_puso_qaoa(instance, layers, strategy, skip=skip_puso)
    quso_results = run_quso_qaoa(instance, layers, strategy, skip=skip_quso)

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
            'valid_configs': instance.get_valid_configs(),
            'best_config': instance.get_best_config(),
            'n_layers': layers,
            'strategy': strategy,
            'execution_time_puso': run_puso_qaoa.execution_times[-1],
            'probabilities_puso': puso_results['probabilities'],
            'problem_circuit_depth_puso': puso_problem_circuit(puso_results['hamiltonian'],
                                                               get_hamiltonian_dimension(puso_results['hamiltonian']),
                                                               Parameter("$\\gamma$")).depth(),
            'circuit_width_puso': get_hamiltonian_dimension(puso_results['hamiltonian']),
            'execution_time_quso': run_quso_qaoa.execution_times[-1],
            'probabilities_quso': quso_results['probabilities'],
            'problem_circuit_depth_quso': puso_problem_circuit(quso_results['hamiltonian'],
                                                               get_hamiltonian_dimension(quso_results['hamiltonian']),
                                                               Parameter("$\\gamma$")).depth(),
            'circuit_width_quso': get_hamiltonian_dimension(quso_results['hamiltonian'])}


def run_instances_and_save_results(instances: list[ProblemInstance], layers: int, strategy: str,
                                   skip_quso: bool = False, skip_puso: bool = False):
    """
        Runs the QAOA algorithm for the given instances and returns the results as a dataframe

        :param instances: The instances to run the algorithm for
        :param layers: The number of layers to use for the algorithm
        :param strategy: The strategy to use for the algorithm
        :param skip_quso: Whether to skip running the algorithm for the quso hamiltonian
    """
    for index, instance in enumerate(instances):
        df = pd.DataFrame([run_instance(instance, layers, strategy, skip_quso=skip_quso, skip_puso=skip_puso)])
        df.to_csv(f"benchmarks\\qaoa-feature-models\\results\\feature_model_{index}.csv")
    return


def get_config_cost(config: str, feature_cost: list[int]) -> int:
    """
        Returns the cost of the given configuration

        :param config: The configuration to get the cost for
        :param feature_cost: The cost of each feature
    """
    cost = 0
    for i in range(len(config)):
        if config[i] == "1":
            cost += feature_cost[i]
    return cost


def get_valid_configs_sorted_by_cost(row: pd.Series) -> list[str]:
    """
        Returns the valid configurations sorted by cost for the given dataframe

        :param row: The row of the dataframe to get the valid configurations sorted by cost for
    """
    config_cost_dict = {}

    valid_configs = re.sub("[\[\] ']", "", row["valid_configs"]).split(",")
    feature_cost = list(map(int, re.sub("[\[\] ']", "", row["feature_cost"]).split(",")))
    for config in valid_configs:
        config_cost_dict[config] = get_config_cost(config, feature_cost)
    config_cost_dict = {k: v for k, v in sorted(config_cost_dict.items(), key=lambda item: item[1])}

    return list(config_cost_dict.keys())


def get_probability_dict(row: pd.Series, puso: bool = True) -> dict:
    """
        Returns the probability dictionary for the given row of the dataframe

        :param row: The row of the dataframe to get the probability dictionaries for
        :param puso: Whether to get the probability dictionaries for the puso or quso hamiltonian
    """
    if puso:
        probabilities_str = re.sub("[\{\} ']", "", row["probabilities_puso"]).split(",")
    else:
        probabilities_str = re.sub("[\{\} ']", "", row["probabilities_quso"]).split(",")

    probability_dict = {}
    for config_probability_pair in probabilities_str:
        config_probability_pair = config_probability_pair.split(":")
        config = config_probability_pair[0]
        probability = float(config_probability_pair[1])
        probability_dict[config] = probability
    return probability_dict


def get_result_quality(row: pd.Series, puso: bool = True) -> float:
    """
        Returns the result quality for the given row of the dataframe

        :param row: The row of the dataframe to get the result quality for
        :param puso: Whether to get the result quality for the puso or quso hamiltonian
    """
    probabilities = get_probability_dict(row, puso)
    valid_configs = get_valid_configs_sorted_by_cost(row)

    result_quality = 0
    for config in valid_configs:
        result_quality += probabilities[config] * 2**int(row["n_features"]) / len(valid_configs)
    return result_quality


def get_probability_of_best_n_configs(row: pd.Series, n: int, puso: bool = True) -> float:
    """
        Returns the probability of getting one of the best n configurations for the given row of the dataframe

        :param row: The row of the dataframe to get the probability of the best n configurations for
        :param n: The number of configurations to get the probability of
                  if n is greater than the number of configurations or the number of valid configurations,
                  the probability of all valid configurations is returned
        :param puso: Whether to get the probability of the best n configurations for the puso or quso hamiltonian
    """
    probabilities = get_probability_dict(row, puso)
    valid_configs = get_valid_configs_sorted_by_cost(row)

    current_n = len(valid_configs) if n > len(valid_configs) else len(probabilities) if n > len(probabilities) else n
    probability_of_best_n_configs = 0
    for i in range(current_n):
        probability_of_best_n_configs += probabilities[valid_configs[i]]

    return probability_of_best_n_configs


def get_n_most_probable_configs(row: pd.Series, n: int, puso: bool = True) -> list[str]:
    """
        Returns the n most probable configurations for the given row of the dataframe

        :param row: The row of the dataframe to get the n most probable configurations for
        :param n: The number of configurations to get
                  if n is greater than the number of configurations, all configurations are returned
        :param puso: Whether to get the n most probable configurations for the puso or quso hamiltonian
    """
    probabilities = get_probability_dict(row, puso)
    n_most_probable_configs = []

    if n > len(probabilities):
        n = len(probabilities)

    for i in range(n):
        n_most_probable_configs.append(max(probabilities, key=probabilities.get))
        probabilities.pop(max(probabilities, key=probabilities.get))

    return n_most_probable_configs


min_feature_cost = 10
max_feature_cost = 100
alpha_sat = None
layers = 40
strategy = 'avg'

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--start", help="start instance", type=int)
parser.add_argument("-e", "--end", help="end instance", type=int)
parser.add_argument("-q", "--skip_quso", help="skip quso", action='store_true')
parser.add_argument("-p", "--skip_puso", help="skip puso", action='store_true')
parser.add_argument("-f", "--file", help="file to read results from", type=str)

args = parser.parse_args()

if not args.file:
    # create results folder if it doesn't exist
    Path("benchmarks\\qaoa-feature-models\\results").mkdir(parents=True, exist_ok=True)

    # get problem instances from dimacs files
    np.random.seed(42)
    instances = [get_problem_instance_from_dimacs(f"benchmarks\\qaoa-feature-models\\feature_model_{i}.dimacs",
                                                  min_feature_cost, max_feature_cost, alpha_sat) for i in
                 range(args.start, args.end + 1)]

    # run the algorithm for the instances
    run_instances_and_save_results(instances, layers, strategy, skip_quso=args.skip_quso, skip_puso=args.skip_puso)

else:
    df = pd.read_csv(args.file)
    df = df.drop(columns=['Unnamed: 0'])
