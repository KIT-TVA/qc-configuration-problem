import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
from qiskit import transpile
from qiskit.circuit import Parameter
from qiskit_aer import StatevectorSimulator

from configproblem.grover_sat import create_ksat_grover, calc_statevector_from
from configproblem.qaoa.qaoa_application import apply_qaoa_statevector
from configproblem.util.hamiltonian_math import get_hamiltonian_dimension
from configproblem.util.problem_instance import ProblemInstance, get_problem_instance_from_dimacs
from configproblem.qaoa.qaoa_mincost_k_sat import problem_circuit as puso_problem_circuit, convert_ancilla_bit_results
from configproblem.qaoa.qaoa_mincost_sat import problem_circuit as quso_problem_circuit
import configproblem.qaoa.qaoa_mixer as mixer
import configproblem.qaoa.qaoa_parameter_optimization as parameter_optimization

min_feature_cost = 10
max_feature_cost = 100
alpha_sat = None

mixer_circuit = mixer.warmstart_mixer
parameter_optimization = parameter_optimization.get_optimizer('COBYLA', maxiter=1000, tol=1e-12)
layers = 5
theta = {"beta": 0.01, "gamma": -0.01}  # start values for optimization
strategy = 'avg'
use_warmstart = True
use_optimizer = True
print_res = False


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
def run_puso_qaoa(instance: ProblemInstance, layers: int, strategy: str, skip=False,
                  warmstart_statevector: np.ndarray = None) -> dict:
    """
        Runs the QAOA algorithm for the given instance using its puso hamiltonian and returns the results

        :param instance: The instance to run the algorithm for
        :param layers: The number of layers to use for the algorithm
        :param strategy: The strategy to use for the algorithm
        :param skip: Whether to skip running the algorithm for the puso hamiltonian
        :param warmstart_statevector: The warmstart statevector to use
    """
    hamiltonian = instance.get_puso_combined_hamiltonian()

    if skip:
        probabilities_dict = {}
    else:
        probabilities, _ = apply_qaoa_statevector(puso_problem_circuit, mixer_circuit, parameter_optimization,
                                                  hamiltonian, layers, get_hamiltonian_dimension(hamiltonian), theta,
                                                  warmstart_statevector, strategy=strategy, use_optimizer=use_optimizer,
                                                  print_res=print_res)
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
def run_quso_qaoa(instance: ProblemInstance, layers: int, strategy: str, skip=False,
                  warmstart_statevector: np.ndarray = None) -> dict:
    """
        Runs the QAOA algorithm for the given instance using its quso hamiltonian and returns the results

        :param instance: The instance to run the algorithm for
        :param layers: The number of layers to use for the algorithm
        :param strategy: The strategy to use for the algorithm
        :param skip: Whether to skip running the algorithm for the quso hamiltonian
        :param warmstart_statevector: The warmstart statevector to use
    """
    hamiltonian = instance.get_quso_combined_hamiltonian()

    if skip:
        probabilities_dict = {}
    else:
        probabilities, _ = apply_qaoa_statevector(quso_problem_circuit, mixer_circuit, parameter_optimization,
                                                  hamiltonian, layers, get_hamiltonian_dimension(hamiltonian), theta,
                                                  warmstart_statevector, strategy=strategy, use_optimizer=use_optimizer,
                                                  print_res=print_res)
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
                 skip_puso: bool = False, warmstart_statevector: np.ndarray = None) -> dict:
    """
        Runs the QAOA algorithm for the given instance and returns the results

        :param instance: The instance to run the algorithm for
        :param layers: The number of layers to use for the algorithm
        :param strategy: The strategy to use for the algorithm
        :param skip_quso: Whether to skip running the algorithm for the quso hamiltonian
        :param skip_puso: Whether to skip running the algorithm for the puso hamiltonian
        :param warmstart_statevector: The warmstart statevector to use
    """
    print("Running PUSO QAOA")
    puso_results = run_puso_qaoa(instance, layers, strategy, skip=skip_puso,
                                 warmstart_statevector=warmstart_statevector)
    print("Running QUSO QAOA")
    quso_results = run_quso_qaoa(instance, layers, strategy, skip=skip_quso,
                                 warmstart_statevector=warmstart_statevector)

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


def get_warmstart_statevector_from_grover(instance: ProblemInstance) -> np.ndarray:
    """
        Runs the grover algorithm on the SAT instance of the given problem instance
        and returns the warmstart statevector

        :param instance: The instance to get the warmstart statevector for
    """
    sat_problem = instance.get_sat_instance()
    n_features = instance.get_num_features()

    # convert sat_problem to use integers for variables instead of boolean_var
    variables = instance.get_boolean_variables()
    converted_sat_problem = []
    for clause in sat_problem:
        converted_clause = []
        for literal in clause:
            converted_clause.append((variables.index(literal[0]), literal[1]))
        converted_sat_problem.append(converted_clause)

    # get the warmstart statevector using the grover algorithm
    main_qc, qc_oracle = create_ksat_grover(converted_sat_problem, 1)

    transpiled_grover_circuit = transpile(main_qc, StatevectorSimulator())
    results = StatevectorSimulator().run(transpiled_grover_circuit, shots=1000).result()
    counts = results.get_counts()
    return calc_statevector_from(counts, n_features)


def run_instances_and_save_results(instances: list[ProblemInstance], layers: int, strategy: str, results_path: str,
                                   skip_quso: bool = False, skip_puso: bool = False,
                                   save_individual_results: bool = False) -> None:
    """
        Runs the QAOA algorithm for the given instances and returns the results as a dataframe

        :param instances: The instances to run the algorithm for
        :param layers: The number of layers to use for the algorithm
        :param strategy: The strategy to use for the algorithm
        :param results_path: The path to save the results to
        :param skip_quso: Whether to skip running the algorithm for the quso hamiltonian
        :param skip_puso: Whether to skip running the algorithm for the puso hamiltonian
        :param save_individual_results: Whether to save the results for each instance individually
    """
    results = pd.DataFrame()
    for index, instance in enumerate(instances):
        print(f"Running instance {index}")
        warmstart_statevector = None
        if use_warmstart:
            warmstart_statevector = get_warmstart_statevector_from_grover(instance)
            skip_quso = True  # warmstart does not account for additional ancilla qubits in quso formulation

        df = pd.DataFrame([run_instance(instance, layers, strategy, skip_quso=skip_quso, skip_puso=skip_puso,
                                        warmstart_statevector=warmstart_statevector)])

        if save_individual_results:
            df.to_csv(results_path + f"feature_model_{index}.csv")
        results = pd.concat([results, df], ignore_index=True)
    results.to_csv(results_path + "all_results.csv")
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
        probabilities_str = re.sub("[{} ']", "", row["probabilities_puso"]).split(",")
    else:
        probabilities_str = re.sub("[{} ']", "", row["probabilities_quso"]).split(",")

    probability_dict = {}
    for config_probability_pair in probabilities_str:
        config_probability_pair = config_probability_pair.split(":")
        if len(config_probability_pair) != 2:
            continue
        config = config_probability_pair[0]
        probability = float(config_probability_pair[1])
        probability_dict[config] = probability
    return probability_dict


def get_validity_quality(row: pd.Series, puso: bool = True) -> float:
    """
        Returns the validity quality for the given row of the dataframe

        :param row: The row of the dataframe to get the result quality for
        :param puso: Whether to get the result quality for the puso or quso hamiltonian
    """
    probabilities = get_probability_dict(row, puso)
    if len(probabilities) == 0:
        return float('nan')
    valid_configs = get_valid_configs_sorted_by_cost(row)

    validity_quality = 0
    for config in valid_configs:
        validity_quality += probabilities[config] * 2 ** int(row["n_features"]) / len(valid_configs)
    return validity_quality


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
    if len(probabilities) == 0:
        return float('nan')
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


def rank_biased_overlap(first_rank: list, second_rank: list, p: float) -> float:
    """
        Returns the rank biased overlap for the given ranks as described in Eq. 23 of the following paper:
        Webber, W., Moffat, A. and Zobel, J., 2010. A similarity measure for
        indefinite rankings. ACM Transactions on Information Systems (TOIS),
        28(4), pp.1-38.
        http://www.williamwebber.com/research/papers/wmz10_tois.pdf

        :param p: The p value of the rank biased overlap
        :param first_rank: The first rank
        :param second_rank: The second rank
    """
    assert 0 <= p <= 1
    assert len(first_rank) == len(set(first_rank))
    assert len(second_rank) == len(set(second_rank))

    if not first_rank and not second_rank:
        return 1  # both lists are empty

    k = max(len(first_rank), len(second_rank))
    x_k = len(set(first_rank).intersection(set(second_rank)))
    summation = 0
    for d in range(1, k + 1):
        x_d = len(set(first_rank[:d]).intersection(set(second_rank[:d])))
        summation += x_d / d * p ** d

    return (x_k / k) * (p ** k) + (1 - p) / p * summation


def process_results(df: pd.DataFrame) -> pd.DataFrame:
    """
        Processes the results of the given dataframe and saves the processed results in a new dataframe

        :param df: The dataframe to process the results for
    """
    processed_dataframe = pd.DataFrame()

    for index, row in df.iterrows():
        avg_valid_probability_puso = (get_probability_of_best_n_configs(row, len(get_valid_configs_sorted_by_cost(row)),
                                                                        puso=True) /
                                      len(get_valid_configs_sorted_by_cost(row)))
        avg_valid_probability_quso = (get_probability_of_best_n_configs(row, len(get_valid_configs_sorted_by_cost(row)),
                                                                        puso=False) /
                                      len(get_valid_configs_sorted_by_cost(row)))

        most_probable_configs_list_puso = get_n_most_probable_configs(row, len(get_valid_configs_sorted_by_cost(row)),
                                                                      puso=True)
        most_probable_configs_list_quso = get_n_most_probable_configs(row, len(get_valid_configs_sorted_by_cost(row)),
                                                                      puso=False)

        processed_dataframe = pd.concat([processed_dataframe, pd.DataFrame(
            {'n_features': row['n_features'],
             'n_clauses': row['n_clauses'],
             'min_literals_per_clause': row['min_literals_per_clause'],
             'max_literals_per_clause': row['max_literals_per_clause'],
             'n_valid_configs': len(get_valid_configs_sorted_by_cost(row)),
             'validity_quality_puso': get_validity_quality(row, puso=True),
             'validity_quality_quso': get_validity_quality(row, puso=False),
             'cost_quality_top_1_puso': get_probability_of_best_n_configs(row, 1, puso=True) / 1 /
                                        avg_valid_probability_puso,
             'cost_quality_top_1_quso': get_probability_of_best_n_configs(row, 1, puso=False) / 1 /
                                        avg_valid_probability_quso,
             'cost_quality_top_5_puso': get_probability_of_best_n_configs(row, 5, puso=True) / 5 /
                                        avg_valid_probability_puso,
             'cost_quality_top_5_quso': get_probability_of_best_n_configs(row, 5, puso=False) / 5 /
                                        avg_valid_probability_quso,
             'rbo_puso': rank_biased_overlap(most_probable_configs_list_puso,
                                             get_valid_configs_sorted_by_cost(row), 0.9),
             'rbo_quso': rank_biased_overlap(most_probable_configs_list_quso,
                                             get_valid_configs_sorted_by_cost(row), 0.9),
             'execution_time_puso': row['execution_time_puso'],
             'execution_time_quso': row['execution_time_quso'],
             'problem_circuit_depth_puso': row['problem_circuit_depth_puso'],
             'problem_circuit_depth_quso': row['problem_circuit_depth_quso'],
             'circuit_width_puso': row['circuit_width_puso'],
             'circuit_width_quso': row['circuit_width_quso']}, index=[0])],
                                        ignore_index=True)

    return processed_dataframe


parser = argparse.ArgumentParser()
parser.add_argument("--first", help="start instance", type=int)
parser.add_argument("--last", help="end instance", type=int)
parser.add_argument("--skip-quso", help="skip quso", action='store_true')
parser.add_argument("--skip-puso", help="skip puso", action='store_true')
parser.add_argument("--save-individual-results", help="save individual results", action='store_true')
parser.add_argument("-f", "--file", help="file to read results from", type=str)
parser.add_argument("--linux", help="Use linux file paths", action='store_true')

args = parser.parse_args()

if args.linux:
    base_path = "benchmarks/qaoa-feature-models/"
    results_path = "results/"
else:
    base_path = "benchmarks\\qaoa-feature-models\\"
    results_path = "results\\"

if not args.file:
    # create results folder if it doesn't exist
    Path(base_path + results_path).mkdir(parents=True, exist_ok=True)

    # get problem instances from dimacs files
    np.random.seed(42)
    instances = [get_problem_instance_from_dimacs(base_path + f"feature_model_{i}.dimacs", min_feature_cost,
                                                  max_feature_cost, alpha_sat) for i in
                 range(args.first, args.last + 1)]

    # run the algorithm for the instances
    run_instances_and_save_results(instances, layers, strategy, skip_quso=args.skip_quso, skip_puso=args.skip_puso,
                                   save_individual_results=args.save_individual_results,
                                   results_path=base_path+results_path)

else:
    df = pd.read_csv(args.file)
    df = df.drop(columns=['Unnamed: 0'])
    processed_df = process_results(df)
    processed_df.to_csv(base_path + results_path + "processed_results.csv")
