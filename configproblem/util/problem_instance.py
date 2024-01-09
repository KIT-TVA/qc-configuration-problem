import numpy as np
from qubovert import boolean_var, PCBO
from qubovert.utils import DictArithmetic

from configproblem.util.dimacs_reader import DimacsReader
from configproblem.util.model_transformation import convert_to_penalty


def generate_problem_instance_set(n_instances: int, n_features: int, min_n_clauses: int, max_n_clauses: int,
                                  min_clause_length: int, max_clause_length: int, min_feature_cost: int,
                                  max_feature_cost: int, alpha_sat: float, seed: int = 42,
                                  generation_type: str = 'random', start_instance: 'ProblemInstance' = None) \
        -> list['ProblemInstance']:
    """
        Generates a set of problem instances

        :param n_instances: number of problem instances to generate
        :param n_features: number of features
        :param min_n_clauses: minimum number of clauses
        :param max_n_clauses: maximum number of clauses
        :param min_clause_length: minimum number of variables in a clause
        :param max_clause_length: maximum number of variables in a clause
        :param min_feature_cost: minimum cost of a feature
        :param max_feature_cost: maximum cost of a feature
        :param alpha_sat: weight of the SAT part of the objective function,
                          will be considered as the maximum weight if generation_type is 'alpha_sat'
        :param seed: seed for the random number generator
        :param generation_type: type of generation, can be 'random', 'growing_num_clauses', 'growing_num_literals',
                                'growing_num_clauses_and_literals', 'append_clauses' or 'alpha_sat'
        :param start_instance: problem instance to use as a starting point for 'alpha_sat' generation type
    """
    np.random.seed(seed)
    if generation_type == 'random':
        return generate_problem_instance_set_random(n_instances, n_features, min_n_clauses, max_n_clauses,
                                                    min_clause_length, max_clause_length, min_feature_cost,
                                                    max_feature_cost, alpha_sat)
    elif generation_type == 'growing_num_clauses' or generation_type == 'growing_num_literals':
        return generate_problem_instance_set_growing_num_clauses_or_literals(n_instances, n_features, min_n_clauses,
                                                                             max_n_clauses, min_clause_length,
                                                                             max_clause_length, min_feature_cost,
                                                                             max_feature_cost, alpha_sat,
                                                                             generation_type=generation_type)
    elif generation_type == 'growing_num_clauses_and_literals':
        return generate_problem_instance_set_growing_num_clauses_and_literals(n_instances, n_features, min_n_clauses,
                                                                              max_n_clauses, min_clause_length,
                                                                              max_clause_length, min_feature_cost,
                                                                              max_feature_cost, alpha_sat)
    elif generation_type == 'append_clauses':
        return generate_problem_instance_set_append_clauses(n_instances, n_features, min_n_clauses, max_n_clauses,
                                                            min_clause_length, max_clause_length, min_feature_cost,
                                                            max_feature_cost, alpha_sat)
    elif generation_type == 'alpha_sat':
        return generate_problem_instance_set_alpha_sat(n_instances, n_features, min_n_clauses, max_n_clauses,
                                                       min_clause_length, max_clause_length, min_feature_cost,
                                                       max_feature_cost, alpha_sat, start_instance=start_instance)
    else:
        raise ValueError(f"generation_type {generation_type} is not supported")


def generate_problem_instance_set_random(n_instances: int, n_features: int, min_n_clauses: int, max_n_clauses: int,
                                         min_clause_length: int, max_clause_length: int, min_feature_cost: int,
                                         max_feature_cost: int, alpha_sat: float) -> list['ProblemInstance']:
    """
        Generates a set of random problem instances

        :param n_instances: number of problem instances to generate
        :param n_features: number of features
        :param min_n_clauses: minimum number of clauses
        :param max_n_clauses: maximum number of clauses
        :param min_clause_length: minimum number of variables in a clause
        :param max_clause_length: maximum number of variables in a clause
        :param min_feature_cost: minimum cost of a feature
        :param max_feature_cost: maximum cost of a feature
        :param alpha_sat: weight of the SAT part of the objective function
    """
    problem_instances = []
    for _ in range(n_instances):
        problem_instances.append(generate_problem_instance(n_features, min_n_clauses, max_n_clauses, min_clause_length,
                                                           max_clause_length, min_feature_cost, max_feature_cost,
                                                           alpha_sat))
    return problem_instances


def generate_problem_instance_set_growing_num_clauses_or_literals(n_instances: int, n_features: int,
                                                                  min_n_clauses: int, max_n_clauses: int,
                                                                  min_clause_length: int, max_clause_length: int,
                                                                  min_feature_cost: int, max_feature_cost: int,
                                                                  alpha_sat: float, generation_type: str) \
        -> list['ProblemInstance']:
    """
        Generates a set of problem instances with a growing number of clauses or a growing number of literals per clause
        for each number of clauses

        :param n_instances: number of problem instances to generate
        :param n_features: number of features
        :param min_n_clauses: minimum number of clauses
        :param max_n_clauses: maximum number of clauses
        :param min_clause_length: minimum number of variables in a clause
        :param max_clause_length: maximum number of variables in a clause
        :param min_feature_cost: minimum cost of a feature
        :param max_feature_cost: maximum cost of a feature
        :param alpha_sat: weight of the SAT part of the objective function
        :param generation_type: type of generation, can be 'growing_num_clauses' or 'growing_num_literals'
    """
    if generation_type == 'growing_num_clauses':
        n_instances_per_n = int(n_instances / (max_n_clauses - min_n_clauses + 1))
        n = range(min_n_clauses, max_n_clauses + 1)
    else:
        n_instances_per_n = int(n_instances / (max_clause_length - min_clause_length + 1))
        n = range(min_clause_length, max_clause_length + 1)

    problem_instances = []
    for n_clauses_or_literals in n:
        for _ in range(n_instances_per_n):
            if generation_type == 'growing_num_clauses':
                problem_instances.append(generate_problem_instance(n_features, n_clauses_or_literals,
                                                                   n_clauses_or_literals, min_clause_length,
                                                                   max_clause_length, min_feature_cost,
                                                                   max_feature_cost, alpha_sat))
            else:
                problem_instances.append(generate_problem_instance(n_features, min_n_clauses, max_n_clauses,
                                                                   n_clauses_or_literals, n_clauses_or_literals,
                                                                   min_feature_cost, max_feature_cost, alpha_sat))
    return problem_instances


def generate_problem_instance_set_growing_num_clauses_and_literals(n_instances: int, n_features: int,
                                                                   min_n_clauses: int, max_n_clauses: int,
                                                                   min_clause_length: int, max_clause_length: int,
                                                                   min_feature_cost: int, max_feature_cost: int,
                                                                   alpha_sat: float) -> list['ProblemInstance']:
    """
        Generates a set of problem instances with a growing number of clauses and growing number of literals per clause
        for each number of clauses

        :param n_instances: number of problem instances to generate
        :param n_features: number of features
        :param min_n_clauses: minimum number of clauses
        :param max_n_clauses: maximum number of clauses
        :param min_clause_length: minimum number of variables in a clause
        :param max_clause_length: maximum number of variables in a clause
        :param min_feature_cost: minimum cost of a feature
        :param max_feature_cost: maximum cost of a feature
        :param alpha_sat: weight of the SAT part of the objective function
    """
    n_instances_per_n_clauses = int(n_instances / (max_n_clauses - min_n_clauses + 1))
    problem_instances = []
    for n_clauses in range(min_n_clauses, max_n_clauses + 1):
        problem_instances.extend(
            generate_problem_instance_set_growing_num_clauses_or_literals(n_instances_per_n_clauses, n_features,
                                                                          n_clauses, n_clauses, min_clause_length,
                                                                          max_clause_length, min_feature_cost,
                                                                          max_feature_cost, alpha_sat,
                                                                          generation_type='growing_num_literals')
        )
    return problem_instances


def generate_problem_instance_set_append_clauses(n_instances: int, n_features: int, min_n_clauses: int,
                                                 max_n_clauses: int, min_clause_length: int, max_clause_length: int,
                                                 min_feature_cost: int, max_feature_cost: int, alpha_sat: float) \
        -> list['ProblemInstance']:
    """
        Generates a set of problem instances by adding clauses to an existing problem instance

        :param n_instances: number of problem instances to generate
        :param n_features: number of features
        :param min_n_clauses: minimum number of clauses
        :param max_n_clauses: maximum number of clauses
        :param min_clause_length: minimum number of variables in a clause
        :param max_clause_length: maximum number of variables in a clause
        :param min_feature_cost: minimum cost of a feature
        :param max_feature_cost: maximum cost of a feature
        :param alpha_sat: weight of the SAT part of the objective function
    """
    n_start_instances = int(n_instances / (max_n_clauses - min_n_clauses + 1))
    problem_instances = []
    for _ in range(n_start_instances):
        problem_instances.append(generate_problem_instance(n_features, min_n_clauses, min_n_clauses, min_clause_length,
                                                           max_clause_length, min_feature_cost, max_feature_cost,
                                                           alpha_sat))
        for _ in range(min_n_clauses + 1, max_n_clauses + 1):
            problem_instances.append(get_problem_instance_with_added_clause(problem_instances[-1], n_features,
                                                                            min_clause_length, max_clause_length,
                                                                            min_feature_cost, max_feature_cost))
    return problem_instances


def generate_problem_instance_set_alpha_sat(n_instances: int, n_features: int, min_n_clauses: int,
                                            max_n_clauses: int, min_clause_length: int, max_clause_length: int,
                                            min_feature_cost: int, max_feature_cost: int, alpha_sat: float,
                                            start_instance: 'ProblemInstance' = None) -> list['ProblemInstance']:
    """
        Generates a set of problem instances with a growing alpha_sat

        :param n_instances: number of problem instances to generate
        :param n_features: number of features
        :param min_n_clauses: minimum number of clauses
        :param max_n_clauses: maximum number of clauses
        :param min_clause_length: minimum number of variables in a clause
        :param max_clause_length: maximum number of variables in a clause
        :param min_feature_cost: minimum cost of a feature
        :param max_feature_cost: maximum cost of a feature
        :param alpha_sat: maximum weight of the SAT part of the objective function
        :param start_instance: problem instance to use as a starting point
    """
    alpha_sat_step = alpha_sat / n_instances
    if start_instance is not None:
        problem_instances = [ProblemInstance(start_instance.get_sat_instance(), start_instance.get_boolean_variables(),
                                             start_instance.get_feature_cost(), alpha_sat_step)]
    else:
        problem_instances = [generate_problem_instance(n_features, min_n_clauses, max_n_clauses, min_clause_length,
                                                       max_clause_length, min_feature_cost, max_feature_cost,
                                                       alpha_sat_step)]
    for _ in range(n_instances - 1):
        problem_instances.append(ProblemInstance(problem_instances[-1].get_sat_instance(),
                                                 problem_instances[-1].get_boolean_variables(),
                                                 problem_instances[-1].get_feature_cost(),
                                                 problem_instances[-1].alpha_sat + alpha_sat_step))
    return problem_instances


def generate_problem_instance(n_features: int, min_n_clauses: int, max_n_clauses: int, min_clause_length: int,
                              max_clause_length: int, min_feature_cost: int, max_feature_cost: int, alpha_sat: float) \
        -> 'ProblemInstance':
    """
        Generates a random problem instance

        :param n_features: number of features
        :param min_n_clauses: minimum number of clauses
        :param max_n_clauses: maximum number of clauses
        :param min_clause_length: minimum number of variables in a clause
        :param max_clause_length: maximum number of variables in a clause
        :param min_feature_cost: minimum cost of a feature
        :param max_feature_cost: maximum cost of a feature
        :param alpha_sat: weight of the SAT part of the objective function
    """
    if min_n_clauses > max_n_clauses:
        raise ValueError("min_n_clauses must be smaller than or equal to max_n_clauses")
    if min_clause_length > max_clause_length:
        raise ValueError("min_clause_length must be smaller than or equal to max_clause_length")
    if min_feature_cost > max_feature_cost:
        raise ValueError("min_feature_cost must be smaller than or equal to max_feature_cost")
    if max_clause_length > n_features:
        raise ValueError("max_clause_length must be smaller than or equal to n_features")

    variables = [boolean_var(f"x{i}") for i in range(n_features)]
    feature_cost = list(np.random.randint(min_feature_cost, max_feature_cost + 1, size=n_features))
    sat_instance = generate_sat_instance(variables, min_n_clauses, max_n_clauses, min_clause_length, max_clause_length)

    return ProblemInstance(sat_instance, variables, feature_cost, alpha_sat)


def get_problem_instance_with_added_clause(problem_instance: 'ProblemInstance', n_features: int, min_clause_length: int,
                                           max_clause_length: int, min_feature_cost: int, max_feature_cost: int) \
        -> 'ProblemInstance':
    """
        Generates a new problem instance by adding a clause to an existing problem instance

        :param problem_instance: problem instance to which a clause will be added
        :param n_features: number of features
        :param min_clause_length: minimum number of variables in a clause
        :param max_clause_length: maximum number of variables in a clause
        :param min_feature_cost: minimum cost of a feature
        :param max_feature_cost: maximum cost of a feature
    """
    sat_instance = problem_instance.get_sat_instance().copy()
    variables = problem_instance.boolean_variables.copy()
    feature_cost = problem_instance.feature_cost.copy()

    if n_features > len(variables):
        variables_str = [list(var.keys())[0][0] for var in variables]
        for i in range(n_features):
            if variables_str.__contains__(f"x{i}"):
                continue
            else:
                variable = boolean_var(f"x{i}")
                variables.append(variable)
                feature_cost.append(np.random.randint(min_feature_cost, max_feature_cost + 1))

        # sort variables and feature_cost by variable name
        variables, feature_cost = map(list, zip(*sorted(zip(variables, feature_cost),
                                                        key=lambda x: list(x[0].keys())[0][0])))

    sat_instance.append(generate_sat_clause(variables, min_clause_length, max_clause_length))
    return ProblemInstance(sat_instance, variables, feature_cost, problem_instance.alpha_sat)


def generate_sat_instance(variables: list[boolean_var], min_n_clauses: int, max_n_clauses: int, min_clause_length: int,
                          max_clause_length: int) -> list[list[tuple[boolean_var, bool]]]:
    """
        Generates a SAT instance with n_clauses clauses, each clause is a list of tuples (variable, is_not_negated)

        :param variables: list of boolean variables that can be used in the SAT instance
        :param min_n_clauses: minimum number of clauses
        :param max_n_clauses: maximum number of clauses
        :param min_clause_length: minimum number of variables in a clause
        :param max_clause_length: maximum number of variables in a clause
    """
    n_clauses = np.random.randint(min_n_clauses, max_n_clauses + 1)
    sat_instance = []
    for _ in range(n_clauses):
        sat_instance.append(generate_sat_clause(variables, min_clause_length, max_clause_length))

    return sat_instance


def generate_sat_clause(variables: list[boolean_var], min_clause_length: int, max_clause_length: int) \
        -> list[tuple[boolean_var, bool]]:
    """
        Generates a random SAT clause with length between min_clause_length and max_clause_length

        :param variables: list of boolean variables that can be used in the SAT clause
        :param min_clause_length: minimum number of variables in a clause
        :param max_clause_length: maximum number of variables in a clause
    """
    clause_length = np.random.randint(min_clause_length, max_clause_length + 1)
    clause = []
    available_variables = variables.copy()
    for _ in range(clause_length):
        variable = np.random.choice(available_variables)
        is_not_negated = np.random.choice([True, False])
        clause.append((variable, is_not_negated))
        available_variables.remove(variable)
    # sort clause by variable name
    clause.sort(key=lambda x: list(x[0].keys())[0][0])

    return clause


def get_sat_instance_from_dimacs(dimacs_file_path: str) \
        -> tuple[list[list[tuple[boolean_var, bool]]], list[boolean_var]]:
    """
        Returns a SAT instance from a DIMACS file

        :param dimacs_file_path: path to the DIMACS file
    """
    sat_instance = []
    dimacs_reader = DimacsReader()
    dimacs_reader.fromFile(dimacs_file_path)
    boolean_vars = [boolean_var(f"x{i}") for i in range(len(dimacs_reader.getFeatures()))]
    for clause in dimacs_reader.clauses:
        sat_clause = []
        for literal in clause:
            if literal == 0:
                break
            variable = boolean_vars[abs(literal) - 1]
            is_not_negated = literal >= 0
            sat_clause.append((variable, is_not_negated))
        sat_instance.append(sat_clause)
    return sat_instance, boolean_vars


def get_problem_instance_from_dimacs(dimacs_file_path: str, min_feature_cost: int, max_feature_cost: int,
                                     alpha_sat: float) -> 'ProblemInstance':
    """
        Returns a problem instance using a SAT instance from a DIMACS file

        :param dimacs_file_path: path to the DIMACS file
        :param min_feature_cost: minimum cost of a feature
        :param max_feature_cost: maximum cost of a feature
        :param alpha_sat: weight of the SAT part of the objective function
    """
    sat_instance, boolean_vars = get_sat_instance_from_dimacs(dimacs_file_path)
    feature_cost = list(np.random.randint(min_feature_cost, max_feature_cost + 1, size=len(boolean_vars)))
    return ProblemInstance(sat_instance, boolean_vars, feature_cost, alpha_sat)


class ProblemInstance:
    """ Represents a SAT instance with a cost associated with each feature"""
    sat_instance: list[list[tuple[boolean_var, bool]]]
    boolean_variables: list[boolean_var]
    feature_cost: list[int]
    alpha_sat: float

    def __init__(self, sat_instance: list[list[tuple[boolean_var, bool]]], boolean_variables: list[boolean_var],
                 feature_cost: list[int], alpha_sat: float = None):
        """
            :param sat_instance: list of clauses, each clause is a list of tuples (variable, is_not_negated)
            :param boolean_variables: ordered list of boolean variables
            :param feature_cost: list of costs for each feature
            :param alpha_sat: weight of SAT part in combined hamiltonian; if None a normalized value will be computed
                              based on feature_cost
        """
        variables = []
        for clause in sat_instance:
            for variable, _ in clause:
                if variable not in variables:
                    variables.append(variable)

        # Check if all variables in sat_instance are in boolean_variables
        for variable in variables:
            if variable not in boolean_variables:
                raise ValueError(f"variable {variable} is used in sat_instance but not in boolean_variables")

        # Identify variables in boolean_variables that are not used in sat_instance
        variables_to_remove = []
        for variable in boolean_variables:
            if variable not in variables:
                variables_to_remove.append(variable)

        # Remove variables from boolean_variables and feature_cost that are not used in sat_instance if their cost is 0
        for variable in variables_to_remove:
            variable_index = boolean_variables.index(variable)
            if feature_cost[variable_index] == 0:
                boolean_variables.remove(variable)
                del feature_cost[variable_index]

        if len(boolean_variables) != len(feature_cost):
            raise ValueError("boolean_variables and feature_cost must have the same length")

        self.sat_instance = sat_instance
        self.boolean_variables = boolean_variables
        self.feature_cost = feature_cost

        if alpha_sat is not None:
            self.alpha_sat = alpha_sat
        else:
            self.alpha_sat = self.__compute_normalized_alpha_sat()

    def __compute_normalized_alpha_sat(self) -> float:
        """
            Computes a normalized value for self.alpha_sat based on feature_cost
        """
        if sum(self.feature_cost) == 0:
            return 1
        else:
            return sum(self.feature_cost) * 1.5

    def __str__(self) -> str:
        return f"sat_instance: " + self.sat_instance_to_string() + "\n" \
                                                                   f"n_features: {len(self.boolean_variables)}\n" \
                                                                   f"boolean_variables: {self.boolean_variables}\n" \
                                                                   f"feature_cost: {self.feature_cost}\n" \
                                                                   f"alpha_sat: {self.alpha_sat}"

    def sat_instance_to_string(self) -> str:
        instance_parts = []
        for clause in self.sat_instance:
            clause_parts = []
            for var, negated in clause:
                literal = r"\neg {}".format(list(var.keys())[0][0]) if not negated else r"{}"\
                    .format(list(var.keys())[0][0])
                clause_parts.append(literal)
            clause = r" \vee ".join(clause_parts)
            instance_parts.append(r"({})".format(clause))
        instance_string = r" \wedge ".join(instance_parts)
        return instance_string

    def get_sat_instance(self) -> list[list[tuple[boolean_var, bool]]]:
        return self.sat_instance

    def get_num_features(self) -> int:
        return len(self.boolean_variables)

    def get_boolean_variables(self) -> list[boolean_var]:
        return self.boolean_variables

    def get_feature_cost(self) -> list[int]:
        return self.feature_cost

    def get_valid_configs(self) -> list[str]:
        valid_configs = []
        for i in range(2 ** len(self.boolean_variables)):
            config = np.binary_repr(i, width=len(self.boolean_variables))
            if self.__check_config_validity(config):
                valid_configs.append(config)
        return valid_configs

    def __check_config_validity(self, config) -> bool:
        """
            Checks if a given config is valid for this problem instance
        """
        for clause in self.sat_instance:
            clause_is_valid = False
            for variable, is_not_negated in clause:
                variable_index = self.boolean_variables.index(variable) + 1
                if (is_not_negated and config[-variable_index] == '1') or \
                        (not is_not_negated and config[-variable_index] == '0'):
                    clause_is_valid = True
                    break
            if not clause_is_valid:
                return False
        return True

    def get_best_config(self) -> str:
        best_config = None
        best_cost = float('inf')
        for config in self.get_valid_configs():
            cost = self.__get_config_cost(config)
            if cost < best_cost:
                best_cost = cost
                best_config = config
        return best_config

    def __get_config_cost(self, config: str) -> int:
        """
            Returns the cost of a given config
        """
        cost = 0
        for i in range(len(self.boolean_variables)):
            if config[-(i + 1)] == '1':
                cost += self.feature_cost[i]
        return cost

    def get_puso_sat_hamiltonian(self) -> DictArithmetic:
        return convert_to_penalty(self.sat_instance).to_puso()

    def __get_cost_pcbo(self) -> PCBO:
        cost_model = PCBO()
        for i in range(len(self.boolean_variables)):
            cost_model += self.feature_cost[i] * self.boolean_variables[i]
        return cost_model

    def get_cost_hamiltonian(self) -> DictArithmetic:
        return self.__get_cost_pcbo().to_puso()

    def __get_combined_pcbo(self) -> PCBO:
        return self.alpha_sat * convert_to_penalty(self.sat_instance) + self.__get_cost_pcbo()

    def get_puso_combined_hamiltonian(self) -> DictArithmetic:
        return self.__get_combined_pcbo().to_puso()

    def get_quso_sat_hamiltonian(self) -> DictArithmetic:
        return convert_to_penalty(self.sat_instance).to_quso()

    def get_quso_combined_hamiltonian(self) -> DictArithmetic:
        return self.__get_combined_pcbo().to_quso()

    def get_success_probability(self, probabilities_dict) -> float:
        """
            Returns the sum of the probabilities for every valid config of a given result,
            returns -1 if no valid configs exist

            :param probabilities_dict: dictionary with probabilities for each configuration
        """
        if len(self.get_valid_configs()) == 0:
            return -1
        success_probability = 0
        for config in self.get_valid_configs():
            success_probability += probabilities_dict[config] if config in probabilities_dict else 0
        return success_probability

    def get_validity_quality(self, probabilities_dict) -> float:
        """
            Returns the validity quality of a given result, returns -1 if no valid configs exist

            :param probabilities_dict: dictionary with probabilities for each configuration
        """
        if len(self.get_valid_configs()) == 0:
            return -1
        success_probability = self.get_success_probability(probabilities_dict)
        return success_probability * (2 ** self.get_num_features() / len(self.get_valid_configs()))

    def convert_solution_dict(self, solution_dict: dict, hamiltonian_type: str) -> dict:
        """
            Converts a solution dict from a QUBO solver to a solution dict for the original problem instance

            :param solution_dict: solution dict from a QUBO solver
            :param hamiltonian_type: type of hamiltonian used to generate the solution dict, can be 'puso_sat',
                                     'quso_sat', 'puso_combined', 'quso_combined' or 'cost'
        """
        if hamiltonian_type == 'puso_sat' or hamiltonian_type == 'quso_sat':
            hamiltonian_pcbo = convert_to_penalty(self.sat_instance)
        elif hamiltonian_type == 'puso_combined' or hamiltonian_type == 'quso_combined':
            hamiltonian_pcbo = self.__get_combined_pcbo()
        elif hamiltonian_type == 'cost':
            hamiltonian_pcbo = self.__get_cost_pcbo()
        else:
            raise ValueError(f"hamiltonian_type {hamiltonian_type} is not supported")

        result_dict = {}
        for key, value in solution_dict.items():
            # convert key from binary string to dict in the form of {var_index: -1/1}
            key_dict = {}
            for i in range(len(key)):
                key_dict[i] = 1 if key[-i - 1] == '0' else -1

            # convert key_dict from dict in the form of {var_index: -1/1} to dict in the form of {'var_name': -1/1}
            result_key_dict = hamiltonian_pcbo.convert_solution(key_dict, spin=True)

            # convert result_key_dict from dict in the form of {'var_name': -1/1} to binary string
            result_key = ''
            for variable in self.boolean_variables:
                if list(variable.keys())[0][0] not in result_key_dict:
                    result_key += '0'
                else:
                    result_key += '1' if result_key_dict[list(variable.keys())[0][0]] == 1 else '0'
            result_key = result_key[::-1]
            result_dict[result_key] = value
        return result_dict
