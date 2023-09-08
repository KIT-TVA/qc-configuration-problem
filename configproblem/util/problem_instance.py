import numpy as np
from qubovert import boolean_var, PCBO
from qubovert.utils import DictArithmetic

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
    elif generation_type == 'growing_num_clauses':
        return generate_problem_instance_set_growing_num_clauses(n_instances, n_features, min_n_clauses, max_n_clauses,
                                                                 min_clause_length, max_clause_length,
                                                                 min_feature_cost, max_feature_cost, alpha_sat)
    elif generation_type == 'growing_num_literals':
        return generate_problem_instance_set_growing_num_literals(n_instances, n_features, min_n_clauses, max_n_clauses,
                                                                  min_clause_length, max_clause_length,
                                                                  min_feature_cost, max_feature_cost, alpha_sat)
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


def generate_problem_instance_set_growing_num_clauses(n_instances: int, n_features: int, min_n_clauses: int,
                                                      max_n_clauses: int, min_clause_length: int,
                                                      max_clause_length: int, min_feature_cost: int,
                                                      max_feature_cost: int, alpha_sat: float) \
        -> list['ProblemInstance']:
    """
        Generates a set of problem instances with a growing number of clauses

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
        for _ in range(n_instances_per_n_clauses):
            problem_instances.append(generate_problem_instance(n_features, n_clauses, n_clauses, min_clause_length,
                                                               max_clause_length, min_feature_cost, max_feature_cost,
                                                               alpha_sat))
    return problem_instances


def generate_problem_instance_set_growing_num_literals(n_instances: int, n_features: int, min_n_clauses: int,
                                                       max_n_clauses: int, min_clause_length: int,
                                                       max_clause_length: int, min_feature_cost: int,
                                                       max_feature_cost: int, alpha_sat: float) \
        -> list['ProblemInstance']:
    """
        Generates a set of problem instances with a growing number of literals per clause

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
    n_instances_per_n_literals = int(n_instances / (max_clause_length - min_clause_length + 1))
    problem_instances = []
    for n_literals in range(min_clause_length, max_clause_length + 1):
        for _ in range(n_instances_per_n_literals):
            problem_instances.append(generate_problem_instance(n_features, min_n_clauses, max_n_clauses, n_literals,
                                                               n_literals, min_feature_cost, max_feature_cost,
                                                               alpha_sat))
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
        problem_instances.extend(generate_problem_instance_set_growing_num_literals(n_instances_per_n_clauses,
                                                                                    n_features, n_clauses, n_clauses,
                                                                                    min_clause_length,
                                                                                    max_clause_length, min_feature_cost,
                                                                                    max_feature_cost, alpha_sat))
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
    for _ in range(n_start_instances - 1):
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
                variables.append(boolean_var(f"x{i}"))
                feature_cost.append(np.random.randint(min_feature_cost, max_feature_cost + 1))

        # sort variables and feature_cost
        variables = [x for _, x in sorted(zip(variables_str, variables))]
        feature_cost = [x for _, x in sorted(zip(variables_str, feature_cost))]

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
    return clause


class ProblemInstance:
    """ Represents a SAT instance with a cost associated with each feature"""
    sat_instance: list[list[tuple[boolean_var, bool]]]
    boolean_variables: list[boolean_var]
    feature_cost: list[int]
    alpha_sat: float
    valid_configs: list[str] = None
    best_config: str = None
    sat_model: PCBO = None
    cost_model: PCBO = None

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

        for variable in variables:
            if variable not in boolean_variables:
                raise ValueError(f"variable {variable} is used in sat_instance but not in boolean_variables")

        for variable in boolean_variables:
            if variable not in variables:
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
        if self.valid_configs is not None:
            return self.valid_configs

        valid_configs = []
        for i in range(2 ** len(self.boolean_variables)):
            config = np.binary_repr(i, width=len(self.boolean_variables))
            if self.__check_config_validity(config):
                valid_configs.append(config)
        self.valid_configs = valid_configs
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
        if self.best_config is not None:
            return self.best_config

        best_config = None
        best_cost = float('inf')
        for config in self.get_valid_configs():
            cost = self.__get_config_cost(config)
            if cost < best_cost:
                best_cost = cost
                best_config = config
        self.best_config = best_config
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
        if self.sat_model is not None:
            return self.sat_model.to_puso()
        self.sat_model = convert_to_penalty(self.sat_instance)
        return self.sat_model.to_puso()

    def get_cost_hamiltonian(self) -> DictArithmetic:
        if self.cost_model is not None:
            return self.cost_model.to_puso()
        cost_model = PCBO()
        for i in range(len(self.boolean_variables)):
            cost_model += self.feature_cost[i] * self.boolean_variables[i]
        self.cost_model = cost_model
        return self.cost_model.to_puso()

    def get_puso_combined_hamiltonian(self) -> DictArithmetic:
        return self.alpha_sat * self.get_puso_sat_hamiltonian() + self.get_cost_hamiltonian()

    def get_quso_sat_hamiltonian(self) -> DictArithmetic:
        if self.sat_model is not None:
            return self.sat_model.to_quso()
        self.sat_model = convert_to_penalty(self.sat_instance)
        return self.sat_model.to_quso()

    def get_quso_combined_hamiltonian(self) -> DictArithmetic:
        return self.alpha_sat * self.get_quso_sat_hamiltonian() + self.get_cost_hamiltonian()
