import numpy as np
from qubovert import boolean_var, PCBO
from qubovert.utils import DictArithmetic

from configproblem.util.model_transformation import convert_to_penalty


def generate_problem_instance(n_features: int, min_n_clauses: int, max_n_clauses: int, min_clause_length: int,
                              max_clause_length: int, min_feature_cost: int, max_feature_cost: int, alpha_sat: float,
                              seed: int = None) -> 'ProblemInstance':
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
        :param seed: seed for the random number generator
    """
    if seed is not None:
        np.random.seed(seed)

    variables = [boolean_var(f"x{i}") for i in range(n_features)]
    feature_cost = list(np.random.randint(min_feature_cost, max_feature_cost + 1, size=n_features))
    sat_instance = generate_sat_instance(variables, min_n_clauses, max_n_clauses, min_clause_length, max_clause_length,
                                         seed)

    return ProblemInstance(sat_instance, variables, feature_cost, alpha_sat)


def generate_sat_instance(variables: list[boolean_var], min_n_clauses: int, max_n_clauses: int, min_clause_length: int,
                          max_clause_length: int, seed: int = None)\
        -> list[list[tuple[boolean_var, bool]]]:
    """
        Generates a SAT instance with n_clauses clauses, each clause is a list of tuples (variable, is_not_negated)

        :param variables: list of boolean variables that can be used in the SAT instance
        :param min_n_clauses: minimum number of clauses
        :param max_n_clauses: maximum number of clauses
        :param min_clause_length: minimum number of variables in a clause
        :param max_clause_length: maximum number of variables in a clause
        :param seed: seed for the random number generator
    """
    if seed is not None:
        np.random.seed(seed)

    n_clauses = np.random.randint(min_n_clauses, max_n_clauses + 1)
    sat_instance = []
    for _ in range(n_clauses):
        clause_length = np.random.randint(min_clause_length, max_clause_length + 1)
        clause = []
        for _ in range(clause_length):
            variable = np.random.choice(variables)
            is_not_negated = np.random.choice([True, False])
            clause.append((variable, is_not_negated))
        sat_instance.append(clause)

    return sat_instance


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
                 feature_cost: list[int], alpha_sat: float):
        """
            :param sat_instance: list of clauses, each clause is a list of tuples (variable, is_not_negated)
            :param boolean_variables: ordered list of boolean variables
            :param feature_cost: list of costs for each feature
        """
        variables = []
        for clause in sat_instance:
            for variable, _ in clause:
                if variable not in variables:
                    variables.append(variable)

        for variable in variables:
            if variable not in boolean_variables:
                raise ValueError(f"variable {variable} is used in sat_instance but not in boolean_variables")

        if len(boolean_variables) != len(feature_cost):
            raise ValueError("boolean_variables and feature_cost must have the same length")

        self.sat_instance = sat_instance
        self.boolean_variables = boolean_variables
        self.feature_cost = feature_cost
        self.alpha_sat = alpha_sat

    def __str__(self) -> str:
        return f"sat_instance: " + self.sat_instance_to_string() + "\n" \
               f"n_features: {len(self.boolean_variables)}\n" \
               f"feature_cost: {self.feature_cost}\n"

    def sat_instance_to_string(self) -> str:
        instance_parts = []
        for clause in self.sat_instance:
            clause_parts = []
            for var, negated in clause:
                literal = r"\neg {}".format(list(var.keys())[0][0]) if negated else r"{}".format(list(var.keys())[0][0])
                clause_parts.append(literal)
            clause = r" \vee ".join(clause_parts)
            instance_parts.append(r"({})".format(clause))
        instance_string = r" \wedge ".join(instance_parts)
        return instance_string

    def get_sat_instance(self) -> list[list[tuple[boolean_var, bool]]]:
        return self.sat_instance

    def get_num_features(self) -> int:
        return len(self.boolean_variables)

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
