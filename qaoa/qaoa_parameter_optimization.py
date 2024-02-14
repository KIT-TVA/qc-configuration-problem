import math
from typing import Callable

import numpy as np
from scipy.optimize import minimize, OptimizeResult


def get_optimizer(method: str, maxiter: int = 1000, tol: float = 1e-12) -> Callable:
    """
        Returns the optimizer function for the given method.

        :param method: the optimization method used to optimize the parameters
        :param maxiter: the maximum number of iterations the optimizer is allowed to perform
        :param tol: the tolerance of the optimizer
    """
    if method == "COBYLA":
        optimizer = cobyla
    elif method == "Powell":
        optimizer = powell
    elif method == "BFGS":
        optimizer = bfgs
    elif method == "Nelder-Mead":
        optimizer = nelder_mead
    elif method == "CG":
        optimizer = cg
    else:
        raise ValueError("Invalid optimization method")

    def optimize(theta: list[dict], expectation: Callable, print_res: bool) -> list[dict]:
        """
            Optimizes the parameters beta and gamma of the QAOA circuit using the given method.

            :param theta: list of dictionaries with keys "beta" and "gamma" that parameterize the QAOA circuit,
                          used as start value when optimizing
            :param expectation: the expectation function used to determine the expected config cost
            :param print_res: indicates whether the results of the optimization should be printed
        """
        parameter_list = []
        for parameter_pair in theta:
            parameter_list.append(parameter_pair["beta"])
            parameter_list.append(parameter_pair["gamma"])

        res = optimizer(expectation, parameter_list, maxiter, tol)

        if print_res:
            print(res)
        optimized_theta = []
        for i in range(len(theta)):
            optimized_theta.append({"beta": res.x[2 * i], "gamma": res.x[(2 * i) + 1]})
        return optimized_theta

    return optimize


def get_bounds(parameter_list: list) -> list[tuple]:
    """
        Returns the bounds for the parameters beta and gamma.

        :param parameter_list: list of parameters beta and gamma that parameterize the QAOA circuit
    """
    bounds = []
    for i in range(math.floor(len(parameter_list) / 2)):
        bounds.append((-np.pi, np.pi))
        bounds.append((-2 * np.pi, 2 * np.pi))
    return bounds


def cobyla(expectation: Callable, parameter_list: list, maxiter: int, tol: float) -> OptimizeResult:
    """
        Optimizes the parameters beta and gamma of the QAOA circuit using the COBYLA algorithm.

        :param expectation: the expectation function used to determine the expected config cost
        :param parameter_list: list of parameters beta and gamma that parameterize the QAOA circuit
        :param maxiter: the maximum number of iterations the optimizer is allowed to perform
        :param tol: the tolerance of the optimizer
    """
    return minimize(expectation, np.array(parameter_list), method='COBYLA',
                    options={'maxiter': maxiter, 'disp': False}, tol=tol)


def powell(expectation: Callable, parameter_list: list, maxiter: int, tol: float) -> OptimizeResult:
    """
        Optimizes the parameters beta and gamma of the QAOA circuit using the Powell algorithm.

        :param expectation: the expectation function used to determine the expected config cost
        :param parameter_list: list of parameters beta and gamma that parameterize the QAOA circuit
        :param maxiter: the maximum number of iterations the optimizer is allowed to perform
        :param tol: the tolerance of the optimizer
    """
    bounds = get_bounds(parameter_list)
    return minimize(expectation, np.array(parameter_list), method='Powell', bounds=bounds,
                    options={'maxiter': maxiter, 'disp': False}, tol=tol)


def bfgs(expectation: Callable, parameter_list: list, maxiter: int, tol: float) -> OptimizeResult:
    """
        Optimizes the parameters beta and gamma of the QAOA circuit using the BFGS algorithm.

        :param expectation: the expectation function used to determine the expected config cost
        :param parameter_list: list of parameters beta and gamma that parameterize the QAOA circuit
        :param maxiter: the maximum number of iterations the optimizer is allowed to perform
        :param tol: the tolerance of the optimizer
    """
    return minimize(expectation, np.array(parameter_list), method='BFGS',
                    options={'maxiter': maxiter, 'disp': False}, tol=tol)


def nelder_mead(expectation: Callable, parameter_list: list, maxiter: int, tol: float) -> OptimizeResult:
    """
        Optimizes the parameters beta and gamma of the QAOA circuit using the Nelder-Mead algorithm.

        :param expectation: the expectation function used to determine the expected config cost
        :param parameter_list: list of parameters beta and gamma that parameterize the QAOA circuit
        :param maxiter: the maximum number of iterations the optimizer is allowed to perform
        :param tol: the tolerance of the optimizer
    """
    bounds = get_bounds(parameter_list)
    return minimize(expectation, np.array(parameter_list), method='Nelder-Mead', bounds=bounds,
                    options={'maxiter': maxiter, 'disp': False}, tol=tol)


def cg(expectation: Callable, parameter_list: list, maxiter: int, tol: float) -> OptimizeResult:
    """
        Optimizes the parameters beta and gamma of the QAOA circuit using the conjugate gradient algorithm.

        :param expectation: the expectation function used to determine the expected config cost
        :param parameter_list: list of parameters beta and gamma that parameterize the QAOA circuit
        :param maxiter: the maximum number of iterations the optimizer is allowed to perform
        :param tol: the tolerance of the optimizer
    """
    return minimize(expectation, np.array(parameter_list), method='CG',
                    options={'maxiter': maxiter, 'disp': False}, tol=tol)
