from typing import Any

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qubovert import boolean_var

import math

from qubovert.utils import DictArithmetic


def k_rz_gate(qc: QuantumCircuit, qubits: list, gate_parameter: float) -> QuantumCircuit:
    """
        Applies a Z gate across the given qubits with the given parameter

        :param qc: The quantum circuit to apply the gate to
        :param qubits: The qubits to apply the gate to
        :param gate_parameter: The parameter to apply the gate with
    """
    if len(qubits) == 0:
        return qc
    for index, qubit in enumerate(qubits):
        if index < len(qubits) - 1:
            qc.cnot(qubit, qubits[index + 1])

    qc.rz(gate_parameter, qubits[-1])

    qubits.reverse()

    for index, qubit in enumerate(qubits):
        if index < len(qubits) - 1:
            qc.cnot(qubits[index + 1], qubit)

    return qc


def problem_circuit(hamiltonian: DictArithmetic, nqubits: int) -> tuple[QuantumCircuit, Parameter]:
    """
        Creates a quantum circuit for the given hamiltonian

        :param hamiltonian: The hamiltonian to create the circuit for
        :param nqubits: The number of qubits to create the circuit for
    """
    gamma = Parameter("$\\gamma$")
    qc_p = QuantumCircuit(nqubits)
    normalization = math.pi/max(hamiltonian.values())

    for key, factor in hamiltonian.items():
        k_rz_gate(qc_p, list(key), 2 * normalization * factor * gamma)

    return qc_p, gamma


def exclude_config(sat_instance: list[list[tuple[boolean_var, bool]]], boolean_vars: list[boolean_var], config: str)\
        -> list[list[tuple[boolean_var, bool]]]:
    """
        Excludes the given configuration from the given sat instance by adding a clause to the sat instance

        :param sat_instance: The sat instance to exclude the configuration from
        :param boolean_vars: The variables of the sat instance
        :param config: The configuration to exclude
    """
    sat_exclusion_clause = []
    for index, value in enumerate(config[::-1]):
        if value == '0':
            sat_exclusion_clause.append((boolean_vars[index], True))
        else:
            sat_exclusion_clause.append((boolean_vars[index], False))
    sat_instance.append(sat_exclusion_clause)
    return sat_instance


def convert_ancilla_bit_results(results: dict[str, Any], nfeatures: int) -> dict[str, Any]:
    """
        Converts the results from the quantum computer to the results for the original problem

        :param results: The results from the quantum computer
        :param nfeatures: The number of features used in the problem
    """
    new_results = {}
    for key, value in results.items():
        new_key = key[-nfeatures:]
        if new_key in new_results:
            new_results[new_key] += value
        else:
            new_results[new_key] = value
    return new_results
