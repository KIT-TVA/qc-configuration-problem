from typing import List, Tuple, Any

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qubovert import boolean_var

from configproblem.util.model_transformation import convert_clause_to_penalty, convert_to_penalty
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


def exclude_config(sat_instance: list[list[tuple[boolean_var, bool]]], vars: list[boolean_var], config: str)\
        -> list[list[tuple[boolean_var, bool]]]:
    sat_exclusion_clause = []
    for index, value in enumerate(config[::-1]):
        if value == '0':
            sat_exclusion_clause.append((vars[index], True))
        else:
            sat_exclusion_clause.append((vars[index], False))
    sat_instance.append(sat_exclusion_clause)
    return sat_instance
