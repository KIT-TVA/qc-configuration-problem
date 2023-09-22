from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

import math

from qubovert.utils import DictArithmetic


def problem_circuit(hamiltonian: DictArithmetic, nqubits: int, param_name_appendix: str = '')\
        -> tuple[QuantumCircuit, Parameter]:
    """
        Creates a quantum circuit for the given hamiltonian

        :param hamiltonian: The hamiltonian to create the circuit for
        :param nqubits: The number of qubits to create the circuit for
        :param param_name_appendix: Appendix of the parameter name in case there are multiple parameters for different
                                    layers of the circuit
    """
    gamma = Parameter("$\\gamma$" + param_name_appendix)
    qc_p = QuantumCircuit(nqubits)
    normalization = math.pi/max(hamiltonian.values())
    
    for key, factor in hamiltonian.items():
        acting_qubits = len(key)
        
        if acting_qubits == 0:
            pass  # identity case
        elif acting_qubits == 1:
            # single qubit term
            q1 = key[0]
            qc_p.rz(2 * normalization * factor * gamma, q1)
        elif acting_qubits == 2:
            # quadratic qubit term
            q1 = key[0]
            q2 = key[1]
            qc_p.rzz(2 * normalization * factor * gamma, q1, q2)
        else:
            # non quadratic, error
            RuntimeError(f"Non quadratic term in hamiltonian: {key, factor}")
            
    return qc_p, gamma
