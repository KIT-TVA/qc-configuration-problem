import dataclasses
import re

from qiskit.circuit import Qubit, QuantumRegister, AncillaRegister, QuantumCircuit, Gate, ClassicalRegister
from qiskit.quantum_info import Operator
from qiskit import Aer, transpile

from typing import Dict, List, Tuple

import numpy as np
import math
import os

from configproblem.util.xml_reader import Extended_Modelreader
from configproblem.util.dimacs_reader import DimacsReader
from configproblem.util.cnf import CNF
from configproblem.util.qasm3 import QASM3

from configproblem.fragments.quantum_states_qasm import add_all_hadamards

np.set_printoptions(threshold=1e6)

def add_all_hadamards(qasm: QASM3, tar: str, indices: List[int]):
    for i in indices:
        qasm.add_command("h", (tar, i))

def create_not_oracle(qasm: QASM3, tar: str, indices: List[int]):
    for i in indices:
        qasm.add_command("x", (tar, i))

def create_and_oracle(qasm: QASM3, ctrl_name: str, ctrl_indices: List[int], tar: Tuple[str, int]):
    qasm.add_command("x", tar, list(map(lambda x: (ctrl_name, x), ctrl_indices)))

def create_or_oracle(qasm: QASM3, ctrl_name: str, ctrl_indices: List[int], tar: Tuple[str, int]):
    create_not_oracle(qasm, ctrl_name, ctrl_indices)
    create_and_oracle(qasm, ctrl_name, ctrl_indices, tar)
    create_not_oracle(qasm, ctrl_name, ctrl_indices)
    create_not_oracle(qasm, tar[0], [tar[1]])


def create_clause_oracle(qasm: QASM3, inp_reg: str, tar: Tuple[str, int], clauses: List[Tuple[int, bool]]):
    all_vars = [clause[0] for clause in clauses]
    negated_vars = list(map(lambda x: x[0], filter(lambda x: not x[1], clauses)))

    create_not_oracle(qasm, inp_reg, negated_vars)
    create_or_oracle(qasm, inp_reg, all_vars, tar)
    create_not_oracle(qasm, inp_reg, negated_vars)


def create_ksat_oracle(qasm: QASM3, inp_reg: str, tar: Tuple[str, int], clauses: List[List[Tuple[int, bool]]]):
    pre_index = None if(len(qasm.commands)-1 < 0) else len(qasm.commands)-1 < 0
    for index, clause in enumerate(clauses):
        create_clause_oracle(qasm, inp_reg, ("a", index), clause)
    inverse_commands = qasm.commands[:pre_index:-1]
    create_and_oracle(qasm, "a", list(range(len(clauses))), tar)
    qasm.add_commands(inverse_commands)

def oracle_converter(qasm: QASM3, tar: Tuple[str, int]):
    commands = [("x", tar, [], []), ("h", tar, [], [])]
    qasm.add_commands(commands, prepend=True)
    qasm.add_commands(commands[::-1])


def init_sat_circuit(problem: List[List[Tuple[int, bool]]]) -> (int, int, QASM3, QASM3):
    oracle = QASM3()
    qasm = QASM3()
    # calculate number of vars by counting distinct numbers
    num_vars = len(set([statement[0] for clause in problem for statement in clause]))
    oracle.add_qubit("feat_reg", num_vars)
    qasm.add_qubit("feat_reg", num_vars)

    oracle.add_qubit("tar")
    qasm.add_qubit("tar")

    num_clauses = len(problem)
    oracle.add_qubit("a", num_clauses)
    qasm.add_qubit("a", num_clauses)

    qasm.add_bit("feat_bit_reg", num_vars)

    create_ksat_oracle(oracle, "feat_reg", ("tar", 0), problem)
    oracle_converter(oracle, ("tar", 0))

    add_all_hadamards(qasm, "feat_reg", list(range(num_vars)))

    return num_vars, num_clauses, oracle, qasm

def add_diffuser(qasm: QASM3, inp_reg: str, num_qubits: int):
    add_all_hadamards(qasm, "feat_reg", list(range(num_qubits)))
    create_not_oracle(qasm, "feat_reg", list(range(num_qubits)))

    qasm.add_command("z", (inp_reg, num_qubits - 1), [(inp_reg, i) for i in range(num_qubits - 1)])

    create_not_oracle(qasm, "feat_reg", list(range(num_qubits)))
    add_all_hadamards(qasm, "feat_reg", list(range(num_qubits)))

def create_ksat_grover(problem: List[List[Tuple[int, bool]]], k: int) -> QASM3:
    (num_inp_qubits, num_clause, oracle, qasm) = init_sat_circuit(problem)

    for i in range(k):
        qasm.add_commands(oracle.commands)
        add_diffuser(qasm, "feat_reg", num_inp_qubits)

    return qasm

def create_grover_for_model(rel_path: str, k:int = 1) -> Tuple[QASM3, Dict[str, int]]:
    # load given model
    current_folder = os.path.dirname(os.path.realpath(__file__))
    some_model_path = os.path.join(current_folder, rel_path)

    if rel_path.split('.')[-1] == "xml":
        reader = Extended_Modelreader()
        feature_model, constraints = reader.readModel(some_model_path)
        # transform to cnf and then to problem
        feature_cnf = feature_model.build_cnf(constraints)
        problem, feature_names = feature_cnf.to_problem()
        return create_ksat_grover(problem, k), feature_names

    elif rel_path.split('.')[-1] in ["dimacs", "cnf"]:
        rd = DimacsReader()
        rd.fromFile(some_model_path)
        problem = CNF().from_dimacs(rd).to_problem()
        return create_ksat_grover(problem[0], k), problem[1]



