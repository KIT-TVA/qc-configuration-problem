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

from fragments.quantum_states_qasm import add_all_hadamards

np.set_printoptions(threshold=1e6)



def create_not_oracle(ctrl_name: str, ctrl_indices: [int] = None) -> str:
    """
        Constructs an oracle for boolean NOT,
        that is a X gate
    """
    if ctrl_indices is not None:
        return "\n".join([f"x {ctrl_name}[{i}];" for i in ctrl_indices])
    else:
        return f"x {ctrl_name};\n"


def create_and_oracle(ctrl_name: str, ctrl_indices: [int], tar: str) -> str:
    """
        Constructs an oracle for boolean AND,
        that is a multi-controlled X gate
    """
    cont_bits = ", ".join([f"{ctrl_name}[{i}]" for i in ctrl_indices[:]])
    qasm = f"ctrl({len(ctrl_indices)}) @ x {cont_bits}, {tar};\n"

    return qasm


def create_or_oracle(ctrl_name: str, ctrl_indices: [int], tar: str) -> str:
    """
        Constructs an oracle for boolean OR,
        from NOT (X) and AND oracles
    """
    # Negate all inputs
    neg_ctrl = create_not_oracle(ctrl_name, ctrl_indices)

    # Call AND oracle
    and_oracle = create_and_oracle(ctrl_name, ctrl_indices, tar)

    # Flip target
    flip_target = f"x {tar};"

    qasm = "\n".join([neg_ctrl,and_oracle,neg_ctrl,flip_target])

    return qasm


def create_clause_oracle(inp_reg: str, tar: str, clause: List[Tuple[int, bool]]) -> str:
    """
        Create an oracle for a SAT clause
    """
    qasm = ""
    # Flip all qubits which are negated in the clause
    qasm += create_not_oracle(inp_reg, [c[0] for c in filter(lambda c: not c[1], clause)]) + "\n"
    # Create an OR oracle for clause
    qasm += create_or_oracle(inp_reg, [c[0] for c in clause], tar) + "\n"
    # Inverse the initial flips
    qasm += create_not_oracle(inp_reg, [c[0] for c in filter(lambda c: not c[1], clause)]) + "\n"

    return qasm


def create_ksat_oracle(inp_reg: str, tar: str, clauses: List[List[Tuple[int, bool]]]) -> str:
    """
        Create an Oracle for a kSAT problem
    """
    ancilla_reg = "a"

    qasm = ""
    # Compose individual clauses
    for index, clause in enumerate(clauses):
        # Use one ancilla for each clause
        qasm += create_clause_oracle(inp_reg, f"{ancilla_reg}[{index}]", clause)

    # Store the conjugate transpose (inverse) for later qubit cleanup
    # inverse_qc = qc.inverse()
    inverse_qasm = "\n".join(qasm.split("\n")[::-1])

    # Use and oracle onto ancilla register and target
    qasm += create_and_oracle(ancilla_reg, range(len(clauses)), tar)

    # Inverse clause oracles
    # qc = qc.compose(inverse_qc)
    qasm += inverse_qasm

    return qasm


def oracle_converter(qasm: str, tar: str) -> (str,str):
    """
        Convert a bit-flip into a phase oracle
    """
    prepend = f"""
x {tar};
h {tar};
    """

    append = f"""
h {tar};
x {tar};
    """

    return prepend + qasm + append


def print_matrix(circuit):
    """
        Prints the matrix of a given quantum circuit (oracle) and analyzes it regarding positive and negative values
    """
    ppod = Operator(circuit).data
    print("Values in Operator", set(ppod.flatten()))
    print(f"Oracle Shape: {ppod.shape}",
          f"Elements;; nonzero:{np.count_nonzero(ppod)}, 1: {np.count_nonzero(ppod.real > 0.99)}, -1: {np.count_nonzero(ppod.real < -0.99)}")
    print("Diagonal values: ", [round(ppod[x][x].real) for x in range(len(ppod[0]))])
    print(ppod)


def print_diagonal_analysis(circuit, measurements=None):
    """
        Prints analysis data about a given quantum circuit (oracle).
        This method was used to determine whether one could trivially derive valid states from a given oracle.
        
        Keyword Arguments"
        circuit -- qiskit quantum circuit representing a grover oracle
        measurements -- dictionary of states and measurement counts as derived by qiskits counts() method after simulation
    """
    od = Operator(circuit).data
    diagonal_values = [round(od[x][x].real) for x in range(len(od[0]))]
    bspace_padding = math.ceil(math.log(len(diagonal_values), 2))

    if measurements is not None:
        measurement_threshold = max(measurements.values()) / (2 ** 2)
        measurements = {key: val for key, val in measurements.items() if val > measurement_threshold}

    for i, v in enumerate(diagonal_values):
        vs = " 1" if v == 1 else "-1"
        state_str = format(i, f"0{bspace_padding}b")

        # when measurements are available, check if a phase flipped state may be good
        state_marking = ""
        if measurements is not None and v == -1:
            if list(filter(state_str.endswith, measurements.keys())) != []:
                state_marking = "good"
            else:
                input_size = len(list(measurements)[0])
                ancilla_State = int(state_str[:-input_size])
                if ancilla_State == 0:
                    state_marking = "!AMBIGUOUS!"
                else:
                    state_marking = "invalid"

        print(vs, state_str, state_marking)


def init_sat_circuit(problem: List[List[Tuple[int, bool]]]) -> (int, int, str, str, str, str):
    """
        Returns calculated number of qubits, created circuit
    """
    print("init")
    # Number of input qubits
    num_vars = len(set([statement[0] for clause in problem for statement in clause]))
    # Number of ancialla qubits
    num_clauses = len(problem)
    num_qubits = num_vars + num_clauses + 1

    qasm = f"""
OPENQASM 3.0;
include "stdgates.inc";
"""

    # Init registers and qubits
    c_reg = "c"
    inp_reg = "q_in"
    tar = "q_tar"
    ancilla_reg = "a"
    qasm += f"""
qubit[{num_vars}] {inp_reg};
qubit[1] {tar};
qubit[{num_clauses}] {ancilla_reg};
bit [{num_vars}] {c_reg};
"""

    init = qasm

    oracle = create_ksat_oracle(inp_reg, tar, problem)
    phase_oracle = oracle_converter(oracle, tar)

    qasm += add_all_hadamards(inp_reg)
    return num_vars, num_qubits, init, qasm, oracle, phase_oracle


def diffuser(inp_reg: str, num_qubits: int) -> str:
    # Apply transformation |s> -> |00..0> (H-gates)
    qasm = add_all_hadamards(inp_reg)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    qasm += create_not_oracle(inp_reg)
    # Do multi-controlled-Z gate
    qasm += f"ctrl({num_qubits-1}) @ z {', '.join([f'{inp_reg}['+str(i)+']' for i in range(num_qubits-1)])}, {inp_reg}[{num_qubits-1}];\n"

    # Apply transformation |11..1> -> |00..0>
    qasm += create_not_oracle(inp_reg)
    # Apply transformation |00..0> -> |s>
    qasm += add_all_hadamards(inp_reg)
    return qasm


def create_ksat_grover(problem: List[List[Tuple[int, bool]]], k) -> (str, str, str, str):
    """
        Creates an circuit for the SAT problem instance and applies Grover k times
    """
    # Init sat circuit
    num_inp_qubits, num_qubits, init, main_qasm, qasm_oracle, qasm_phase_oracle = init_sat_circuit(problem)

    # Add grover diffuser
    diff = diffuser("q_in", num_inp_qubits)

    # Grover loop: add the oracle and diffusor step k times
    for i in range(k):
        main_qasm += qasm_phase_oracle
        main_qasm += diff

    main_qasm_pre_meas = main_qasm
    # Add measurements of input qubits
    main_qasm += "c = measure q_in;\n"

    return main_qasm, init, qasm_phase_oracle, main_qasm_pre_meas


def calc_statevector_from(counts, width=None):
    threshold = max(counts.values()) / 1e2  # one order of magnitude below the most often measured results
    count_vector = []
    shots = 0  # measured shots

    # derive width from counts if not given
    if width is None:
        width = len(list(counts.keys()))

    # create statevector by using counts
    for i in range(2 ** width):
        b = format(i, f"0{width}b")
        c = counts.get(b)
        # print(i, b, c)
        if c is None:
            count_vector.append(0)
        else:
            count_vector.append(c)
            shots += c

    # normalize vector
    count_arr = np.array(count_vector)
    norm_vector = count_arr / count_arr.sum()

    # sqrt vector
    statevector = np.sqrt(norm_vector)
    return statevector


def create_grover_for_model(rel_path, k=1):
    # load given model
    current_folder = os.path.dirname(os.path.realpath(__file__))
    some_model_path = os.path.join(current_folder, rel_path)

    if rel_path.split('.')[-1] == "xml":
        reader = Extended_Modelreader()
        feature_model, constraints = reader.readModel(some_model_path)
        # transform to cnf and then to problem
        feature_cnf = feature_model.build_cnf(constraints)
        print(feature_cnf)
        problem = feature_cnf.to_problem()

    elif rel_path.split('.')[-1] in ["dimacs", "cnf"]:
        rd = DimacsReader()
        rd.fromFile(some_model_path)
        problem = CNF().from_dimacs(rd).to_problem()

    print("created model")
    # create grover circuit
    problem_qc, _, _, _ = create_ksat_grover(problem, k)  # Create the circuit
    return problem_qc


def collect_circuit_info(circuit, backend="aer_simulator", shots=100, simulate=False):
    # transpile and collect meta data
    info = {}
    simulator = Aer.get_backend(backend)
    transpiled_grover_circuit = transpile(circuit, backend=simulator)
    info['depth'] = transpiled_grover_circuit.depth()
    info['width'] = transpiled_grover_circuit.num_qubits

    # print(f"Circuit depth: {transpiled_grover_circuit.depth()}gates - width: {transpiled_grover_circuit.num_qubits}qubits")

    # try to run/simulate
    if simulate:
        results = simulator.run(transpiled_grover_circuit, shots=shots).result()
        info['counts'] = results.get_counts()

    return info
