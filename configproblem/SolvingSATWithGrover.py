from qiskit.circuit import Qubit, QuantumRegister, AncillaRegister, QuantumCircuit, Gate, ClassicalRegister
from qiskit.quantum_info import Operator
from qiskit import Aer, transpile

from typing import Dict, List, Tuple

import numpy as np
import math
import os
np.set_printoptions(threshold=1e6)

from grover import diffuser
from util.xml_reader import Extended_Modelreader
from util.dimacs_reader import DimacsReader
from util.cnf import CNF


def create_and_oracle(inp_reg: QuantumRegister, tar: Qubit) -> QuantumCircuit:
    """
        Constructs an oracle for boolean AND,
        that is a multi-controlled X gate
    """
    tar_reg = QuantumRegister(bits=[tar])
    qc = QuantumCircuit(inp_reg, tar_reg)

    qc.mcx(inp_reg, tar_reg)

    return qc


def create_or_oracle(inp_reg: QuantumRegister, tar: Qubit) -> QuantumCircuit:
    """
        Constructs an oracle for boolean OR,
        from NOT (X) and AND oracles
    """
    tar_reg = QuantumRegister(bits=[tar])
    qc = QuantumCircuit(inp_reg, tar_reg)

    # Negate all inputs
    for i in inp_reg:
        qc.x(i)
    
    # Call AND oracle
    and_oracle = create_and_oracle(inp_reg, tar).to_gate(label="$U_{and}$")
    qc.append(and_oracle, inp_reg[:]+tar_reg[:])

    # Inverse negation
    for i in inp_reg:
        # Inverse of x is x
        qc.x(i)

    # Flip target
    qc.x(tar_reg[0])
    
    return qc


def get_clause_qubits(inp_reg: QuantumRegister, clause: List[Tuple[int, bool]]) -> List[Qubit]:
    """
        Return a register containing only relevant qubits for a SAT clause
    """

    clause_qubits:list(Qubit) = []

    for index, _ in clause:
        clause_qubits.append(inp_reg[index])

    return clause_qubits


def create_clause_oracle(inp_reg: QuantumRegister, tar: Qubit, clause: List[Tuple[int, bool]]) -> QuantumCircuit:
    """
        Create an oracle for a SAT clause
    """
    tar_reg = QuantumRegister(bits=[tar], name="q_tar")
    qc = QuantumCircuit(inp_reg, tar_reg)

    # Flip all qubits which are negated in the clause
    for index, positive in clause:
        if not positive:
            qc.x(index)
    
    
    # Get Clause Qubits
    clause_qubits = get_clause_qubits(inp_reg, clause)
    clause_reg = QuantumRegister(bits=clause_qubits)

    # Create an OR oracle for clause
    clause_oracle = create_or_oracle(clause_reg, tar).to_gate(label="$U_{or}$")
    qc.append(clause_oracle, clause_reg[:]+tar_reg[:])

    # Inverse the initial flips
    for index, positive in clause:
        if not positive:
            qc.x(index)

    return qc


def create_ksat_oracle(inp_reg: QuantumRegister, tar: Qubit, clauses: List[List[Tuple[int, bool]]]) -> Gate:
    """
        Create an Oracle for a kSAT problem
    """
    ancilla_reg = AncillaRegister(len(clauses), name="a")
    tar_reg = QuantumRegister(bits=[tar], name="q_tar")
    qc = QuantumCircuit(inp_reg, tar_reg, ancilla_reg)

    # Compose individual clauses
    for index, clause in enumerate(clauses):
        # Use one ancilla for each clause
        clause_oracle = create_clause_oracle(inp_reg, ancilla_reg[index], clause).to_gate(label="$U_{clause}$")
        qc.append(clause_oracle, inp_reg[:]+[ancilla_reg[index]])
    
    # Store the conjugate transpose (inverse) for later qubit cleanup
    inverse_qc = qc.inverse()
    
    # Use and oracle onto ancilla register and target
    and_oracle = create_and_oracle(ancilla_reg, tar).to_gate(label="$U_{and}$")
    qc.append(and_oracle, ancilla_reg[:]+tar_reg[:])

    # Inverse clause oracles
    qc = qc.compose(inverse_qc)

    return qc


def oracle_converter(oracle_qc: QuantumCircuit, target_idx: int) -> QuantumCircuit:
    """
        Convert a bit-flip into a phase oracle
    """
    phase_qc = oracle_qc.copy()

    qc_conv = QuantumCircuit(1, name="$U_{phase}$")
    qc_conv.x(0)
    qc_conv.h(0)
    # Prepend the phase transformation
    phase_qc = phase_qc.compose(qc_conv, qubits=[target_idx], front=True)
    # Append the phase transformation 
    phase_qc = phase_qc.compose(qc_conv.inverse(), qubits=[target_idx])

    return phase_qc


def print_matrix(circuit):
    """
        Prints the matrix of a given quantum circuit (oracle) and analyzes it regarding positive and negative values
    """
    ppod = Operator(circuit).data
    print("Values in Operator", set(ppod.flatten()))
    print(f"Oracle Shape: {ppod.shape}", f"Elements;; nonzero:{np.count_nonzero(ppod)}, 1: {np.count_nonzero(ppod.real > 0.99)}, -1: {np.count_nonzero(ppod.real < -0.99)}")
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
        measurement_threshold = max(measurements.values()) / (2**2)
        measurements = {key:val for key, val in measurements.items() if val > measurement_threshold}
    
    for i, v in enumerate(diagonal_values):
        vs = " 1" if  v == 1 else "-1"
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
        

def initialize_s(qc, qubits):
    """
        Apply a H-gate to 'qubits' in qc
    """
    for q in qubits:
        qc.h(q)
    return qc


def init_sat_circuit(problem):
    """
        Returns calculated number of qubits, created circuit
    """
    # Number of input qubits
    num_vars = len(set([statement[0] for clause in problem for statement in clause]))
    # Number of ancialla qubits
    num_clauses = len(problem)
    num_qubits = num_vars + num_clauses + 1

    # Init registers and qubits
    inp_reg = QuantumRegister(num_vars, name="q_in")
    tar = Qubit()
    tar_reg = QuantumRegister(bits=[tar], name="q_tar")
    ancilla_reg = AncillaRegister(num_clauses, name="a")

    # Create oracle for this SAT problem instance
    qc_oracle = QuantumCircuit(num_qubits)
    qc_oracle.append(create_ksat_oracle(inp_reg, tar, problem).to_gate(label="$U_{ksat}$"), qc_oracle.qubits)
    qc_phase_oracle = oracle_converter(qc_oracle, len(inp_reg))

    # Construct main quantum circuit
    c_regs = ClassicalRegister(num_vars, 'c')
    main_qc = QuantumCircuit(inp_reg, tar_reg, ancilla_reg, c_regs)

    # Create uniform superposition
    main_qc = initialize_s(main_qc, range(num_vars))

    return (num_vars, num_qubits, main_qc, qc_oracle, qc_phase_oracle)


def create_ksat_grover(problem: List[List[Tuple[int, bool]]], k) -> Tuple[QuantumCircuit, QuantumCircuit]:
    """
        Creates an circuit for the SAT problem instance and applies Grover k times
    """
    # Init sat circuit
    num_inp_qubits, num_qubits, main_qc, qc_oracle, qc_phase_oracle = init_sat_circuit(problem)

    # Add grover diffuser
    diff = diffuser(num_inp_qubits)

    # Grover loop: add the oracle and diffusor step k times
    phase_oracle_gate = qc_phase_oracle.to_gate(label='U$_{oracle}$')
    register_map = list(range(num_inp_qubits))
    for i in range(k):
        main_qc.append(phase_oracle_gate, range(num_qubits))
        main_qc = main_qc.compose(diff, register_map)
        
    # Add measurements of input qubits
    main_qc.measure(register_map, register_map)
#     main_qc.measure_all()
    
    return (main_qc, qc_phase_oracle)


def calc_statevector_from(counts, width=None):
    threshold = max(counts.values())/1e2 # one order of magnitude below the most often measured results
    count_vector = []
    shots = 0 # measured shots
    
    # derive width from counts if not given
    if width is None:
        width = len(list(counts.keys()))
    
    # create statevector by using counts
    for i in range(2**width):
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
    current_folder = globals()['_dh'][0]
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
    
    # create grover circuit
    problem_qc, problem_oracle = create_ksat_grover(problem, k) # Create the circuit
    return problem_qc


def collect_circuit_info(circuit, backend="aer_simulator", shots=100, simulate=False):
    # transpile and collect meta data
    info = {}
    simulator = Aer.get_backend(backend)
    transpiled_grover_circuit = transpile(circuit, backend=simulator)
    info['depth'] = transpiled_grover_circuit.depth()
    info['width'] = transpiled_grover_circuit.num_qubits
    
    #print(f"Circuit depth: {transpiled_grover_circuit.depth()}gates - width: {transpiled_grover_circuit.num_qubits}qubits")
    
    # try to run/simulate
    if simulate:
        results = simulator.run(transpiled_grover_circuit, shots=shots).result()
        info['counts'] = results.get_counts()
    
    return info