"""
Test Module for Grovers Algorithm according to Qiskit Textbook
In the future, replace the static oracle with our own that implements the configuration problem
"""

#initialization
import matplotlib.pyplot as plt
import numpy as np
import math

# importing Qiskit
from qiskit import IBMQ, Aer, BasicAer, transpile, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy
from qiskit.quantum_info.operators import Operator
import qiskit.compiler as qispiler

# import basic plot tools
from qiskit.visualization import plot_histogram

def initialize_s(qc, qubits):
    """Apply a H-gate to 'qubits' in qc"""
    for q in qubits:
        qc.h(q)
    return qc

def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    # We will return the diffuser as a gate
    U_s = qc.to_gate()
    U_s.name = "U$_{Diffuser}$"
    return U_s

def main():
    # Oracle
    nqubits = 8
    qc = QuantumCircuit(nqubits)
    # qc.cz(0, 2)
    #qc.cz(1, 2)
    oracle_matrix = [[1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, -1]]

    def create_oracle(nqubits, solutions):
        """ Create an identity matrix of size nqubits and flip phase of solutions"""
        I = Operator([[1,0],[0,1]])
        unitary = I
        for i in range(nqubits-1):
            unitary = unitary.tensor(I)
        
        for s in solutions:
            index = 0
            for i in range(nqubits):
                index += (2**i)*s[nqubits-i-1]
            unitary.data[index][index] = -1

        ## BEGIN TRANSPILATION
        unitary.name = "U$_{oracle}$"
        temporary_qc = QuantumCircuit(nqubits)
        temporary_qc.unitary(unitary, range(nqubits), label="oracle")
        gate_set_nairobi = ['h', 'cx', 'id', 'rz', 'sx', 'x'] # plus hadamard
        gate_set_default = ['h', 'u1', 'u2', 'u3', 'cx']
        backend = Aer.get_backend('aer_simulator_stabilizer')
        qcd = qispiler.transpile(temporary_qc, basis_gates=gate_set_default, optimization_level=3)
        # qcd = qispiler.transpile(temporary_qc, backend=backend, optimization_level=3)
        qcd.draw(output="mpl")
        ## END TRANSPILATION

        return unitary 

    solutions = [[1,0,0,1,1,0,0,1],[1,0,0,1,1,1,0,1]]
    #k = math.floor((math.pi/4)*math.sqrt(2**nqubits))
    #k = round(math.sqrt(float((2**nqubits)/len(solutions))))
    N = 2**nqubits
    M = len(solutions)
    theta = math.asin(2*math.sqrt(M*(N-M))/N)
    k = round(math.acos(math.sqrt(M/N))/theta)


    print(f"Number of iterations: {k}")

    #oracle_operator = Operator(oracle_matrix)
    qc.unitary(create_oracle(nqubits,solutions), range(nqubits), label="oracle")
    oracle_ex3 = qc.to_gate()
    oracle_ex3.name = "U$_\omega$"

    grover_circuit = QuantumCircuit(nqubits)
    grover_circuit = initialize_s(grover_circuit, range(nqubits))

    for i in range(k):
        grover_circuit.append(oracle_ex3, range(nqubits))
        grover_circuit.append(diffuser(nqubits), range(nqubits))

    grover_circuit.measure_all()
    grover_circuit.draw(output="mpl")

    #plt.show()

    qasm_sim = Aer.get_backend('qasm_simulator')
    transpiled_grover_circuit = transpile(grover_circuit, qasm_sim)
    results = qasm_sim.run(transpiled_grover_circuit).result()
    counts = results.get_counts()
    histogram = plot_histogram(counts)
    plt.show()

if __name__ == '__main__':
    main()