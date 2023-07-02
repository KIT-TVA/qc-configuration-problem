from qiskit.circuit import QuantumRegister, Qubit
from configproblem import grover_sat_qasm as gsq

r = QuantumRegister(3)
t = Qubit()

gsq.create_and_oracle(r, t)
