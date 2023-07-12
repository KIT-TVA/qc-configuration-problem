from qiskit_qasm3_import import parse
from qiskit import QuantumRegister
from qiskit.circuit import Qubit
from configproblem import grover_sat_qasm as gsq
from configproblem import grover_sat as gs

# (a v b) & !c
# = a!c v b!c
# = 110 v 100 v 010
problem = [[(0, True), (1, True)], [(2, False)]]


# new
print("<================ New ================>")
_, _, _, qc = gsq.init_sat_circuit(problem)
qc = parse(qc)
print(qc)
print(qc.qasm())

# old
print("<================ Old ================>")
_, _, _, _, qc = gs.init_sat_circuit(problem)
print(qc)
print(qc.qasm())
