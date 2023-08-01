from qiskit_qasm3_import import parse
from qiskit import QuantumRegister
from qiskit.circuit import Qubit
from configproblem import grover_sat_qasm as gsq
from configproblem import grover_sat as gs

# (a v b) & !c
# = a!c v b!c
# = 110 v 100 v 010
problem = [[(0, True), (1, True)], [(2, False)]]
model = "../benchmarks/featureide-examples/pc-config.dimacs"

qasm = gsq.create_grover_for_model(model)
print("created qasm with depth:")
print(qasm)

qc = parse(qasm)
