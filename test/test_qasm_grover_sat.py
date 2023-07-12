from qiskit import QuantumCircuit, Aer, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import StatevectorSimulator
from qiskit.circuit import QuantumRegister, Qubit, AncillaRegister
from configproblem import grover_sat_qasm as gsq
from configproblem import grover_sat as gs

inp_reg = QuantumRegister(7, name="q_in")
tar = Qubit()
tar_reg = QuantumRegister(bits=[tar], name="q_tar")
ancilla_reg = AncillaRegister(3, name="a")

problem_f = [[(k, True),(k+1, False)] for k in range(300)]

qc, _ = gs.create_ksat_grover(problem_f, 1)
transpiled_grover_circuit = transpile(qc, StatevectorSimulator())
results = StatevectorSimulator().run(transpiled_grover_circuit, shots=1000).result()
counts = results.get_counts()
print(counts)
