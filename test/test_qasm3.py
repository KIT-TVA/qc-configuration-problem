from qiskit_qasm3_import import parse
from qiskit import QuantumRegister
from qiskit.circuit import Qubit
from configproblem import grover_sat_qasm as gsq
from configproblem import grover_sat as gs

# (a v b) & !c
# = a!c v b!c
# = 110 v 100 v 010
if __name__ == "__main__":
    try:
        problem = [[(0, True), (1, True)], [(2, False)]]

        model = "../benchmarks/featureide-examples/pc-config.dimacs"

        #qasm = gsq.create_grover_for_model(model)
        main_qasm, init, qasm_phase_oracle, main_qasm_pre_meas = gsq.create_ksat_grover(problem, 1)

        print("created qasm with depth:")
        # print(qasm)
        print(gsq.get_circuit_depth_qasm(main_qasm))
    except KeyboardInterrupt:
        pass