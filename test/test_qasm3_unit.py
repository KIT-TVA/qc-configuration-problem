import unittest

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Qubit
from qiskit.quantum_info import Statevector

import grover_sat_qasm_new as gsq
import grover_sat as gs

from configproblem.util.qasm3 import QASM3

from qiskit_qasm3_import import parse

class MyTestCase(unittest.TestCase):
    gates = ["$U_{ksat}$", "U$_{oracle}$", "U$_{diffuser}$", "$U_{clause}$", "$U_{and}$", "$U_{or}$"] + [f"circuit-{i}_dg" for i in range(200)]
    def _compare_(self, qc: QuantumCircuit, qc_new: QuantumCircuit):
        print(qc.decompose(self.gates, reps=5))
        print(qc_new)
        return Statevector.from_instruction(qc).equiv(Statevector.from_instruction(qc_new))
    def test_not_data(self):
        qasm = QASM3()
        qasm.add_qubit("in_reg", 3)
        gsq.create_not_oracle(qasm, "in_reg", [0, 1, 2])
        qc_new = parse(qasm.to_qasm())

    def test_and_data(self):
        qasm = QASM3()
        qasm.add_qubit("in_reg", 3)
        qasm.add_qubit("tar")
        gsq.add_all_hadamards(qasm, "in_reg", [0, 1, 2])
        gsq.create_and_oracle(qasm, "in_reg", [0, 1, 2], ("tar", 0))
        qc_new = parse(qasm.to_qasm())
        reg = QuantumRegister(3)
        tar = Qubit()
        qc = QuantumCircuit(reg, [tar])
        qc.h(reg)
        qc.compose(gs.create_and_oracle(reg, tar), [0,1,2,3], inplace=True)

        print(qc_new)
        print(qc)
        self.assertTrue(self._compare_(qc, qc_new))

    def test_or_data(self):
        qasm = QASM3()
        qasm.add_qubit("in_reg", 3)
        qasm.add_qubit("tar")
        gsq.add_all_hadamards(qasm, "in_reg", [0, 1, 2])
        gsq.create_or_oracle(qasm, "in_reg", [0, 1, 2], ("tar", 0))
        qc_new = parse(qasm.to_qasm())
        reg = QuantumRegister(3)
        tar = Qubit()
        qc = QuantumCircuit(reg, [tar])
        qc.h(reg)
        qc.compose(gs.create_or_oracle(reg, tar), [0,1,2,3], inplace=True)

        print(qc_new)
        print(qc)
        self.assertTrue(self._compare_(qc, qc_new))

    def test_clause_oracle_data(self):
        problem = [(0, True), (1, True), (2, False)]
        qasm = QASM3()
        qasm.add_qubit("in_reg", 3)
        qasm.add_qubit("tar")
        gsq.add_all_hadamards(qasm, "in_reg", [0, 1, 2])
        gsq.create_clause_oracle(qasm, "in_reg", ("tar", 0), problem)
        qc_new = parse(qasm.to_qasm())
        reg = QuantumRegister(3)
        tar = Qubit()
        qc = QuantumCircuit(reg, [tar])
        qc.h(reg)
        qc.compose(gs.create_clause_oracle(reg, tar, problem), [0,1,2,3], inplace=True)

        print(qc_new)
        print(qc)
        self.assertTrue(self._compare_(qc, qc_new))

    def test_ksat_oracle_data(self):
        problem = [[(0, True), (1, False)], [(2, True)], [(1, True), (3, False)]]
        qasm = QASM3()
        qasm.add_qubit("in_reg", 4)
        qasm.add_qubit("tar")
        qasm.add_qubit("a", 3)
        gsq.add_all_hadamards(qasm, "in_reg", [0, 1, 2, 3])
        gsq.create_ksat_oracle(qasm, "in_reg", ("tar", 0), problem)
        qc_new = parse(qasm.to_qasm())
        reg = QuantumRegister(4)
        tar = Qubit()
        anc = QuantumRegister(3)
        qc = QuantumCircuit(reg, [tar], anc)
        qc.h(reg)
        qc.compose(gs.create_ksat_oracle(reg, tar, problem), list(range(8)), inplace=True)

        print(qc_new)
        print(qc.decompose(self.gates, reps=3))
        self.assertTrue(self._compare_(qc, qc_new))
    def test_oracle_converter(self):
        problem = [[(0, True), (1, False)], [(2, True)], [(1, True), (3, False)]]
        qasm = QASM3()
        qasm.add_qubit("in_reg", 4)
        qasm.add_qubit("tar")
        qasm.add_qubit("a", 3)
        gsq.add_all_hadamards(qasm, "in_reg", [0, 1, 2, 3])
        gsq.create_ksat_oracle(qasm, "in_reg", ("tar", 0), problem)
        gsq.oracle_converter(qasm, ("tar", 0))
        qc_new = parse(qasm.to_qasm())
        reg = QuantumRegister(4)
        tar = Qubit()
        anc = QuantumRegister(3)
        qc = QuantumCircuit(reg, [tar], anc)
        qc.h(reg)
        qc.compose(gs.create_ksat_oracle(reg, tar, problem), list(range(8)), inplace=True)
        qc = gs.oracle_converter(qc, 4)

        print(qc_new)
        print(qc.decompose(self.gates, reps=3))
        self.assertTrue(self._compare_(qc, qc_new))

    def test_diffuser(self):
        qasm = QASM3()
        qasm.add_qubit("in_reg", 3)
        gsq.add_diffuser(qasm, "in_reg", 3)
        qc_new = parse(qasm.to_qasm())

        qc = QuantumCircuit(3)
        qc = qc.compose(gs.diffuser(3), list(range(3)))

        print(qc_new)
        print(qc.decompose(self.gates, reps=3))
        self.assertTrue(self._compare_(qc, qc_new))

    def test_init_sat_circuit(self):
        problem = [[(0, True), (1, False)], [(2, True)], [(1, True), (3, False)]]
        num_vars_new, num_clauses, oracle_new, qasm_new = gsq.init_sat_circuit(problem)
        qc_oracle_new = parse(oracle_new.to_qasm())
        qc_qasm_new = parse(qasm_new.to_qasm())
        num_vars, num_qubits, qasm, _, phase_oracle = gs.init_sat_circuit(problem)
        self.assertEqual(num_vars_new, num_vars)
        self.assertEqual(num_clauses + num_vars + 1, num_qubits)
        self.assertTrue(self._compare_(phase_oracle, qc_oracle_new))
        self.assertTrue(self._compare_(qasm, qc_qasm_new))


    def test_ksat_grover(self):
        problem = [[(0, True), (1, False)], [(2, True)], [(1, True), (3, False)]]
        qasm = gsq.create_ksat_grover(problem, 1)
        qc_new = parse(qasm.to_qasm())
        _, _, qc = gs.create_ksat_grover(problem, 1)

        print(qc_new.draw(output="text"))
        print(qc.decompose(self.gates, reps=5).draw(output="text"))
        self.assertTrue(self._compare_(qc, qc_new))
        print(qc.decompose(self.gates, reps=5).depth())

    def test_grover_for_model(self):
        model = "../benchmarks/featureide-examples/sandwich.dimacs"
        qasm = gsq.create_grover_for_model(model, k=1)
        qc_new = parse(qasm.to_qasm())
        qc = gs.create_grover_for_model(model, k=1)
        depth = qasm.calculate_depth()
        print(f"calculated: {max(depth.values())},  expected: {qc_new.depth()}, wrt: {qc.decompose(self.gates, reps=5).depth()}")
        self.assertTrue(self._compare_(qc_new, qc))

if __name__ == '__main__':
    unittest.main()
