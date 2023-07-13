import unittest
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Qubit
from qiskit_qasm3_import import parse
from qiskit.quantum_info import Statevector

import grover_sat as gs
import grover_sat_qasm as gsq
class TestOracle(unittest.TestCase):

    def test_or(self):
        in_reg = QuantumRegister(3)
        tar = Qubit()
        or_oracle = gs.create_or_oracle(in_reg, tar)
        or_oracle_new = f"""OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q2;
qubit[1] q3;
{gsq.create_or_oracle("q2", range(3), "q3")}
"""
        or_oracle_new = parse(or_oracle_new)
        self.assertTrue(Statevector.from_instruction(or_oracle).equiv(Statevector.from_instruction(or_oracle_new)))

    def test_and(self):
        in_reg = QuantumRegister(3)
        tar = Qubit()
        or_oracle = gs.create_and_oracle(in_reg, tar)
        or_oracle_new = f"""OPENQASM 3.0;
    include "stdgates.inc";
    qubit[3] q2;
    qubit[1] q3;
    {gsq.create_and_oracle("q2", range(3), "q3")}
    """
        or_oracle_new = parse(or_oracle_new)
        self.assertTrue(Statevector.from_instruction(or_oracle).equiv(Statevector.from_instruction(or_oracle_new)))

    def test_clause_oracle(self):
        problem = [(0, True), (1, False), (2, True)]
        in_reg = QuantumRegister(3)
        tar = Qubit()
        or_oracle = gs.create_clause_oracle(in_reg, tar, problem)
        or_oracle_new = f"""OPENQASM 3.0;
       include "stdgates.inc";
       qubit[3] q2;
       qubit[1] q3;
       {gsq.create_clause_oracle("q2", "q3", problem)}
       """
        or_oracle_new = parse(or_oracle_new)
        self.assertTrue(Statevector.from_instruction(or_oracle).equiv(Statevector.from_instruction(or_oracle_new)))

    def test_ksat_oracle(self):
        problem = [[(0, True), (1, False), (2, True)], [(0, True), (1, True), (2, False)]]
        in_reg = QuantumRegister(3)
        tar = Qubit()
        or_oracle = gs.create_ksat_oracle(in_reg, tar, problem)
        or_oracle_new = f"""OPENQASM 3.0;
       include "stdgates.inc";
       qubit[3] q0;
       qubit[1] q_tar;
       qubit[2] a;
       {gsq.create_ksat_oracle("q0", "q_tar", problem)}
       """
        or_oracle_new = parse(or_oracle_new)
        print(or_oracle.decompose(reps=5))
        print(or_oracle_new.decompose(reps=5))
        self.assertTrue(Statevector.from_instruction(or_oracle).equiv(Statevector.from_instruction(or_oracle_new)))

    def test_oracle_converter(self):
        problem = [[(0, True), (1, False), (2, True)], [(0, True), (1, True), (2, False)]]
        in_reg = QuantumRegister(3)
        tar = Qubit()
        or_phase_oracle = gs.oracle_converter(gs.create_ksat_oracle(in_reg, tar, problem), len(in_reg))
        or_phase_oracle_new = f"""OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q0;
        qubit[1] q_tar;
        qubit[2] a;
        {gsq.oracle_converter(gsq.create_ksat_oracle("q0", "q_tar", problem), "q_tar")}
        """
        or_phase_oracle_new = parse(or_phase_oracle_new)
        print(or_phase_oracle.decompose(reps=5))
        print(or_phase_oracle_new.decompose(reps=5))
        self.assertTrue(Statevector.from_instruction(or_phase_oracle).equiv(Statevector.from_instruction(or_phase_oracle_new)))

    def test_init_sat(self):
        problem = [[(0, True), (1, False), (2, True)], [(0, True), (1, True), (2, False)]]

        num_vars, num_qubits, main_qc, qc_oracle, qc_phase_oracle = gs.init_sat_circuit(problem)
        num_vars_new, num_qubits_new, init, main_qasm_new, qasm_oracle_new, qasm_phase_oracle_new = gsq.init_sat_circuit(problem)

        self.assertEqual(num_vars_new, num_vars)
        self.assertEqual(num_qubits_new, num_qubits)

        main_qc_new = parse(main_qasm_new)
        self.assertTrue(Statevector.from_instruction(main_qc).equiv(Statevector.from_instruction(main_qc_new)))

        qc_oracle_new = parse(init + qasm_oracle_new)
        self.assertTrue(Statevector.from_instruction(qc_oracle).equiv(Statevector.from_instruction(qc_oracle_new)))

        qc_phase_oracle_new = parse(init + qasm_phase_oracle_new)
        self.assertTrue(Statevector.from_instruction(qc_phase_oracle).equiv(Statevector.from_instruction(qc_phase_oracle_new)))

    def test_diffuser(self):
        init = f"""
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q0;
        """
        in_reg = QuantumRegister(3)

        main_qc = QuantumCircuit(in_reg)

        main_qc = main_qc.compose(gs.diffuser(3))
        diffuser_new = gsq.diffuser("q0", 3)
        diffuser_new = parse(init + diffuser_new)

        self.assertTrue(Statevector.from_instruction(main_qc).equiv(Statevector.from_instruction(diffuser_new)))

    def test_ksat_grover(self):
        problem = [[(0, True), (1, False), (2, True)], [(0, True), (1, True), (2, False)]]

        main_qc, _ = gs.create_ksat_grover(problem, 2)
        main_qasm_new, init, _ = gsq.create_ksat_grover(problem, 2)

        main_qc_new = parse(main_qasm_new)

        print(main_qc_new.qasm())

        print(main_qc.decompose(["U$_{oracle}$", "U$_{Diffuser}$", "$U_{ksat}$", "$U_{clause}$",
                                 "$U_{and}$", "$U_{or}$", "circuit-87_dg", "circuit-96_dg",
                                 "circuit-97_dg", "circuit-88_dg", "circuit-98_dg", "circuit-89_dg",
                                 ], reps=5))
        print(main_qc_new)

        self.assertTrue(Statevector.from_instruction(main_qc).equiv(Statevector.from_instruction(main_qc_new)))



if __name__ == '__main__':
    unittest.main()
