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

if __name__ == '__main__':
    unittest.main()
