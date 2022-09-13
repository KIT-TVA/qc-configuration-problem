# QISKIT USES LITTLE ENDIAN NOTATION!

from unittest import result
from qiskit.circuit import Qubit, QuantumRegister, QuantumCircuit
from qiskit.quantum_info import Operator

def create_and_oracle(inp_reg: QuantumRegister, tar: Qubit):
    """
        Constructs an oracle for boolean AND,
        that is a multi-controlled X gate
    """
    tar_reg = QuantumRegister(bits=[tar])
    qc = QuantumCircuit(inp_reg, tar_reg)

    qc.mcx(inp_reg, tar_reg)

    return qc.to_gate(label="$U_{AND}$")

def create_or_oracle(inp_reg: QuantumRegister, tar: Qubit):
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
    and_oracle = create_and_oracle(inp_reg, tar)
    qc.append(and_oracle, inp_reg[:]+tar_reg[:])

    # Inverse negation
    for i in inp_reg:
        # Inverse of x is x
        qc.x(i)

    # Flip target
    qc.x(tar_reg[0])
    
    return qc.to_gate(label="$U_{OR}$")

import unittest
from numpy.testing import assert_array_equal

from qiskit.circuit.library import CXGate, CCXGate
from qiskit.providers.aer import StatevectorSimulator
from qiskit import transpile
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector

class TestSATOracle(unittest.TestCase):

    test_backend = StatevectorSimulator()

    def assert_operation_statevecs(self, mapping, op, inp_reg, tar_reg):
        """
            For a given operator (op) check that the defined state vector mapping holds
        """
        for input, expected in mapping.items():
            input_vec = Statevector.from_label(input)

            qc = QuantumCircuit(inp_reg, tar_reg)
            qc.initialize(input_vec, qc.qubits)
            qc.append(op, inp_reg[:]+tar_reg[:])
            # print(qc.draw(output='text'))

            transpiled_qc = transpile(qc, self.test_backend)
            results = self.test_backend.run(transpiled_qc).result()
            counts = results.get_counts()
            # print(counts)

            self.assertTrue(counts[expected] == 1.0)
    
    def test_and_oracle(self):
        inp_reg1 = QuantumRegister(1)
        inp_reg2 = QuantumRegister(2)
        tar = Qubit()
        tar_reg = QuantumRegister(bits=[tar])

        # Operator of oracle equals cx with one input / control qubit
        and_op1 = Operator(create_and_oracle(inp_reg1, tar))
        cx_op = Operator(CXGate())

        assert_array_equal(and_op1.data, cx_op.data)

        # Operator of oracle equals ccx with two input / control qubits
        and_op2 = Operator(create_and_oracle(inp_reg2, tar))
        ccx_op = Operator(CCXGate())

        assert_array_equal(and_op2.data, ccx_op.data)

        # Now test for CCX (as at least 3 qubits are required) / and oracle that it behaves as expected
        # So for each input basis state check that the expected output basis state is computed
        and_oracle_expected_statevec_mapping = {
            # Little endian: t, c2, c1
            '000': '000',
            '001': '001',
            '010': '010',
            '011': '111', # If both controls are true, flip target
            '100': '100',
            '101': '101',
            '110': '110',
            '111': '011', # If both controls are true, flip target
        }

        self.assert_operation_statevecs(
            and_oracle_expected_statevec_mapping, and_op2, inp_reg2, tar_reg)

    def test_or_oracle(self):
        inp_reg2 = QuantumRegister(2)
        tar = Qubit()
        tar_reg = QuantumRegister(bits=[tar])

        or_op2 = Operator(create_or_oracle(inp_reg2, tar))
        
        # Test for 3 qubits or oracle that it behaves as expected
        # So for each input basis state check that the expected output basis state is computed
        or_oracle_expected_statevec_mapping = {
            # Little endian: t, c2, c1
            '000': '000',
            '001': '101', # If at least one control is true, flip target
            '010': '110', # If at least one control is true, flip target
            '011': '111', # If at least one control is true, flip target
            '100': '100',
            '101': '001', # If at least one control is true, flip target
            '110': '010', # If at least one control is true, flip target
            '111': '011', # If at least one control is true, flip target
        }

        self.assert_operation_statevecs(
            or_oracle_expected_statevec_mapping, or_op2, inp_reg2, tar_reg)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)