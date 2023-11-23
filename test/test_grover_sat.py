from configproblem.grover_sat import create_and_oracle, create_or_oracle, create_clause_oracle, create_ksat_oracle, create_ksat_grover

from qiskit.circuit import Qubit, QuantumRegister, AncillaRegister, QuantumCircuit
from qiskit.quantum_info import Operator
import unittest
from numpy.testing import assert_array_equal

from qiskit.circuit.library import CXGate, CCXGate
from qiskit.providers.aer import StatevectorSimulator
from qiskit import transpile
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector

class TestSATOracle(unittest.TestCase):

    test_backend = StatevectorSimulator()

    def assert_operation_statevecs(self, mapping, op, inp_reg, tar_reg, ancilla_reg=None):
        """
            For a given operator (op) check that the defined state vector mapping holds
        """
        for input, expected in mapping.items():
            input_vec = Statevector.from_label(input)

            if(ancilla_reg is None):
                qc = QuantumCircuit(inp_reg, tar_reg)
                qc.initialize(input_vec, qc.qubits)
                qc.append(op, inp_reg[:]+tar_reg[:])
            else: 
                qc = QuantumCircuit(inp_reg, tar_reg, ancilla_reg)
                qc.initialize(input_vec, qc.qubits)
                qc.append(op, inp_reg[:]+tar_reg[:]+ancilla_reg[:])
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
            # Little endian: t, in2, in1
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
            # Little endian: t, in2, in1
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

    def test_clause_oracle(self):
        inp_reg2 = QuantumRegister(2)
        inp_reg3 = QuantumRegister(3)
        tar = Qubit()
        tar_reg = QuantumRegister(bits=[tar])

        clause = [(0, True),(1, False)]
        clause_op2 = Operator(create_clause_oracle(inp_reg2, tar, clause))

        # Test for 3 qubits or oracle that it behaves as expected
        # So for each input basis state check that the expected output basis state is computed
        clause_oracle_expected_statevec_mapping1 = {
            # Little endian: t, in2, in1
            # Flip t if in1 or not(in2)
            '000': '100',
            '001': '101',
            '010': '010',
            '011': '111',
            '100': '000',
            '101': '001',
            '110': '110',
            '111': '011',
        }

        self.assert_operation_statevecs(
            clause_oracle_expected_statevec_mapping1, clause_op2, inp_reg2, tar_reg)

        clause = [(0, True),(1, False),(2,False)]
        clause_op3 = Operator(create_clause_oracle(inp_reg3, tar, clause))
        # Also test for a more complex clause
        clause_oracle_expected_statevec_mapping2 = {
            # Little endian: t, in3, in2, in1
            # Flip t if in1 or not(in2) or not(in3)
            '0000': '1000',
            '0001': '1001',
            '0010': '1010',
            '0011': '1011',
            '0100': '1100',
            '0101': '1101',
            '0110': '0110',
            '0111': '1111',
            '1000': '0000',
            '1001': '0001',
            '1010': '0010',
            '1011': '0011',
            '1100': '0100',
            '1101': '0101',
            '1110': '1110',
            '1111': '0111',
        }

        self.assert_operation_statevecs(
            clause_oracle_expected_statevec_mapping2, clause_op3, inp_reg3, tar_reg)

    def test_ksat_oracle(self):
        inp_reg3 = QuantumRegister(3)
        tar = Qubit()
        tar_reg = QuantumRegister(bits=[tar])
        ancilla_reg2 = AncillaRegister(2)

        problem = [[(2, True)], [(0, True),(1, False)]]
        ksat_op = Operator(create_ksat_oracle(inp_reg3, tar, problem))

        clause_oracle_expected_statevec_mapping2 = {
            # Little endian: a2, a1, t, in3, in2, in1
            # We ignore the ancillas so just set them to 0
            # Flip t if (in3) and (in1 or not(in2))
            '000000': '000000',
            '000001': '000001',
            '000010': '000010',
            '000011': '000011',
            '000100': '001100', # Flip
            '000101': '001101', # Flip
            '000110': '000110',
            '000111': '001111', # Flip
            '001000': '001000',
            '001001': '001001',
            '001010': '001010',
            '001011': '001011',
            '001100': '000100', # Flip
            '001101': '000101', # Flip
            '001110': '001110',
            '001111': '000111', # Flip
        }

        self.assert_operation_statevecs(
            clause_oracle_expected_statevec_mapping2, ksat_op, inp_reg3, tar_reg, ancilla_reg2)


    def test_ksat_grover(self):
        problem = [[(2, True)], [(0, True),(1, False)]]

        qc, qc_oracle = create_ksat_grover(problem, 1)
        qc.remove_final_measurements()

        # Reapply Marking Oracle for correctness check
        num_vars = len(set([statement[0] for clause in problem for statement in clause]))
        num_clauses = len(problem)

        qreg_out = QuantumRegister(size=1, name="q_out")
        qc.add_register(qreg_out)
        # Oracle over vars and ancillas but on clean output qubit
        # In the example they do this AFTER measuring,
        # but we omit measuring here because we can test the circuit more easily then
        marking_out_map = list(range(num_vars))
        marking_out_map.extend(list(range(num_vars+1, num_vars+1+num_clauses+1)))
        qc = qc.compose(qc_oracle, marking_out_map)

        # Simulate
        transpiled_qc = transpile(qc, self.test_backend)
        results = self.test_backend.run(transpiled_qc).result()
        counts = results.get_counts()

        expectedCounts = {
            # Little endian: a2, a1, out, tar, in3, in2, in1
            # We ignore the ancillas so just set them to 0
            # Amplify state of solutions to problem: if (in3) and (in1 or not(in2)) -> '100', '101' and '111'
            '0000000': 0.03125, 
            '0000001': 0.03125, 
            '0000010': 0.03125, 
            '0000011': 0.03125, 
            '0010100': 0.28125, # Correctly amplified and marked as correct solutions by last oracle
            '0010101': 0.28125, # Correctly amplified and marked as correct solutions by last oracle
            '0000110': 0.03125, 
            '0010111': 0.28125  # Correctly amplified and marked as correct solutions by last oracle
        }

        self.assertEqual(counts, expectedCounts)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)