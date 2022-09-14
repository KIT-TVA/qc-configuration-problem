# QISKIT USES LITTLE ENDIAN NOTATION!

from qiskit.circuit import Qubit, QuantumRegister, AncillaRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Operator

from typing import Dict, List, Tuple

def create_and_oracle(inp_reg: QuantumRegister, tar: Qubit):
    """
        Constructs an oracle for boolean AND,
        that is a multi-controlled X gate
    """
    tar_reg = QuantumRegister(bits=[tar])
    qc = QuantumCircuit(inp_reg, tar_reg)

    qc.mcx(inp_reg, tar_reg)

    return qc.to_gate(label="$U_{and}$")

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
    
    return qc.to_gate(label="$U_{or}$")

def get_clause_qubits(inp_reg: QuantumRegister, clause: List[Tuple[int, bool]]) -> QuantumRegister:
    """
        Return a register containing only relevant qubits for a SAT clause
    """

    clause_qubits:list(Qubit) = []

    for index, _ in clause:
        clause_qubits.append(inp_reg[index])

    return clause_qubits

def create_clause_oracle(inp_reg: QuantumRegister, tar: Qubit, clause: List[Tuple[int, bool]]):
    """
        Create an oracle for a SAT clause
    """
    tar_reg = QuantumRegister(bits=[tar])
    qc = QuantumCircuit(inp_reg, tar_reg)

    # Flip all qubits which are negated in the clause
    for index, positive in clause:
        if not positive:
            qc.x(index)
    
    
    # Get Clause Qubits
    clause_qubits = get_clause_qubits(inp_reg, clause)
    clause_reg = QuantumRegister(bits=clause_qubits)

    # Create an OR oracle for clause
    clause_oracle = create_or_oracle(clause_reg, tar)
    qc.append(clause_oracle, clause_reg[:]+tar_reg[:])

    # Inverse the initial flips
    for index, positive in clause:
        if not positive:
            qc.x(index)

    return qc.to_gate(label="$U_{clause}$")

def create_ksat_oracle(inp_reg: QuantumRegister, tar: Qubit, clauses: List[List[Tuple[int, bool]]]):
    ancilla_reg = AncillaRegister(len(clauses))
    tar_reg = QuantumRegister(bits=[tar])
    qc = QuantumCircuit(inp_reg, tar_reg, ancilla_reg)

    # Compose individual clauses
    for index, clause in enumerate(clauses):
        # Use one ancilla for each clause
        clause_oracle = create_clause_oracle(inp_reg, ancilla_reg[index], clause)
        qc.append(clause_oracle, inp_reg[:]+[ancilla_reg[index]])
    
    # Store the conjugate transpose (inverse) for later qubit cleanup
    inverse_qc = qc.inverse()
    
    # Use and oracle onto ancilla register and target
    and_oracle = create_and_oracle(ancilla_reg, tar)
    qc.append(and_oracle, ancilla_reg[:]+tar_reg[:])

    # Inverse clause oracles
    qc = qc.compose(inverse_qc)

    return qc.to_gate(label="$U_{ksat}$")

def oracle_converter(oracle_qc, target_idx):
    phase_qc = oracle_qc.copy()

    qc_conv = QuantumCircuit(1, name="phase")
    qc_conv.x(0)
    qc_conv.h(0)
    # Prepend the phase transformation
    phase_qc = phase_qc.compose(qc_conv, qubits=[target_idx], front=True)
    # Append the phase transformation 
    phase_qc = phase_qc.compose(qc_conv.inverse(), qubits=[target_idx])

    return phase_qc

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
            # Little endian: t, c2, c1
            # Flip t if c1 or not(c2)
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
            # Little endian: t, c3, c2, c1
            # flip t if c1 or not(c2) or not(c3)
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
            # Little endian: a2, a1 t, c3, c2, c1
            # We ignore the ancillas so just set them to 0
            # Flip t if (c3) and (c1 or not(c2))
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

    def test_grover(self):
        problem = [[(2, True)], [(0, True),(1, False)]]

        # TODO register sizes could be calculated from problem -> extract function
        inp_reg3 = QuantumRegister(3)
        tar = Qubit()
        tar_reg = QuantumRegister(bits=[tar])
        ancilla_reg2 = AncillaRegister(2)

        problem = [[(2, True)], [(0, True),(1, False)]]
        qc_oracle = QuantumCircuit(6)
        qc_oracle.append(create_ksat_oracle(inp_reg3, tar, problem), qc_oracle.qubits)
        phase_qc = oracle_converter(qc_oracle, len(inp_reg3))

        num_var = phase_qc.num_qubits-phase_qc.num_ancillas
        from grover import diffuser
        diff = diffuser(num_var-1)

        # Construct main quantum circuit
        main_qc = phase_qc.copy_empty_like()
        creg = ClassicalRegister(num_var, name="c")
        main_qc.add_register(creg)

        # Create uniform superposition
        from grover import initialize_s
        main_qc = initialize_s(main_qc, range(num_var-1))

        # Grover loop
        # TODO find k
        k=1

        # Add the oracle-diffusor step k times
        for i in range(k):
            main_qc.append(phase_qc.to_gate(), range(main_qc.num_qubits))
            register_map = list(range(num_var-1))
            main_qc = main_qc.compose(diff, register_map)
    
        # Measure Inputs and Target
        for i, c in enumerate(creg):
            if i < len(creg)-1:
                main_qc.measure(i, c) # measure inputs

        # Reapply Marking Oracle for correctness check
        qreg_out = QuantumRegister(size=1, name="q_out")
        main_qc.add_register(qreg_out)
        marking_out_map = list(range(num_var+main_qc.num_ancillas-1))
        marking_out_map.append(-1)
        main_qc = main_qc.compose(qc_oracle, marking_out_map)
        main_qc.measure(-1, -1) # measure q_out into last classical register

        print('\n')
        print(main_qc.draw(output='text'))

        from qiskit import Aer, transpile
        from qiskit.visualization import plot_histogram

        qasm_sim = Aer.get_backend('qasm_simulator')
        transpiled_grover_circuit = transpile(main_qc, qasm_sim)
        results = qasm_sim.run(transpiled_grover_circuit).result()
        counts = results.get_counts()
        histogram = plot_histogram(counts)
        import matplotlib.pyplot as plt
        plt.show()

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)