import unittest
from typing import Dict

from qiskit import QuantumCircuit

import configproblem.grover_feature_qasm as gfq
import configproblem.grover_sat_qasm_new as gsq
import configproblem.grover_sat as gs

from qiskit_qasm3_import import parse
from qiskit.quantum_info import Statevector



class MyTestCase(unittest.TestCase):
    gates = ["$U_{ksat}$", "U$_{oracle}$", "U$_{diffuser}$", "$U_{clause}$", "$U_{and}$", "$U_{or}$"] + [
        f"circuit-{i}_dg" for i in range(200)]

    def align_qubits(self, counts: Dict[str, int], from_dic: Dict[str, int], to_dic: Dict[str, int]):
        l: Dict[int, int] = {}
        for name, index in from_dic.items():
            l[index] = to_dic[name]
        r: Dict[str, int] = {}
        print(l)
        for key in counts.keys():
            new = ["" for _ in range(len(key))]
            for i in range(len(key)):
                new[len(key) - 1 - l[i]] = key[len(key) - 1 - i]
            r["".join(new)] = counts[key]

        return r




    def _compare_(self, qc: QuantumCircuit, qc_new: QuantumCircuit):
        print(qc.decompose(self.gates, reps=5).draw(fold=-1))
        print(qc_new.decompose(self.gates, reps=5).draw(fold=-1))
        return Statevector.from_instruction(qc).equiv(Statevector.from_instruction(qc_new))

    def test_oracle_from_model(self):
        feature_oracle, _ = gfq.feature_oracle("../benchmarks/featureide-examples/small-sandwich.xml")
        feature_oracle_parsed = parse(feature_oracle.to_qasm())
        print(feature_oracle_parsed.qasm())

        cnf_oracle = gs.create_grover_for_model("../benchmarks/featureide-examples/small-sandwich.xml")
        cnf_oracle_parsed = parse(cnf_oracle.to_qasm())
        print(cnf_oracle_parsed.draw())

    def test(self):
        c = gs.create_grover_for_model("../benchmarks/test.dimacs")
        print(c.decompose(gates_to_decompose=self.gates, reps=5).draw())
#
    def test_big(self):
        model = "../benchmarks/featureide-examples/axTLS.dimacs"
        circ, _ = gs.create_grover_for_model(model)
        print(circ.decompose(gates_to_decompose=self.gates, reps=5).depth())

    def test_circuit_from_model(self):
        model = "../benchmarks/featureide-examples/car.xml"
        simulate = True
        shots = 10000
        k = 1
        feature_circuit, feature_assignment = gfq.init_feature_circuit(model, k=k)
        # non_feature_circuit, non_feature_assignment = gsq.create_grover_for_model(model, k=k)
        # old_circuit, old_assignment = gs.create_grover_for_model(model, k=k)

        feature_parsed = parse(feature_circuit.to_qasm())
        # non_feature_parsed = parse(non_feature_circuit.to_qasm())

        feature_results = gs.collect_circuit_info(feature_parsed, simulate=simulate, shots=shots)
        # non_feature_results = gs.collect_circuit_info(non_feature_parsed, simulate=simulate, shots=shots)
        # old_results = gs.collect_circuit_info(old_circuit, simulate=simulate, shots=shots)

        print(feature_results)
        # print("CNF:", non_feature_assignment)
        # print("Old: ", old_assignment)
        # print("----")
        # print(feature_results)
        # print(max(feature_circuit.calculate_depth().values()))
        # print("----")
        # print(non_feature_results)
        # print(max(non_feature_circuit.calculate_depth().values()))
        # print("----")
        # print(old_results)
        # print(old_circuit.decompose(gates_to_decompose=self.gates, reps=5).depth())
        # print("----")
        # print(self.align_qubits(old_results["counts"], old_assignment, feature_assignment))

        # self.assertTrue(self._compare_(feature_circuit_parsed, old_circuit))


