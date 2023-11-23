from typing import Dict, List, Tuple

import numpy as np
import os

from sympy import Basic

from configproblem.util.feature import FeatureType
from configproblem.util.xml_reader_feature import Extended_Modelreader
from configproblem.util.qasm3 import QASM3

from configproblem.fragments.quantum_states_qasm import add_all_hadamards


np.set_printoptions(threshold=1e6)

Qubit = Tuple[str, int]


def add_all_hadamards(qasm: QASM3, qubits: List[Qubit]):
    for q in qubits:
        qasm.add_command("h", tar=q)


def create_not_oracle(qasm: QASM3, qubits: List[Qubit]):
    for q in qubits:
        qasm.add_command("x", tar=q)


def create_and_oracle(qasm: QASM3, tar: Qubit, ctrl: List[Qubit] = None, negctrl: List[Qubit] = None):
    qasm.add_command("x", tar=tar, controls=ctrl, negative_controls=negctrl)


def create_or_oracle(qasm: QASM3,  tar: Qubit, ctrl: List[Qubit] = None, negctrl: List[Qubit] = None):
    create_and_oracle(qasm, ctrl=negctrl, negctrl=ctrl, tar=tar)
    create_not_oracle(qasm, qubits=[tar])


def create_root_feature(qasm: QASM3, root: Qubit, target: Qubit):
    qasm.add_command("x", tar=target, controls=[root])


def create_mandatory_sub(qasm: QASM3, feature: Qubit, sub_feature: Qubit, target: Qubit):
    qasm.add_command("x", tar=target, controls=[feature])
    qasm.add_command("x", tar=target, controls=[sub_feature])
    qasm.add_command("x", tar=target)


def create_optional_sub(qasm: QASM3, feature: Qubit, sub_feature: Qubit, target: Qubit):
    create_or_oracle(qasm, tar=target, ctrl=[feature], negctrl=[sub_feature])


def create_or_subs(qasm: QASM3, feature: Qubit, sub_features: List[Qubit], target: Qubit):
    create_and_oracle(qasm, tar=target, negctrl=sub_features)
    qasm.add_command("x", tar=target, controls=[feature])


def create_xor_subs(qasm: QASM3, feature: Qubit, sub_features: [Qubit], target: Qubit, ancilla: Qubit):
    create_or_subs(qasm, feature, sub_features, ancilla)
    for i in range(len(sub_features)):
        create_and_oracle(qasm, tar=target, ctrl=[sub_features[i]], negctrl=sub_features[:i]+sub_features[i+1:])


def create_equivalence(qasm: QASM3, feature_1: Qubit, feature_2: Qubit, target: Qubit):
    qasm.add_command("x", tar=target, controls=[feature_1])
    qasm.add_command("x", tar=target, controls=[feature_2])
    qasm.add_command("x", tar=target)


def create_implication(qasm: QASM3, feature: Qubit, requirement: Qubit, target: Qubit):
    create_or_oracle(qasm, tar=target, ctrl=[requirement], negctrl=[feature])


def create_exclusion(qasm: QASM3, feature: Qubit, exclusion: Qubit, target: Qubit):
    create_and_oracle(qasm, ctrl=[feature, exclusion], tar=target)
    qasm.add_command("x", tar=target)


def oracle_converter(qasm: QASM3, tar: Qubit):
    commands = [("x", tar, [], []), ("h", tar, [], [])]
    qasm.add_commands(commands, prepend=True)
    qasm.add_commands(commands[::-1])


def feature_oracle(feature_model_path: str) -> Tuple[QASM3, List[Qubit], Dict[str, int]]:
    # load feature model
    feature_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), feature_model_path)
    if feature_model_path.split(".")[-1] != "xml":
        print("Error can only parse xml files")
        raise Exception("Can only parse xml files")
    reader = Extended_Modelreader()
    model, constraints = reader.readModel(feature_model_path)

    features = model.get_all_features()
    feature_names = {x.name: i for (i, x) in enumerate(features)}
    feature_count = len(feature_names)
    ancilla_count = 0

    # Create initialize qasm objects
    oracle = QASM3()
    oracle.add_qubit("feat_reg", feature_count)
    oracle.add_bit("feat_bit_reg", feature_count)
    feature_qubits = [("feat_reg", i) for i in range(feature_count)]
    oracle.add_qubit("tar_reg", 1)

    # Parse XML and apply necessary quantum operations on circuit
    for feature in features:
        match feature.type:
            case FeatureType.AND:
                for child in feature.get_children():
                    if child.mandatory:
                        create_mandatory_sub(oracle,
                                             feature=("feat_reg", feature_names[feature.name]),
                                             sub_feature=("feat_reg", feature_names[child.name]),
                                             target=("anc_reg", ancilla_count))
                    else:
                        create_optional_sub(oracle,
                                            feature=("feat_reg", feature_names[feature.name]),
                                            sub_feature=("feat_reg", feature_names[child.name]),
                                            target=("anc_reg", ancilla_count))
                    ancilla_count += 1
            case FeatureType.OR:
                children_qubits = [("feat_reg", feature_names[c.name]) for c in feature.get_children()]
                create_or_subs(oracle, feature=("feat_reg", feature_names[feature.name]),
                               sub_features=children_qubits,
                               target=("anc_reg", ancilla_count))
                ancilla_count += 1
            case FeatureType.ALT:
                children_qubits = [("feat_reg", feature_names[c.name]) for c in feature.get_children()]
                create_xor_subs(oracle, feature=("feat_reg", feature_names[feature.name]),
                                sub_features=children_qubits,
                                target=("anc_reg", ancilla_count),
                                ancilla=("anc_reg", ancilla_count + 1))
                ancilla_count += 2

    # Parse and add constraints
    # parse_constraints(oracle, constraints, feature_qubits, feature_names, ancilla_count)
    pre_constraints_count = ancilla_count
    controls = []
    if constraints is not None:
        con_ancillas = []
        for c in constraints:
            ancilla_count, con = parse_constraints(oracle, c, feature_qubits, feature_names, ancilla_count)
            con_ancillas.append(con)
        con_ancilla = ("anc_reg", ancilla_count)
        controls.append(con_ancilla)
        create_and_oracle(oracle, tar=con_ancilla, ctrl=con_ancillas)
        ancilla_count += 1
    inverse_commands = oracle.commands[::-1]

    controls += [("anc_reg", i) for i in range(pre_constraints_count)]
    controls += [("feat_reg", feature_names[model.name])]
    create_and_oracle(oracle, tar=("tar_reg", 0), ctrl=controls)

    oracle.add_commands(inverse_commands)

    # add necessary count of ancilla qubits
    oracle.add_qubit("anc_reg", ancilla_count)

    # convert oracle to phase flipping oracle
    oracle_converter(oracle, ("tar_reg", 0))

    return oracle, feature_qubits, feature_names


def parse_constraints(qasm: QASM3, constraints,  feature_qubits: List[Qubit], feature_names: Dict[str, int], ancilla_count: int) -> Tuple[int, Qubit]:
    match constraints[0]:
        case "implies":
            feat, req = constraints[1], constraints[2]
            target = ("anc_reg", ancilla_count)
            ancilla_count += 1
            ancilla_count, feat_anc = parse_constraints(qasm, feat, feature_qubits, feature_names, ancilla_count)
            ancilla_count, req_anc = parse_constraints(qasm, req, feature_qubits, feature_names, ancilla_count)
            create_implication(qasm, target=target, feature=feat_anc, requirement=req_anc)
            return ancilla_count, target

        case "equivalent":
            l, r = constraints[1], constraints[2]
            target = ("anc_reg", ancilla_count)
            ancilla_count += 1
            ancilla_count, l_anc = parse_constraints(qasm, l, feature_qubits, feature_names, ancilla_count)
            ancilla_count, r_anc = parse_constraints(qasm, r, feature_qubits, feature_names, ancilla_count)
            create_equivalence(qasm, feature_1=l_anc, feature_2=r_anc, target=target)
            return ancilla_count, target

        case "and":
            values = list(constraints[1:])
            elements = []
            target = ("anc_reg", ancilla_count)
            ancilla_count += 1
            for c in values:
                if c[0] == "and":
                    values.append(c[1])
                    values.append((c[2]))
                else:
                    ancilla_count, e = parse_constraints(qasm, c, feature_qubits, feature_names, ancilla_count)
                    elements.append(e)
            create_and_oracle(qasm, tar=target, ctrl=elements)

            return ancilla_count, target

        case "or":
            values = list(constraints[1:])
            elements = []
            target = ("anc_reg", ancilla_count)
            ancilla_count += 1
            for c in values:
                if c[0] == "or":
                    values.append(c[1])
                    values.append((c[2]))
                else:
                    ancilla_count, e = parse_constraints(qasm, c, feature_qubits, feature_names, ancilla_count)
                    elements.append(e)
            create_or_oracle(qasm, target, ctrl=elements)
            return ancilla_count, target

        case "not":
            target = ("anc_reg", ancilla_count)
            ancilla_count += 1
            ancilla_count, qubit = parse_constraints(qasm, constraints[1], feature_qubits, feature_names, ancilla_count)
            qasm.add_command("x", target, controls=[qubit])
            qasm.add_command("x", target)
            return ancilla_count, target

        case "var":
            name = constraints[1]
            return ancilla_count, ("feat_reg", feature_names[name])

        case _:
            print("MISSING", constraints)
            return None


def add_diffusion_feature(qasm: QASM3, feature_qubits: List[Qubit]):
    add_all_hadamards(qasm, feature_qubits)
    create_not_oracle(qasm, [feature_qubits[-1]])
    qasm.add_command("z", negative_controls=feature_qubits[:-1], tar=feature_qubits[-1])
    create_not_oracle(qasm, [feature_qubits[-1]])
    add_all_hadamards(qasm, feature_qubits)


def init_feature_circuit(rel_path: str, k:int = 1) -> Tuple[QASM3, Dict[str, int]]:

    # Create oracle and add diffusion operator to get one iteration
    oracle, feature_qubits, feature_names = feature_oracle(rel_path)
    add_diffusion_feature(oracle, feature_qubits)

    # Create final circuit
    circuit = QASM3()
    circuit.qubit_reg = oracle.qubit_reg
    circuit.bit_reg = oracle.bit_reg

    add_all_hadamards(circuit, feature_qubits)
    for i in range(k):
        circuit.add_commands(oracle.commands)

    return circuit, feature_names
