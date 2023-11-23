"""
Read e-xml (Parser, Stringtable for keys) into dict
Generate CNF from dict / read from dimacs
Create QUBO using qubovert from CNF
Create cost hamiltonian from attributes in dict
Combine hamiltonian models with qubovert

Output the hamiltonian.
"""
from enum import Enum
import xml.etree.ElementTree as ET
from typing import Tuple

from .cnf import Symbol
from .feature import Feature, FeatureType

from sympy.logic import simplify_logic
from sympy.logic.boolalg import to_cnf, to_nnf, BooleanFalse, BooleanTrue
from sympy import And, Or, Not, Implies, Equivalent, Basic
from sympy import Symbol as SympySymbol

class XMLT(str, Enum):
    """ Holds the relevant xml tags for extended featuremodel files"""
    EMF = "extendedFeatureModel"
    STRUCT = "struct"
    AND = "and"
    OR = "or"
    ALT = "alt"
    FEAT = "feature"
    ATTR = "attribute"

    CONSTR = "constraints"
    RULE = "rule"
    IMP = "imp"
    DISJ = "disj"
    CONJ = "conj"
    VAR = "var"
    NOT = "not"
    EQ = "eq"


class XMLA(str, Enum):
    """ Holds the relevant xml attribute keys for extended featuremodel files"""
    NAME = "name"
    VAL = "value"
    MAND = "mandatory"


class Extended_Modelreader:
    """ Used to read extended featuremodel xml files into its own datastructure """
    # List of relevant xml tags

    def readModel(self, exml_path) -> Tuple[Feature,  And | None | Symbol | Not | Implies | BooleanFalse | BooleanTrue | Basic | Or]:
        with open(exml_path) as f:
            model_xml = ET.parse(f)

        # ======================
        # get boolean expression
        # ======================
        structure = model_xml.getroot().find(XMLT.STRUCT)
        feature_model = self.traverse_structure(structure)

        constraints_xml = model_xml.getroot().find(XMLT.CONSTR)
        if constraints_xml is not None:
            constraints = self.traverse_constraints(constraints_xml)
            return feature_model, constraints
        
        # get feature attributes
        
        return feature_model, None
    

    def traverse_structure(self, xml, parent=None) -> Feature:
        """parent = parent feature group"""
        for child in xml:
            if child.tag in [XMLT.ALT, XMLT.AND, XMLT.OR, XMLT.FEAT]:
                # some kind of feature, determine characteristics
                fname = child.attrib.get(XMLA.NAME)
                fmand = child.attrib.get(XMLA.MAND)
                if child.tag == XMLT.FEAT:
                    ftype = FeatureType.LEAF
                elif child.tag == XMLT.ALT:
                    ftype = FeatureType.ALT
                elif child.tag == XMLT.AND:
                    ftype = FeatureType.AND
                elif child.tag == XMLT.OR:
                    ftype = FeatureType.OR
                else:
                    raise RuntimeError("Unexpected Feature Tag: " + str(child))
                
                # add to data structure
                feature = Feature(fname, ftype, fmand)
                if parent is None:
                    parent = feature
                else:
                    parent.add_child(feature)

                # traverse features downwards
                self.traverse_structure(child, feature)

            elif child.tag == XMLT.ATTR:
                # feature attribute
                if parent is not None:
                    aname = child.attrib.get(XMLA.NAME)
                    avalue = child.attrib.get(XMLA.VAL)
                    if aname is None or avalue is None:
                        pass # attribute has no name/value and is therefore not interesting
                    else:
                        parent.add_attribute(aname, avalue)

        return parent


    constraints_tags_1 = [XMLT.NOT, XMLT.VAR, XMLT.RULE]
    constraints_tags_2 = [XMLT.IMP, XMLT.DISJ, XMLT.CONJ, XMLT.EQ]

    def traverse_constraints(self, xml):
        """Parse a sympy expression from a feature model constraint xml"""
        # special case of root feature
        if xml.tag == XMLT.CONSTR:
            # prepare for list of rules
            rules = [] 
            for child in xml:
                if child.tag == XMLT.RULE:
                    # rules are implicitly conjunct
                    rules.append(self.traverse_constraints(child))

            return rules

        if xml.tag in self.constraints_tags_1:
            # singles
            return self.handle_single_constraints(xml)

        elif xml.tag in self.constraints_tags_2:
            # doubles
            return self.handle_double_constraints(xml)
                

    def handle_single_constraints(self, xml):
        """Descend into the only child if it even exists otherwise create a leaf symbol"""
        if xml.tag == XMLT.VAR:
            # A single Symbol, leaf of the tree
            stripped_symbol_name = ''.join(xml.text.split())
            return "var", stripped_symbol_name

        elif xml.tag == XMLT.NOT:
            # Negation of the following expression structure
            return "not", self.traverse_constraints(xml[0])

        elif xml.tag == XMLT.RULE:
            # rules are treated specially in the beginning, now just pass on
            return self.traverse_constraints(xml[0])

        else:
            raise RuntimeError(f"Encountered unexpected XML tag in constraints: {xml.tag}")
  

    def handle_double_constraints(self, xml):
        """Descends into the constraint structure in an Left-Right-Operate fashion"""
        if len(xml) == 2:
            left_hand_side = self.traverse_constraints(xml[0])
            right_hand_side = self.traverse_constraints(xml[1])

            if xml.tag == XMLT.IMP:
                return "implies", left_hand_side, right_hand_side
            elif xml.tag == XMLT.EQ:
                return "equivalent", left_hand_side, right_hand_side
            elif xml.tag == XMLT.CONJ:
                return "and", left_hand_side, right_hand_side
            elif xml.tag == XMLT.DISJ:
                return "or", left_hand_side, right_hand_side
            else:
                raise RuntimeError(f"Encountered unexpected XML tag in constraints: {xml.tag}")
        elif len(xml) == 1:
            left_hand_side = self.traverse_constraints(xml[0])
            if xml.tag == XMLT.CONJ:
                return left_hand_side
            elif xml.tag == XMLT.DISJ:
                return left_hand_side
            else:
                raise RuntimeError(f"Encountered invalid XML tag length in constraints for: {xml.tag}")
        else:
            raise RuntimeError(f"Encountered invalid XML tag length in constraints: {xml.tag}")


