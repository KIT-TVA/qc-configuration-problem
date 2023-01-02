from enum import Enum
import itertools
from .cnf import CNF, Clause, Symbol

from sympy.logic.boolalg import to_cnf
from sympy import And

class FeatureType(Enum):
    LEAF = 0,
    AND = 1,
    OR = 2,
    ALT = 3, #xor


class Feature:
    def __init__(self, name: str, type: FeatureType, mandatory: bool):
        self.name = ''.join(name.split())
        self.type = type
        self.mandatory = mandatory
        self.children = []
        self.attributes = {}
        self.parent = None

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def add_child(self, child):
        self.children.append(child)
        child.set_parent(self)

    def remove_child(self, child):
        self.children.remove(child)
        child.set_parent(None)

    def get_children(self):
        return self.children

    def set_parent(self, parent):
        self.parent = parent

    def get_parent(self):
        return self.parent

    def add_attribute(self, key, value):
        self.attributes[key] = value

    def remove_attribute(self, key):
        self.attributes.delete(key)

    def get_attributes(self):
        return self.attributes

    def set_mandatory(self, mandatory):
        self.mandatory = mandatory

    def get_mandatory(self):
        return self.mandatory

    def __str__(self) -> str:
        return self.build_str(self)

    def build_str(self, feature, depth=0):
        tick = "-"
        ret = f"{tick*depth} {feature.name}\n"
        for child in feature.get_children():
            ret += self.build_str(child, depth+1)
        return ret

    def count_features(self, feature=None, n=0) -> int:
        """ Recursively count all features """
        if feature is None:
            feature = self

        n = 1
        for child in feature.get_children():
            n += self.count_features(child, n)
        return n

    def get_all_features(self, feature=None, features=set([])):
        if feature is None:
            feature = self
        
        features.add(feature)
        for child in feature.get_children():
            self.get_all_features(child, features)

        return features

    def build_cnf(self, constraints=None) -> CNF:
        """ Construct a CNF based on the FeatureIDE code at:
            https://github.com/FeatureIDE/FeatureIDE/plugins/de.ovgu.featureide.fm.core/src/de/ovgu/featureide/fm/core/analysis/cnf/CNFCreator.java
        """
        root = self
        cnf = CNF()
        clause_root = Clause()
        clause_root.add_symbol(Symbol(root.get_name()))
        cnf.add_clause(clause_root)

        features = root.get_all_features()

        for feature in features:
            for child in feature.get_children():
                # Child implies parent
                clause = Clause()
                clause.add_symbols([Symbol(child.get_name(), negated=True), feature.get_name()])
                cnf.add_clause(clause)
            
            if len(feature.get_children()) > 0:
                if feature.type == FeatureType.AND:
                    # add clauses for mandatory children
                    for child in feature.get_children():
                        if child.get_mandatory():
                            clause = Clause()
                            clause.add_symbols([child.get_name(), Symbol(feature.get_name(), negated=True)])
                            cnf.add_clause(clause)

                elif feature.type == FeatureType.OR or FeatureType.ALT:
                    # add a clause containing all children
                    or_clause = Clause()
                    or_clause.add_symbols([c.get_name() for c in feature.get_children()])

                    # dependence on parent
                    or_clause.add_symbol(Symbol(feature.get_name(), negated=True))
                    cnf.add_clause(or_clause)

                    # additional clauses required for XOR/Alternatives
                    if feature.type == FeatureType.ALT:
                        symbols = set([Symbol(c.get_name(), negated=True) for c in feature.get_children()])
                        for pair in itertools.combinations(symbols, 2):
                            xor_clause = Clause()
                            xor_clause.add_symbols(pair)
                            cnf.add_clause(xor_clause)

        # non-structure constraints
        if constraints is not None:
            structure_sympy_cnf = cnf.to_sympy()
            joint_constraints = And(structure_sympy_cnf, constraints)
            joint_constraints = to_cnf(joint_constraints, simplify=True, force=True)
            cnf = cnf.from_sympy(joint_constraints)

        return cnf


    def boolean_representation(self, feature=None, constraints=[], queue=[]):
        """ Builds the boolean expression recursively from the @param feature downwards"""
        is_root = False
        if feature is None:
            is_root = True
            feature = self
            constraints.append(feature.name)  # root feature must be present

        expr = "("
        operator = ""
        if feature.type == FeatureType.AND:
            operator = "&"
        elif feature.type == FeatureType.ALT:
            operator = "⊕"  # xor
        elif feature.type == FeatureType.OR:
            operator = "|"
        elif feature.type == FeatureType.LEAF:
            return constraints # we can abort here 

        # feature is parent of grouped features g1,...gn ==> feature>>(g1 op ... op gn)
        if len(feature.get_children()) > 0:
            expr += feature.get_name() + ">>("

        for child in feature.get_children():
            queue.append(child)
            
            # mandatory subfeatures imply their parent
            if child.get_mandatory():
                constraints.append(f"{child.get_name()}>>{feature.get_name()}")
            
            # build the grouped feature expression
            if feature.type == FeatureType.AND:
                # only add mandatory features for and groups
                if child.get_mandatory():
                    expr += child.get_name() + operator
            else:
                expr += child.get_name() + operator

        else:
            # loop finished without breaks
            # remove trailing operator and close brackets
            expr = expr[:-1] + "))"

        constraints.append(expr)
        
        # work on the queue
        if is_root:
            for child in queue:
                self.boolean_representation(child, constraints, queue)

        return constraints
        

