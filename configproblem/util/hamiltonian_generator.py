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
from .Feature import Feature, FeatureType


class XMLT(str, Enum):
    """ Holds the relevant xml tags for extended featuremodel files"""
    EMF = "extendedFeatureModel"
    STRUCT = "struct"
    AND = "and"
    OR = "or"
    ALT = "alt"
    FEAT = "feature"
    ATTR = "attribute"


class XMLA(str, Enum):
    """ Holds the relevant xml attribute keys for extended featuremodel files"""
    NAME = "name"
    VAL = "value"
    MAND = "mandatory"


class Extended_Modelreader:
    """ Used to read extended featuremodel xml files into its own datastructure """
    # List of relevant xml tags

    def readModel(self, exml_path):
        with open(exml_path) as f:
            model_xml = ET.parse(f)

        # ======================
        # get boolean expression
        # ======================
        structure = model_xml.getroot().find(XMLT.STRUCT)
        feature_model = self.traverse_structure(structure)

        # print(feature_model)
        # print(feature_model.count_features())
        # cnf = feature_model.build_cnf()
        # print(cnf.to_qubo(debug=True))
        
        # get feature attributes
        
        return feature_model

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



if __name__ == "__main__":
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    some_model_path = os.path.join(dir_path, "../../benchmarks/sandwich.xml")
    reader = Extended_Modelreader()
    reader.readModel(some_model_path)