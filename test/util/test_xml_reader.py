import unittest
import os 

from configproblem.util.xml_reader import Extended_Modelreader

class TestModelreader(unittest.TestCase):

    reader = Extended_Modelreader()

    def test_read_model(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        some_model_path = os.path.join(dir_path, "../../benchmarks/featureide-examples/pc-config.xml")
        
        feature_model, constraints_cnf = self.reader.readModel(some_model_path)
        self.assertEqual(feature_model.count_features(), 377)