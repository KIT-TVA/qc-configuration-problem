import unittest
import os 

from configproblem.util.sat_verifier import SatVerifier

class TestModelreader(unittest.TestCase):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    some_model_path = os.path.join(dir_path, '../../benchmarks/test.dimacs')
    verifier = SatVerifier(some_model_path)  

    def test_is_correct(self): 
        self.assertTrue(self.verifier.is_correct('000'))
        self.assertFalse(self.verifier.is_correct('010'))
        self.assertFalse(self.verifier.is_correct('001'))
        self.assertTrue(self.verifier.is_correct('011'))
        self.assertFalse(self.verifier.is_correct('100'))
        self.assertFalse(self.verifier.is_correct('110'))
        self.assertTrue(self.verifier.is_correct('101'))
        self.assertFalse(self.verifier.is_correct('111'))