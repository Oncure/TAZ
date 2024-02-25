import sys
sys.path.append('../TAZ')
from TAZ.Theory.RMatrix import Rho, PenetrationFactor

import numpy as np

import unittest

__doc__ == """
This file tests the general theory equations.
"""
        
class TestGeneralTheory(unittest.TestCase):

    def test_penetration(self):
        """
        This file tests the penetration factor equations by relating the iterative representation
        with the explicit representation of the penetration factor.
        """

        self.skipTest('Not implemented yet.')

if __name__ == '__main__':
    unittest.main()