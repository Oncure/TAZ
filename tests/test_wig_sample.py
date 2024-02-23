import sys
sys.path.append('../TAZ')
import TAZ
from .utils import chi2_test

import numpy as np

import unittest

class TestBayesSampler(unittest.TestCase):
    """
    The purpose of this test is to verify that the WigSample algorithm is working correctly. This
    will be verified with distribution analysis, including chi-squared goodness of fit on the
    level-spacing distribution.
    """

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters:
        cls.EB = (1e-5,5000)
        cls.false_dens = 1/8.0
        cls.lvl_dens  = [1/4.3166, 1/4.3166]
        cls.gn2m  = [44.11355, 33.38697]
        cls.gg2m   = [55.00000, 55.00000]
        cls.dfn   = [1, 1]
        cls.dfg   = [250, 250]
        cls.l     = [0, 0]
        cls.j     = [3.0, 4.0]

        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile, lvl_dens=cls.lvl_dens, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB, false_dens=cls.false_dens)
        cls.res_ladder = cls.reaction.sample(cls.ensemble)[0]

    def test_distributions(self):
        """
        ...
        """
        NUM_BINS = 40
        E  = self.res_ladder.E
        self.reaction.distributions(dist_type='Wigner')
        raise NotImplementedError('...')
    
    def test_level_densities(self):
        """
        ...
        """
        E  = self.res_ladder.E
        self.reaction.distributions(dist_type='Wigner')
        raise NotImplementedError('...')
    
if __name__ == '__main__':
    unittest.main()