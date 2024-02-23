import sys
sys.path.append('../TAZ')
import TAZ

import numpy as np

import unittest

class TestBayesSampler(unittest.TestCase):
    """
    The purpose of this test is to verify that the WigBayes algorithm is working correctly. This
    will be verified with cross-case verification and special cases with known results.
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

    def test_1_2_sg_match(self):
        """
        Here, we intend to verify that the 1-spingroup Encore algorithms converges to the
        2-spingroup Encore algorithms when the one spingroup is very infrequent (i.e. low
        level-density).
        """
        eps = 0.005
        distributions = self.reaction.distributions(dist_type='Wigner')
        E  = self.res_ladder.E
        raise NotImplementedError('...')
    
    def test_2_3_sg_match(self):
        """
        Here, we intend to verify that the 2-spingroup Encore algorithms converges to the
        3-spingroup Encore algorithms when the one spingroup is very infrequent (i.e. low
        level-density).
        """
        eps = 0.005
        distributions = self.reaction.distributions(dist_type='Wigner')
        E  = self.res_ladder.E
        raise NotImplementedError('...')
    
    def test_poisson(self):
        """
        Here, we intend to verify that WigBayes returns the prior when provided Poisson
        distributions.
        """
        distributions = self.reaction.distributions(dist_type='Poisson')
        E  = self.res_ladder.E
        raise NotImplementedError('...')
    
    def test_probability_frequency(self):
        """
        Here, we intend to verify that WigBayes returns probabilities that match the fraction
        of resonances with said probability within statistical error.
        """
        distributions = self.reaction.distributions(dist_type='Wigner')
        E  = self.res_ladder.E
        raise NotImplementedError('...')
    
if __name__ == '__main__':
    unittest.main()