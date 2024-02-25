import sys
sys.path.append('../TAZ')
import TAZ

import numpy as np

import unittest

class TestBayesSampler(unittest.TestCase):
    """
    The purpose of this test is to verify that the WigMaxLikelihood algorithm is working correctly.
    """

    ensemble = 'NNE' # Nearest Neighbor Ensemble
    err = 1e-8

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters
        cls.EB = (1e-5, 500)
        cls.false_dens = 1/20.0
        cls.lvl_dens  = [1/5.0, 1/5.0]
        cls.gn2m  = [40, 70]
        cls.gg2m   = [55.00000, 55.00000]
        cls.dfn   = [1, 1]
        cls.dfg   = [250, 250]
        cls.l     = [0, 0]
        cls.j     = [3.0, 4.0]

        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile, lvl_dens=cls.lvl_dens, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB, false_dens=cls.false_dens)
        cls.res_ladder = cls.reaction.sample(cls.ensemble)[0]

    def test_poisson(self):
        """
        Test that WigMaxLikelihood returns the spingroups with the maximum prior probabilities
        when provided Poisson level-spacing distributions.
        """
        E  = self.res_ladder.E
        prior, log_likelihood_prior = TAZ.PTBayes(self.res_ladder, self.reaction)
        most_likely_sample_prior = np.argmax(prior, axis=1)

        distributions = self.reaction.distributions(dist_type='Poisson')
        most_likely_sample_posterior = TAZ.RunMaster.WigMaxLikelihood(E, self.EB, distributions, self.err, prior)

        self.assertTrue(np.all(most_likely_sample_prior == most_likely_sample_posterior), """
The prior and posterior samples do not match with Poisson spacing distributions.
""")
    
if __name__ == '__main__':
    unittest.main()