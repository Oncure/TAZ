import sys
sys.path.append('../TAZ')
import TAZ
from TAZ.analysis import correlate_probabilities

import numpy as np

import unittest

class TestBayesSampler2SG(unittest.TestCase):
    """
    The purpose of this test is to verify that the WigBayes algorithm is working correctly. This
    will be verified with cross-case verification and special cases with known results.
    """

    ensemble = 'NNE' # Nearest Neighbor Ensemble
    err = 1e-8
    num_groups = 2

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters:
        cls.EB = (1e-5,1000)
        cls.false_dens = 1/15.0
        cls.lvl_dens  = [1/4.3166, 1/4.3166]
        cls.gn2m  = [44.11355, 33.38697]
        cls.gg2m   = [55.00000, 55.00000]
        cls.dfn   = [1, 1]
        cls.dfg   = [250, 250]
        cls.l     = [0, 0]
        cls.j     = [3.0, 4.0]

        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile, lvl_dens=cls.lvl_dens, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB, false_dens=cls.false_dens)
        cls.res_ladder, cls.true_assignments, _, _ = cls.reaction.sample(cls.ensemble)
    
    def test_poisson(self):
        """
        Here, we intend to verify that WigBayes returns the prior when provided Poisson
        distributions.
        """
        E  = self.res_ladder.E
        prior, log_likelihood_prior = TAZ.PTBayes(self.res_ladder, self.reaction)
        distributions = self.reaction.distributions(dist_type='Poisson')
        runmaster = TAZ.RunMaster(E, self.EB, distributions, self.false_dens, prior, log_likelihood_prior, err=self.err)
        posterior = runmaster.WigBayes()
        
        perrors = abs(posterior - prior) / prior
        perror_max = np.max(perrors)
        perror_mean = np.mean(perrors)
        self.assertTrue(np.allclose(prior, posterior, rtol=1e-6, atol=1e-15), f"""
The prior and posterior were not the same when Poisson distribution was used.
Maximum error = {perror_max:.6%}
Mean error    = {perror_mean:.6%}
""")
    
    def test_probability_frequency(self):
        """
        Here, we intend to verify that WigBayes returns probabilities that match the fraction
        of resonances with said probability within statistical error.
        """
        # self.skipTest('Not implemented')
        E  = self.res_ladder.E
        prior, log_likelihood_prior = TAZ.PTBayes(self.res_ladder, self.reaction)
        distributions = self.reaction.distributions(dist_type='Wigner')
        runmaster = TAZ.RunMaster(E, self.EB, distributions, self.false_dens, prior, log_likelihood_prior, err=self.err)
        posterior = runmaster.WigBayes()

        prob_expected, prob_ans_cor_est, prob_ans_cor_std = correlate_probabilities(posterior, self.true_assignments)
        for g in range(self.num_groups):
            errors = abs(prob_expected[g] - prob_ans_cor_est[g]) / prob_ans_cor_std[g]

            stderr = 3.5
            self.assertTrue(np.all(errors < stderr), f"""
WigBayes probabilities do not match the frequency of correct sampling to within {stderr} standard deviations.
Maximum discrepancy = {np.max(errors):.5f} standard deviations.
""")

class TestBayesSampler1or2SG(unittest.TestCase):

    ensemble = 'NNE' # Nearest Neighbor Ensemble
    eps = 0.005
    err = 1e-8

    def test_1_2_sg_match(self):
        """
        Here, we intend to verify that the 1-spingroup Encore algorithms converges to the
        2-spingroup Encore algorithms when the one spingroup is very infrequent (i.e. low
        level-density).
        """
        self.skipTest('Not implemented')
        E  = self.res_ladder.E
        prior, log_likelihood_prior = TAZ.PTBayes(self.res_ladder, self.reaction)
        distributions = self.reaction.distributions(dist_type='Wigner')
        runmaster = TAZ.RunMaster(E, self.EB, distributions, self.false_dens, prior, log_likelihood_prior, err=self.err)

class TestBayesSampler2or3SG(unittest.TestCase):

    ensemble = 'NNE' # Nearest Neighbor Ensemble
    eps = 0.005
    err = 1e-8

    def test_2_3_sg_match(self):
        """
        Here, we intend to verify that the 2-spingroup Encore algorithms converges to the
        3-spingroup Encore algorithms when the one spingroup is very infrequent (i.e. low
        level-density).
        """
        self.skipTest('Not implemented')
        E  = self.res_ladder.E
        prior, log_likelihood_prior = TAZ.PTBayes(self.res_ladder, self.reaction)
        distributions = self.reaction.distributions(dist_type='Wigner')
        runmaster = TAZ.RunMaster(E, self.EB, distributions, self.false_dens, prior, log_likelihood_prior, err=self.err)
    
if __name__ == '__main__':
    unittest.main()