import sys
sys.path.append('../TAZ')
import TAZ
from TAZ.Theory.distributions import wigner_dist, lvl_spacing_ratio_dist, porter_thomas_dist, deltaMehta3, deltaMehtaPredict
from TAZ.Theory.WidthDists import ReduceFactor

import numpy as np
import pandas as pd
from scipy.stats import chisquare

import unittest

__doc__ == """
This file tests resonance sampling using level-spacing distributions (Wigner distribution),
level-spacing ratio distributions, Dyson-Mehta Delta-3 statistic, and reduced width distributions
(Porter-Thomas distribution).
"""

def chi2_test(dist, data, num_bins:int):
    """
    Performs a Pearson's Chi-squared test with the provided distribution and data.
    """

    data_len = len(data)
    quantiles = np.linspace(0.0, 1.0, num_bins+1)
    with np.errstate(divide='ignore'):
        edges = dist.ppf(quantiles)
    obs_counts, edges = np.histogram(data, edges)
    exp_counts = (data_len / num_bins) * np.ones((num_bins,))
    chi2, p = chisquare(f_obs=obs_counts, f_exp=exp_counts)
    chi2_bar = chi2 / num_bins
    return chi2_bar, p
        
class TestResonanceGeneration(unittest.TestCase):
    E0 = 200 # eV
    num_res = 10_000 # resonances
    ensemble = 'NNE' # Nearest Neighbor Ensemble

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters:
        cls.EB   = (1e-5,5000)
        cls.mls  = [4.3166]
        cls.gn2m = [441.1355]
        cls.gg2m = [55.00000]
        cls.dfn  = [1]
        cls.dfg  = [250]
        cls.l    = [0]
        cls.j    = [3.0]

        # 2 Spingroup Case:
        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile, MLS=cls.mls, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB)
        cls.res_ladder = cls.reaction.sample('NNE')[0]

    def test_wigner(self):
        """
        Tests if the resonance ladder follows Wigner distribution.
        """

        NUM_BINS = 40
        E = self.res_ladder.E
        lvl_spacing = np.diff(E)
        dist = wigner_dist(scale=self.mls[0], beta=1)
        chi2_bar, p = chi2_test(dist, lvl_spacing, NUM_BINS)
        self.assertGreater(p, 0.001, f"""
The NNE level-spacings do not follow Wigner distribution according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
    def test_gamma_widths(self):
        """
        Tests if gamma widths follow the expected Chi-squared distribution.
        """

        NUM_BINS = 40
        Gg = self.res_ladder.Gg
        gg2 = Gg
        dist = porter_thomas_dist(mean=self.gg2m[0], df=self.dfg[0], trunc=0.0)
        chi2_bar, p = chi2_test(dist, gg2, NUM_BINS)
        self.assertGreater(p, 0.001, f"""
The gamma widths do not follow the expected Porter-Thomas distribution according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
    def test_neutron_widths(self):
        """
        Tests if neutron widths follow the expected Chi-squared distribution.
        """

        NUM_BINS = 40
        E  = self.res_ladder.E
        Gn = self.res_ladder.Gn
        gn2 = Gn * ReduceFactor(E, self.l[0], self.reaction.targ.mass, self.reaction.ac, self.reaction.proj.mass)
        dist = porter_thomas_dist(mean=self.gn2m[0], df=self.dfn[0], trunc=0.0)
        chi2_bar, p = chi2_test(dist, gn2, NUM_BINS)
        self.assertGreater(p, 0.001, f"""
The neutron widths do not follow the expected Porter-Thomas distribution according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
class TestGOESampler(unittest.TestCase):

    E0 = 200 # eV
    num_res = 10_000 # resonances
    ensemble = 'GOE' # Gaussian Orthogonal Ensemble
    beta = 1

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters:
        cls.EB   = (1e-5,5000)
        cls.mls  = [4.3166]
        cls.gn2m = [441.1355]
        cls.gg2m = [55.00000]
        cls.dfn  = [1]
        cls.dfg  = [250]
        cls.l    = [0]
        cls.j    = [3.0]

        # 2 Spingroup Case:
        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile, MLS=cls.mls, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB)
        cls.res_ladder = cls.reaction.sample(cls.ensemble)[0]

    def test_dyson_mehta_3(self):
        """
        Tests if the resonance ladder's Dyson-Mehta ∆3 statistic aligns with the prediction.
        """

        E = self.res_ladder.E
        D3_calc = deltaMehta3(E, self.EB)
        D3_pred = deltaMehtaPredict(len(E), 'GOE')

        perc_err = (D3_calc-D3_pred)/D3_pred
        self.assertLess(perc_err, 0.4, f"""
The {self.ensemble} calculated and predicted Dyson-Mehta ∆3 statistic differ by {perc_err:.2%}.
Calculated ∆3 = {D3_calc:.5f}
Predicted ∆3  = {D3_pred:.5f}
""")
    
    def test_uniform_density(self):
        """
        Tests if the resonance ladder is sampling resonances with uniform level densities.
        """

        NUM_BINS = 40
        E = self.res_ladder.E
        num_ergs = len(E)
        obs_counts, bin_edges = np.histogram(E, NUM_BINS)
        exp_counts = (num_ergs/NUM_BINS) * np.ones((NUM_BINS,))
        chi2, p = chisquare(f_obs=obs_counts, f_exp=exp_counts)
        chi2_bar = chi2 / NUM_BINS
        self.assertGreater(p, 0.001, f"""
The {self.ensemble} energies do not follow a uniform density curve according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
    def test_wigner(self):
        """
        Tests if the resonance ladder follows Wigner distribution.
        """

        NUM_BINS = 40
        E = self.res_ladder.E
        lvl_spacing = np.diff(E)
        dist = wigner_dist(scale=self.mls[0], beta=self.beta)
        chi2_bar, p = chi2_test(dist, lvl_spacing, NUM_BINS)
        self.assertGreater(p, 0.001, f"""
The {self.ensemble} level-spacings do not follow Wigner distribution according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")

    def test_level_spacing_ratio(self):
        """
        Tests if the resonance ladder follows the level-spacing ratio distribution.
        """

        NUM_BINS = 40
        E = self.res_ladder.E
        lvl_spacing = np.diff(E)
        ratio = lvl_spacing[1:] / lvl_spacing[:-1]
        dist = lvl_spacing_ratio_dist(beta=self.beta)
        chi2_bar, p = chi2_test(dist, ratio, NUM_BINS)
        self.assertGreater(p, 0.001, f"""
The {self.ensemble} level-spacing ratios do not follow the expected curve according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
class TestGUESampler(unittest.TestCase):

    E0 = 200 # eV
    num_res = 10_000 # resonances
    ensemble = 'GUE' # Gaussian Unitary Ensemble
    beta = 2

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters:
        cls.EB   = (1e-5,5000)
        cls.mls  = [4.3166]
        cls.gn2m = [441.1355]
        cls.gg2m = [55.00000]
        cls.dfn  = [1]
        cls.dfg  = [250]
        cls.l    = [0]
        cls.j    = [3.0]

        # 2 Spingroup Case:
        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile, MLS=cls.mls, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB)
        cls.res_ladder = cls.reaction.sample(cls.ensemble)[0]
    
    def test_uniform_density(self):
        """
        Tests if the resonance ladder is sampling resonances with uniform level densities.
        """

        NUM_BINS = 40
        E = self.res_ladder.E
        num_ergs = len(E)
        obs_counts, bin_edges = np.histogram(E, NUM_BINS)
        exp_counts = (num_ergs/NUM_BINS) * np.ones((NUM_BINS,))
        chi2, p = chisquare(f_obs=obs_counts, f_exp=exp_counts)
        chi2_bar = chi2 / NUM_BINS
        self.assertGreater(p, 0.001, f"""
The {self.ensemble} energies do not follow a uniform density curve according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
    def test_wigner(self):
        """
        Tests if the resonance ladder follows Wigner distribution.
        """

        NUM_BINS = 40
        E = self.res_ladder.E
        lvl_spacing = np.diff(E)
        dist = wigner_dist(scale=self.mls[0], beta=self.beta)
        chi2_bar, p = chi2_test(dist, lvl_spacing, NUM_BINS)
        self.assertGreater(p, 0.001, f"""
The {self.ensemble} level-spacings do not follow Wigner distribution according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")

    def test_level_spacing_ratio(self):
        """
        Tests if the resonance ladder follows the level-spacing ratio distribution.
        """

        NUM_BINS = 40
        E = self.res_ladder.E
        lvl_spacing = np.diff(E)
        ratio = lvl_spacing[1:] / lvl_spacing[:-1]
        dist = lvl_spacing_ratio_dist(beta=self.beta)
        chi2_bar, p = chi2_test(dist, ratio, NUM_BINS)
        self.assertGreater(p, 0.001, f"""
The {self.ensemble} level-spacing ratios do not follow the expected curve according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
class TestGSESampler(unittest.TestCase):

    E0 = 200 # eV
    num_res = 10_000 # resonances
    ensemble = 'GSE' # Gaussian Symplectic Ensemble
    beta = 4

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters:
        cls.EB   = (1e-5,5000)
        cls.mls  = [4.3166]
        cls.gn2m = [441.1355]
        cls.gg2m = [55.00000]
        cls.dfn  = [1]
        cls.dfg  = [250]
        cls.l    = [0]
        cls.j    = [3.0]

        # 2 Spingroup Case:
        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile, MLS=cls.mls, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB)
        cls.res_ladder = cls.reaction.sample(cls.ensemble)[0]
    
    def test_uniform_density(self):
        """
        Tests if the resonance ladder is sampling resonances with uniform level densities.
        """

        NUM_BINS = 40
        E = self.res_ladder.E
        num_ergs = len(E)
        obs_counts, bin_edges = np.histogram(E, NUM_BINS)
        exp_counts = (num_ergs/NUM_BINS) * np.ones((NUM_BINS,))
        chi2, p = chisquare(f_obs=obs_counts, f_exp=exp_counts)
        chi2_bar = chi2 / NUM_BINS
        self.assertGreater(p, 0.001, f"""
The {self.ensemble} energies do not follow a uniform density curve according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
    def test_wigner(self):
        """
        Tests if the resonance ladder follows Wigner distribution.
        """

        NUM_BINS = 40
        E = self.res_ladder.E
        lvl_spacing = np.diff(E)
        dist = wigner_dist(scale=self.mls, beta=self.beta)
        chi2_bar, p = chi2_test(dist, lvl_spacing, NUM_BINS)
        self.assertGreater(p, 0.001, f"""
The {self.ensemble} level-spacings do not follow Wigner distribution according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")

    def test_level_spacing_ratio(self):
        """
        Tests if the resonance ladder follows the level-spacing ratio distribution.
        """

        NUM_BINS = 40
        E = self.res_ladder.E
        lvl_spacing = np.diff(E)
        ratio = lvl_spacing[1:] / lvl_spacing[:-1]
        dist = lvl_spacing_ratio_dist(beta=self.beta)
        chi2_bar, p = chi2_test(dist, ratio, NUM_BINS)
        self.assertGreater(p, 0.001, f"""
The {self.ensemble} level-spacing ratios do not follow the expected curve according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")

if __name__ == '__main__':
    unittest.main()