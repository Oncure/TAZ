import sys
sys.path.append('../TAZ')
import TAZ
from TAZ.Theory.LevelSpacingDists import PoissonGen, WignerGen, BrodyGen, MissingGen, HighOrderSpacingGen

import numpy as np
from scipy.integrate import quad

import unittest

class TestSpacingDistributions(unittest.TestCase):
    """
    Here, we intend to verify that all of the level-spacing distribution quantities are correct
    and match the expected distributions.
    """

    places = 7

    def test_poisson(self):
        'Tests the PoissonGen distribution generator.'

        MLS = 42.0

        X = np.array([0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])

        DISTNAME = f'PoissonGen(lvl_dens={1/MLS})'
        dist = PoissonGen(lvl_dens=1/MLS)

        I = quad(dist.f0, 0, np.inf)[0]
        self.assertAlmostEqual(I, 1.0, self.places, f'"{DISTNAME}.f0" does not integrate to 1, but instead {I}.')

        I = quad(dist.f1, 0, np.inf)[0]
        self.assertAlmostEqual(I, 1.0, self.places, f'"{DISTNAME}.f1" does not integrate to 1, but instead {I}.')

        xf = lambda x: x * dist.f0(x)
        E = quad(xf, 0, np.inf)[0]
        self.assertAlmostEqual(E, MLS, self.places, f'"{DISTNAME}.f0" should have a mean level-spacing of {MLS}, but it is instead {E}.')

        for x in X:
            F1 = dist.f1(x) * MLS
            F1_  = quad(dist.f0, x, np.inf)[0]
            self.assertAlmostEqual(F1, F1_, self.places, f'"{DISTNAME}.f1({x})" should be the integral of f0 from {x} to infinity.')

            F2 = dist.f2(x) * MLS
            F2_  = quad(dist.f1, x, np.inf)[0]
            self.assertAlmostEqual(F2, F2_, self.places, f'"{DISTNAME}.f2({x})" should be the integral of f1 from {x} to infinity.')

        X_ = dist.iF0(MLS*dist.f1(X))
        for x, x_ in zip(X, X_):
            self.assertAlmostEqual(x, x_, self.places, f'{DISTNAME}.iF0 is not the inverse CDF of f0 when evaluated at x = {x}.')

        X_ = dist.iF1(MLS*dist.f2(X))
        for x, x_ in zip(X, X_):
            self.assertAlmostEqual(x, x_, self.places, f'{DISTNAME}.iF1 is not the inverse CDF of f1 when evaluated at x = {x}.')

        R1 = dist.r1(X)
        R2 = dist.r2(X)
        F0 = dist.f0(X)
        F1 = dist.f1(X)
        F2 = dist.f2(X)
        for i, x, in enumerate(X):
            self.assertAlmostEqual(R1[i], F0[i]/F1[i], self.places, f'{DISTNAME}.r1 is not f0/f1 when evaluated at x = {x}.')
            self.assertAlmostEqual(R2[i], F1[i]/F2[i], self.places, f'{DISTNAME}.r2 is not f1/f2 when evaluated at x = {x}.')

    def test_wigner(self):
        'Tests the WignerGen distribution generator.'

        MLS = 42.0

        X = np.array([0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])

        DISTNAME = f'WignerGen(lvl_dens={1/MLS})'
        dist = WignerGen(lvl_dens=1/MLS)

        I = quad(dist.f0, 0, np.inf)[0]
        self.assertAlmostEqual(I, 1.0, self.places, f'"{DISTNAME}.f0" does not integrate to 1, but instead {I}.')

        I = quad(dist.f1, 0, np.inf)[0]
        self.assertAlmostEqual(I, 1.0, self.places, f'"{DISTNAME}.f1" does not integrate to 1, but instead {I}.')

        xf = lambda x: x * dist.f0(x)
        E = quad(xf, 0, np.inf)[0]
        self.assertAlmostEqual(E, MLS, self.places, f'"{DISTNAME}.f0" should have a mean level-spacing of {MLS}, but it is instead {E}.')

        for x in X:
            F1 = dist.f1(x) * MLS
            F1_  = quad(dist.f0, x, np.inf)[0]
            self.assertAlmostEqual(F1, F1_, self.places, f'"{DISTNAME}.f1({x})" should be the integral of f0 from {x} to infinity.')

            F2 = dist.f2(x) * MLS
            F2_  = quad(dist.f1, x, np.inf)[0]
            self.assertAlmostEqual(F2, F2_, self.places, f'"{DISTNAME}.f2({x})" should be the integral of f1 from {x} to infinity.')

        X_ = dist.iF0(MLS*dist.f1(X))
        for x, x_ in zip(X, X_):
            self.assertAlmostEqual(x, x_, self.places, f'{DISTNAME}.iF0 is not the inverse CDF of f0 when evaluated at x = {x}.')

        X_ = dist.iF1(MLS*dist.f2(X))
        for x, x_ in zip(X, X_):
            self.assertAlmostEqual(x, x_, self.places, f'{DISTNAME}.iF1 is not the inverse CDF of f1 when evaluated at x = {x}.')

        R1 = dist.r1(X)
        R2 = dist.r2(X)
        F0 = dist.f0(X)
        F1 = dist.f1(X)
        F2 = dist.f2(X)
        for i, x, in enumerate(X):
            self.assertAlmostEqual(R1[i], F0[i]/F1[i], self.places, f'{DISTNAME}.r1 is not f0/f1 when evaluated at x = {x}.')
            self.assertAlmostEqual(R2[i], F1[i]/F2[i], self.places, f'{DISTNAME}.r2 is not f1/f2 when evaluated at x = {x}.')
    
    def test_brody(self):
        'Tests the BrodyGen distribution generator.'

        MLS = 42.0
        w = 0.8

        X = np.array([0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])

        DISTNAME = f'BrodyGen(lvl_dens={1/MLS}, w={w})'
        dist = BrodyGen(lvl_dens=1/MLS, w=w)

        I = quad(dist.f0, 0, np.inf)[0]
        self.assertAlmostEqual(I, 1.0, self.places, f'"{DISTNAME}.f0" does not integrate to 1, but instead {I}.')

        I = quad(dist.f1, 0, np.inf)[0]
        self.assertAlmostEqual(I, 1.0, self.places, f'"{DISTNAME}.f1" does not integrate to 1, but instead {I}.')

        xf = lambda x: x * dist.f0(x)
        E = quad(xf, 0, np.inf)[0]
        self.assertAlmostEqual(E, MLS, self.places, f'"{DISTNAME}.f0" should have a mean level-spacing of {MLS}, but it is instead {E}.')

        for x in X:
            F1 = dist.f1(x) * MLS
            F1_  = quad(dist.f0, x, np.inf)[0]
            self.assertAlmostEqual(F1, F1_, self.places, f'"{DISTNAME}.f1({x})" should be the integral of f0 from {x} to infinity.')

            F2 = dist.f2(x) * MLS
            F2_  = quad(dist.f1, x, np.inf)[0]
            self.assertAlmostEqual(F2, F2_, self.places, f'"{DISTNAME}.f2({x})" should be the integral of f1 from {x} to infinity.')

        X_ = dist.iF0(MLS*dist.f1(X))
        for x, x_ in zip(X, X_):
            self.assertAlmostEqual(x, x_, self.places, f'{DISTNAME}.iF0 is not the inverse CDF of f0 when evaluated at x = {x}.')

        X_ = dist.iF1(MLS*dist.f2(X))
        for x, x_ in zip(X, X_):
            self.assertAlmostEqual(x, x_, self.places, f'{DISTNAME}.iF1 is not the inverse CDF of f1 when evaluated at x = {x}.')

        R1 = dist.r1(X)
        R2 = dist.r2(X)
        F0 = dist.f0(X)
        F1 = dist.f1(X)
        F2 = dist.f2(X)
        for i, x, in enumerate(X):
            self.assertAlmostEqual(R1[i], F0[i]/F1[i], self.places, f'{DISTNAME}.r1 is not f0/f1 when evaluated at x = {x}.')
            self.assertAlmostEqual(R2[i], F1[i]/F2[i], self.places, f'{DISTNAME}.r2 is not f1/f2 when evaluated at x = {x}.')
    
    def test_missing(self):
        'Tests the MissingGen distribution generator.'

        MLS = 42.0
        pM = 0.2
        err = 1e-6

        X = np.array([0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])

        DISTNAME = f'MissingGen(lvl_dens={1/MLS}, pM={pM}, err={err})'
        dist = MissingGen(lvl_dens=1/MLS, pM=pM, err=err)

        I = quad(dist.f0, 0, np.inf)[0]
        self.assertAlmostEqual(I, 1.0, 4, f'"{DISTNAME}.f0" does not integrate to 1, but instead {I}.')

        I = quad(dist.f1, 0, np.inf)[0]
        self.assertAlmostEqual(I, 1.0, 4, f'"{DISTNAME}.f1" does not integrate to 1, but instead {I}.')

        xf = lambda x: x * dist.f0(x)
        E = quad(xf, 0, np.inf)[0]
        self.assertAlmostEqual(E, MLS, 4, f'"{DISTNAME}.f0" should have a mean level-spacing of {MLS}, but it is instead {E}.')

        for x in X:
            F1 = dist.f1(x) * MLS
            F1_  = quad(dist.f0, x, np.inf)[0]
            self.assertAlmostEqual(F1, F1_, 4, f'"{DISTNAME}.f1({x})" should be the integral of f0 from {x} to infinity.')

            F2 = dist.f2(x) * MLS
            F2_  = quad(dist.f1, x, np.inf)[0]
            self.assertAlmostEqual(F2, F2_, 4, f'"{DISTNAME}.f2({x})" should be the integral of f1 from {x} to infinity.')

        X_ = dist.iF0(MLS*dist.f1(X))
        for x, x_ in zip(X, X_):
            self.assertAlmostEqual(x, x_, 4, f'{DISTNAME}.iF0 is not the inverse CDF of f0 when evaluated at x = {x}.')

        X_ = dist.iF1(MLS*dist.f2(X))
        for x, x_ in zip(X, X_):
            self.assertAlmostEqual(x, x_, 4, f'{DISTNAME}.iF1 is not the inverse CDF of f1 when evaluated at x = {x}.')

        R1 = dist.r1(X)
        R2 = dist.r2(X)
        F0 = dist.f0(X)
        F1 = dist.f1(X)
        F2 = dist.f2(X)
        for i, x, in enumerate(X):
            self.assertAlmostEqual(R1[i], F0[i]/F1[i], self.places, f'{DISTNAME}.r1 is not f0/f1 when evaluated at x = {x}.')
            self.assertAlmostEqual(R2[i], F1[i]/F2[i], self.places, f'{DISTNAME}.r2 is not f1/f2 when evaluated at x = {x}.')
    
    # def test_high_order(self):
    #     '...'
    #     raise NotImplementedError('...')
        
if __name__ == '__main__':
    unittest.main()