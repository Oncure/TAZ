from typing import Tuple
from math import pi, sqrt, log, ceil
from sys import float_info
import numpy as np
from numpy import newaxis as NA
from scipy.special import gamma, gammaincc, gammainccinv, erfc, erfcx, erfcinv
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

# =================================================================================================
#    Spacing Distribution:
# =================================================================================================

class SpacingDistribution:
    """
    ...
    """

    def __init__(self, lvl_dens:float=1.0, **kwargs):
        'Sets SpacingDistribution attributes.'
        self.lvl_dens = float(lvl_dens)
        for key, value in kwargs.items():
            setattr(self, key, value)

    # Probability Distributions:
    def _f0(self, x):
        raise ValueError('A "_f0" method must be defined.')
    def _f1(self, x):
        func = lambda t: self._f0(t)
        intfunc = lambda x: quad(func, a=x, b=np.inf)[0]
        return np.vectorize(intfunc)(x) * self.lvl_dens
    def _f2(self, x):
        func = lambda t,x: (t-x)*self._f0(t)
        intfunc = lambda x: quad(func, a=x, b=np.inf, args=(x,))[0]
        return np.vectorize(intfunc)(x) * self.lvl_dens**2
    
    # Distribution Ratios:
    def _r1(self, x):
        return self._f0(x) / self._f1(x)
    def _r2(self, x):
        return self._f1(x) / self._f2(x)
    
    # Survival Inverse Functions:
    def _iF0(self, q):
        'Inverse of the survival function of the "f0" probability density function.'
        def func(u, y):
            x = -np.log(u)
            return self._f1(x) - y
        def __invfunc(y):
            u = brentq(func, a=0.0, b=1.0, args=(y,), xtol=float_info.epsilon, rtol=1e-15)
            x = -np.log(u)
            return x
        return np.vectorize(__invfunc)(q)
    def _iF1(self, q):
        'Inverse of the survival function of the "f1" probability density function.'
        def func(u, y):
            x = -np.log(u)
            return self._f2(x) - y
        def __invfunc(y):
            u = brentq(func, a=0.0, b=1.0, args=(y,), xtol=float_info.epsilon, rtol=1e-15)
            x = -np.log(u)
            return x
        return np.vectorize(__invfunc)(q)
    
    # Level density handlers:
    def f0(self, x):
        return self.lvl_dens * self._f0(self.lvl_dens * x)
    def f1(self, x):
        return self.lvl_dens * self._f1(self.lvl_dens * x)
    def f2(self, x):
        return self.lvl_dens * self._f2(self.lvl_dens * x)
    def r1(self, x):
        return self._r1(self.lvl_dens * x)
    def r2(self, x):
        return self._r2(self.lvl_dens * x)
    def iF0(self, q):
        'Inverse of the survival function of the "f0" probability density function.'
        return self._iF0(q) / self.lvl_dens
    def iF1(self, q):
        'Inverse of the survival function of the "f1" probability density function.'
        return self._iF1(q) / self.lvl_dens
    
    # Samplers:
    def sample_f0(self, size:tuple=None, rng=None, seed:int=None):
        'Sampling of "f0" distribution.'
        if rng is None:
            rng = np.random.RandomState(seed)
        return self.iF0(rng.random(size))
    def sample_f1(self, size:tuple=None, rng=None, seed:int=None):
        'Sampling of "f1" distribution.'
        if rng is None:
            rng = np.random.RandomState(seed)
        return self.iF1(rng.random(size))
    
    # Nicely Named Functions:
    def pdf(self, x):
        'Probability density function (same as f0).'
        return self.lvl_dens*self._f0(self.lvl_dens * x)
    def cdf(self, x):
        'Cumulative probability density function.'
        return 1 - self.sf(x)
    def sf(self, x):
        'Survival function.'
        return self._f1(self.lvl_dens * x)
    
    # xMax functions:
    def xMax_f0(self, err):
        return self.iF0(err)
    def xMax_f1(self, err):
        return self.iF1(err)
    
    def __call__(self, x):
        return self.pdf(x)
    
# =================================================================================================
#    Poisson Distribution:
# =================================================================================================
    
class PoissonGen(SpacingDistribution):
    """
    Generates a Poisson level-spacing distribution.

    Great for debugging TAZ.
    """
    def _f0(self, x):
        return np.exp(-x)
    def _f1(self, x):
        return np.exp(-x)
    def _f2(self, x):
        return np.exp(-x)
    def _r1(self, x):
        return x*0+1 # lazy way of making ones with same shape
    def _r2(self, x):
        return x*0+1 # lazy way of making ones with same shape
    def _iF0(self, q):
        return -np.log(q)
    def _iF1(self, q):
        return -np.log(q)
    
# =================================================================================================
#    Wigner Distribution:
# =================================================================================================
    
class WignerGen(SpacingDistribution):
    """
    Generates a Wigner level-spacing distribution.
    """
    def _f0(self, x):
        coef = pi/4
        return 2 * coef * x * np.exp(-coef * x*x)
    def _f1(self, x):
        return np.exp((-pi/4) * x*x)
    def _f2(self, x):
        root_coef = sqrt(pi/4)
        return erfc(root_coef * x)
    def _r1(self, x):
        return (pi/2) * x
    def _r2(self, x):
        root_coef = sqrt(pi/4)
        return 1.0 / erfcx(root_coef * x)
    def _iF0(self, q):
        return np.sqrt((-4/pi) * np.log(q))
    def _iF1(self, q):
        root_coef = sqrt(pi/4)
        return erfcinv(q) / root_coef
    
# =================================================================================================
#    Brody Distribution:
# =================================================================================================
    
class BrodyGen(SpacingDistribution):
    """
    Generates a Brody level-spacing distribution.
    """
    def _f0(self, x):
        w1 = self.w + 1.0
        a = gamma(1/w1+1)**w1
        axw = a*x**self.w
        return w1 * axw * np.exp(-axw*x)
    def _f1(self, x):
        w1 = self.w + 1.0
        a = gamma(1/w1+1)**w1
        return np.exp(-a*x**w1)
    def _f2(self, x):
        w1 = self.w + 1.0
        w1i = 1.0 / w1
        a = gamma(w1i+1)**w1
        return gammaincc(w1i, a*x**w1)
        # return (w1i*a**(-w1i)) * gammaincc(w1i, a*x**w1)
    def _r1(self, x):
        w1 = self.w + 1.0
        a = gamma(1/w1+1)**w1
        return w1*a*x**self.w
    def _r2(self, x):
        w1 = self.w + 1.0
        w1i = 1.0 / w1
        a = gamma(w1i+1)**w1
        axw1 = a*x**w1
        F2 = gammaincc(w1i, axw1)
        # F2 = (w1i * a**(-w1i)) * gammaincc(w1i, axw1)
        return np.exp(-axw1) / F2
    def _iF0(self, q):
        w1 = self.w + 1.0
        w1i = 1.0 / w1
        a = gamma(w1i+1)**w1
        return (-np.log(q) / a) ** w1i
    def _iF1(self, q):
        w1 = self.w + 1
        w1i = 1.0 / w1
        a = gamma(w1i+1)**w1
        return (gammainccinv(w1i, q) / a) ** w1i
        # inside = (w1*(a**w1i)) * q
        # return (gammainccinv(w1i, inside) / a) ** w1i
    
# =================================================================================================
#    Missing Distribution:
# =================================================================================================
    
class MissingGen(SpacingDistribution):
    """
    Generates a missing resonances level-spacing distribution.
    """
    @property
    def true_lvl_dens(self):
        return self.lvl_dens / (1-self.pM)
    def _f0(self, x):
        N_max = ceil(log(self.err, self.pM))
        mult_fact = (1-self.pM) / (1 - self.pM**(N_max+1))
        y = x*0
        coef = mult_fact
        for n in range(N_max+1):
            func_n = HighOrderSpacingGen(lvl_dens=1/(1-self.pM), n=n).f0(x)
            y += coef * func_n
            coef *= self.pM
        return y
    def _f1(self, x):
        N_max = ceil(log(self.err, self.pM))
        mult_fact = (1-self.pM)**2 / (1 - self.pM**(N_max+1))
        y = x*0
        coef = mult_fact
        for n in range(N_max+1):
            func_n = HighOrderSpacingGen(lvl_dens=1/(1-self.pM), n=n).f1(x)
            y += coef * func_n
            coef *= self.pM
        return y
    def _f2(self, x):
        N_max = ceil(log(self.err, self.pM))
        mult_fact = (1-self.pM)**3 / (1 - self.pM**(N_max+1))
        y = x*0
        coef = mult_fact
        for n in range(N_max+1):
            func_n = HighOrderSpacingGen(lvl_dens=1/(1-self.pM), n=n).f2(x)
            y += coef * func_n
            coef *= self.pM
        return y
    # def _f0(self, x):
    #     x = np.array(x, ndmin=1)
    #     N_max = ceil(log(self.err, self.pM))
    #     coef = (self.pM**np.arange(N_max+1))[NA,:]
    #     func_n = np.zeros((len(x),N_max+1))
    #     mult_fact = (1-self.pM) / (1 - self.pM**(N_max+1))
    #     np.sum(coef * func_n, axis=1)
    #     for n in range(N_max+1):
    #         func_n[:,n] = HighOrderSpacingGen(n=n)._f0(x)
    #     return mult_fact * np.sum(coef * func_n, axis=1)
    # def _f1(self, x):
    #     x = np.array(x, ndmin=1)
    #     N_max = ceil(log(self.err, self.pM))
    #     coef = (self.pM**np.arange(N_max+1))[NA,:]
    #     func_n = np.zeros((len(x),N_max+1))
    #     mult_fact = (1-self.pM)**2 / (1 - self.pM**(N_max+1))
    #     for n in range(N_max+1):
    #         func_n[:,n] = HighOrderSpacingGen(n=n)._f1(x)
    #     return mult_fact * np.sum(coef * func_n, axis=1)
    # def _f2(self, x):
    #     x = np.array(x, ndmin=1)
    #     N_max = ceil(log(self.err, self.pM))
    #     coef = (self.pM**np.arange(N_max+1))[NA,:]
    #     func_n = np.zeros((len(x),N_max+1))
    #     mult_fact = (1-self.pM)**3 / (1 - self.pM**(N_max+1))
    #     for n in range(N_max+1):
    #         func_n[:,n] = HighOrderSpacingGen(n=n)._f2(x)
    #     return mult_fact * np.sum(coef * func_n, axis=1)
    
# =================================================================================================
#    Higher-Order Spacing Distributions:
# =================================================================================================
    
def _gamma_ratio(x):
    """
    A function to calculate the ratio, `Gamma(x/2) / Gamma((x-1)/2)`. This function is used instead
    of calculating each gamma separately for numerical stability.

    Parameters:
    ----------
    x :: int
        The function parameter of `Gamma(x/2) / Gamma((x-1)/2)`.

    Returns:
    -------
    ratio :: float
        The calculated ratio `Gamma(x/2) / Gamma((x-1)/2)`.

    See Also
    --------
    `HighOrderSpacingGen`
    """
    rpii = 1.0 / sqrt(pi)
    if hasattr(x, '__iter__'):
        ratio = np.zeros(len(x))
        for idx, w in enumerate(x):
            q = rpii
            for i in range(3,int(w)):
                q = (i-2) / (2*q)
            ratio[idx] = q
    else:
        ratio = rpii
        for i in range(3,int(x)+1):
            ratio = (i-2) / (2*ratio)
    return ratio

def _high_order_variance(n:int):
    """
    A function for calculating the variance of the `n+1`-th nearest level-spacing distribution.
    This is used for the Gaussian Approximation when the analytical solution becomes too costly
    to compute.

    Parameters:
    ----------
    n :: int
        The number of levels between the two selected levels.

    Returns:
    -------
    variance :: float
        The variance of the high-order level-spacing distribution.

    See Also
    --------
    `HighOrderSpacingGen`
    """
    a = (n**2 + 5*n + 2)/2
    B = (_gamma_ratio(a+2) / (n+1))**2
    variance = (a+1)/(2*B) - (n+1)**2
    return variance

class HighOrderSpacingGen(SpacingDistribution):
    """
    Generates the `n+1`-th nearest neighbor level-spacing distribution as determined by the
    Gaussian Orthogonal Ensemble (GOE). The distribution is calculated at each value in the numpy
    array, `x`.

    Source: https://journals.aps.org/pre/pdf/10.1103/PhysRevE.60.5371
    """

    def _f0(self, x):
        n = self.n
        if n <= 15: # Lower n --> Exact Calculation
            a = n + (n+1)*(n+2)/2 # (Eq. 10)
            rB = _gamma_ratio(a+2) / (n+1) # square root of B (Eq. 12)
            coef = 2 * rB / gamma((a+1)/2)
            rBx  = rB * x
            return coef * rBx**a * np.exp(-rBx**2) # (Eq. 11)
        else: # Higher n --> Gaussian Approximation
            sig = np.sqrt(_high_order_variance(n))
            return norm.pdf(x, n+1, sig)
        
    def _f1(self, x):
        n = self.n
        if n <= 15: # Lower n --> Exact Calculation
            a = n + (n+1)*(n+2)/2 # (Eq. 10)
            rB = _gamma_ratio(a+2) / (n+1) # square root of B (Eq. 12)
            return gammaincc((a+1)/2, (rB * x)**2)
        else: # Higher n --> Gaussian Approximation
            sig = np.sqrt(_high_order_variance(n))
            return (1/2) * erfc((x+n+1)/(sig*np.sqrt(2)))
        
    def _f2(self, x):
        n = self.n
        if n <= 15: # Lower n --> Exact Calculation
            a = n + (n+1)*(n+2)/2 # (Eq. 10)
            rB = _gamma_ratio(a+2) / (n+1) # square root of B (Eq. 12)
            return (n+1)*gammaincc((a+2)/2, (rB * x)**2) - x*gammaincc((a+1)/2, (rB * x)**2)
        else: # Higher n --> Gaussian Approximation
            sig = np.sqrt(_high_order_variance(n))
            f0 = norm.pdf(x, n+1, sig)
            f1 = (1/2) * erfc((x+n+1)/(sig*np.sqrt(2)))
            return sig**2*f0 + (x+1)*f1 - 1

# =================================================================================================
#    Higher-Order Spacing Distributions:
# =================================================================================================

class MergedDistributionBase:
    """
    ...
    """
    def __init__(self, lvl_dens:float):
        self.lvl_dens = lvl_dens
    def f0(self, x, priorL, priorR):
        raise ValueError('"f0" has not been defined.')
    def f1(self, x, prior):
        raise ValueError('"f1" has not been defined.')
    def f2(self, x):
        raise ValueError('"f2" has not been defined.')
    def xMax_f0(self, err):
        raise ValueError('"xMax_f0" has not been defined.')
    def xMax_f1(self, err):
        raise ValueError('"xMax_f1" has not been defined.')
    def r1(self, x):
        return self.f0(x) / self.f1(x)
    def r2(self, x):
        return self.f1(x) / self.f2(x)
    def iF0(self, *args, **kwargs):
        return NotImplementedError('Inverse functions have not been implemented for merged distributions.')
    def iF1(self, *args, **kwargs):
        return NotImplementedError('Inverse functions have not been implemented for merged distributions.')
    def sample_f0(self, *args, **kwargs):
        return NotImplementedError('Sampling has not been implemented for merged distributions.')
    def sample_f1(self, *args, **kwargs):
        return NotImplementedError('Sampling has not been implemented for merged distributions.')
    def pdf(self, x, priorL, priorR):
        return self.f0(x, priorL, priorR)
    def cdf(self, x, priorL, priorR):
        return NotImplementedError('"cdf" has not been implemented for merged distributions.')
    def sf(self, x, priorL, priorR):
        return NotImplementedError('"sf" has not been implemented for merged distributions.')

def merge(*distributions:Tuple[SpacingDistribution]):
    """
    ...
    """
    
    G = len(distributions)
    if G == 1:
        distribution = distributions[0]
        lvl_dens_comb = distribution.lvl_dens
        class MergedSpacingDistributionGen(MergedDistributionBase):
            def f0(self, x, priorL, priorR):
                return distribution.f0(x)
            def f1(self, x, prior):
                return distribution.f1(x)
            def f2(self, x):
                return distribution.f2(x)
            def xMax_f0(self, err):
                return distribution.iF0(err)
            def xMax_f1(self, err):
                return distribution.iF1(err)
        Z0 = Z1 = Z2 = None
    else:
        lvl_dens_comb = np.sum([distribution.lvl_dens for distribution in distributions])
        def c_func(x):
            x = np.array(x)
            c = np.ones(x.shape)
            for distribution in distributions:
                c *= distribution.f2(x)
            return c

        # Normalization Factors:
        Z0 = np.zeros((G,G))
        for i in range(1,G):
            for j in range(i):
                func = lambda x: c_func(x) * distributions[i].r2(x) * distributions[j].r2(x)
                Z0[i,j] = quad(func, a=0.0, b=np.inf)[0]
                Z0[j,i] = Z0[i,j]
        for i in range(G):
            func = lambda x: c_func(x) * distributions[i].r2(x) * distributions[i].r1(x)
            Z0[i,i] = quad(func, a=0.0, b=np.inf)[0]
        Z1 = np.zeros((G,))
        for i in range(G):
            func = lambda x: c_func(x) * distributions[i].r2(x)
            Z1[i] = quad(func, a=0.0, b=np.inf)[0]
        Z2 = quad(c_func, a=0.0, b=np.inf)[0]

        # Level-densities:
        lvl_denses = np.zeros((G,))
        for i,distribution in enumerate(distributions):
            lvl_denses[i] = distribution.lvl_dens

        # Merged Distribution:
        class MergedSpacingDistributionGen(MergedDistributionBase):
            def f0(self, x, priorL, priorR):
                x = np.array(x)
                L = len(x)
                priorL = np.array(priorL)
                priorR = np.array(priorR)
                v = np.zeros((L,G))
                d = np.zeros((L,G))
                norm = (priorL[:,NA,:] @ Z0[NA,:,:] @ priorR[:,:,NA])[:,0,0]
                for i, distribution in enumerate(distributions):
                    v[:,i] = distribution.r2(x)
                    u = distribution.r1(x)
                    d[:,i] = v[:,i] * (u - v[:,i])
                F = c_func(x) / norm * ( \
                    np.sum(priorL * v, axis=1) \
                    * np.sum(priorR * v, axis=1) \
                    + np.sum(priorL * priorR * d, axis=1))
                return F
            def f1(self, x, prior):
                x = np.array(x)
                L = len(x)
                norm = np.sum(prior * Z1)
                prior = np.array(prior)
                v = np.zeros((L,G))
                for i, distribution in enumerate(distributions):
                    v[:,i] = distribution.r2(x)
                F = c_func(x) / norm * np.sum(prior*v, axis=1)
                return F
            def f2(self, x):
                F = c_func(x) / Z2
                return F
            
            def xMax_f0(self, err):
                def func(u):
                    if u == 0.0:
                        return -err
                    x = -np.log(u)
                    fx_max = 0.0
                    for g,distribution in enumerate(distributions):
                        fx = distribution.r2(x) / np.min(lvl_denses*Z0[:,g])
                        if fx > fx_max:
                            fx_max = fx
                    fx *= c_func(x)
                    return fx - err
                u = brentq(func, a=0.0, b=1.0, xtol=float_info.epsilon, rtol=1e-15)
                x = -np.log(u)
                return x
            def xMax_f1(self, err):
                def func(u):
                    if u == 0.0:
                        return -err
                    x = -np.log(u)
                    fx = c_func(x) / np.min(lvl_denses*Z1)
                    return fx - err
                u = brentq(func, a=0.0, b=1.0, xtol=float_info.epsilon, rtol=1e-15)
                x = -np.log(u)
                return x

    merged_spacing_distribution = MergedSpacingDistributionGen(lvl_dens=lvl_dens_comb)
    return merged_spacing_distribution