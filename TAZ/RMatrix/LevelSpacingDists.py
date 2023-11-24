from math import pi, sqrt, ceil, log
import numpy as np
from scipy.special import gamma, gammaincc, gammainccinv, erfc, erfcx, erfcinv
from scipy.stats import norm

__doc__ = """
This module is the collection of level-spacing distribution functions and classes.
"""

# =================================================================================================
#    Level-Spacing Probability Distributions
# =================================================================================================

def _gamma_ratio(x):
    """
    A function to calculate the ratio, `Gamma(x/2) / Gamma((x-1)/2)`. This function is used instead
    of calculating each gamma separately for numerical stability.
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
    """
    a = (n**2 + 5*n + 2)/2
    B =(_gamma_ratio(a+2) / (n+1))**2
    return (a+1)/(2*B) - (n+1)**2

def _high_order_level_spacing_parts(X, n:int, orders:tuple=(0,1,2)):
    """
    Generates the `n+1`-th nearest neighbor level-spacing distribution as determined by the
    Gaussian Orthogonal Ensemble (GOE). The distribution is calculated at each value in the numpy
    array, `X`. Each order in `orders` request the order-th integrated level-spacing distribution.

    Source: https://journals.aps.org/pre/pdf/10.1103/PhysRevE.60.5371
    """
    out = []
    if n <= 15: # Lower n --> Exact Calculation
        a = n + (n+1)*(n+2)/2 # (Eq. 10)
        rB = _gamma_ratio(a+2) / (n+1) # square root of B (Eq. 12)
        coef = 2 * rB / gamma((a+1)/2)
        rBX  = rB * X
        for order in orders:
            if   order == 0: # Level-Spacing PDF
                F0 = coef * rBX**a * np.exp(-rBX**2) # (Eq. 11)
                out.append(F0)
            elif order == 1: # First Integral
                F1 = gammaincc((a+1)/2, rBX**2)
                out.append(F1)
            elif order == 2: # Second Integral
                F2 = gammaincc((a+2)/2, rBX**2)
                out.append(F2)
    else: # Higher n --> Gaussian Approximation
        sig = np.sqrt(_high_order_variance(n))
        for order in orders:
            if   order == 0: # Level-Spacing PDF
                F0 = norm.pdf(X, n+1, sig)
                out.append(F0)
            elif order == 1: # First Integral
                F1 = (1/2) * erfc((X+n+1)/(sig*sqrt(2)))
                out.append(F1)
            elif order == 2: # Second Integral
                if 0 not in orders:      F0 = norm.pdf(X, n+1, sig)
                if 1 not in orders:      F1 = (1/2) * erfc((X+n+1)/(sig*sqrt(2)))
                F2 = sig**2*F0 + (X+1)*F1 - 1
                out.append(F2)
    return tuple(out)

def high_order_level_spacing(X, n:int):
    return _high_order_level_spacing_parts(X, n, orders=(0,))[0]

class Distribution:
    """
    A class for level-spacing distributions, their integrals and inverses. Such distributions
    have been defined for Wigner distribution, Brody distribution, and the missing distribution.
    """
    def __init__(self, f0, f1=None, f2=None, parts=None, if1=None, if2=None, Freq=None):
        """
        ...
        """
        self.f0 = f0

        if f1 is None:
            raise NotImplementedError('Integration for f1 has not been implemented yet.')
        else:
            self.f1 = f1

        if f2 is None:
            raise NotImplementedError('Integration for f2 has not been implemented yet.')
        else:
            self.f2 = f2

        if parts is None:
            def parts(x):
                F0 = self.f0(x)
                F1 = self.f1(x)
                F2 = self.f2(x)
                return F2, F0/F1, F1/F2
        else:
            self.parts = parts

        if if1 is None:
            raise NotImplementedError('The inverse of f1 has not been implemented yet.')
        else:
            self.if1 = if1

        if if2 is None:
            raise NotImplementedError('The inverse of f2 has not been implemented yet.')
        else:
            self.if2 = if2

        self.Freq = Freq

    def __call__(self, _X):
        return self.f0(_X)
    @property
    def pdf(self):
        return self.f0
    @property
    def cdf(self):
        return lambda _X: 1.0 - self.f1(_X)

    def sample_f0(self, size:tuple=None, rng=None, seed:int=None):
        'Inverse CDF Sampling on f0.'
        if rng is None:
            rng = np.random.default_rng(seed)
        return self.if1(rng.random(size))
    
    def sample_f1(self, size:tuple=None, rng=None, seed:int=None):
        'Inverse CDF Sampling on f1.'
        if rng is None:
            rng = np.random.default_rng(seed)
        return self.if2(rng.random(size))

    @classmethod
    def wigner(cls, Freq:float=1.0):
        """
        Sample Wigner distribution without missing resonances considered.

        Inputs:
        ------
        Freq :: float
            Mean level-density. Default = 1.0.
        """
        pid4  = pi/4
        coef = pid4*Freq**2
        root_coef = sqrt(coef)

        def get_f0(X):
            return (2*coef) * X * np.exp(-coef * X*X)
        def get_f1(X):
            return np.exp(-coef * X*X)
        def get_f2(X):
            return erfc(root_coef * X) / Freq
        def get_parts(X):
            fX = root_coef * X
            R1 = 2 * root_coef * fX
            R2 = Freq / erfcx(fX)       # "erfcx(x)" is "exp(x^2)*erfc(x)"
            F2 = np.exp(-fX*fX) / R2    # Getting the "erfc(x)" from "erfcx(x)"
            return F2, R1, R2
        def get_if1(R):
            return np.sqrt(-np.log(R)) / root_coef
        def get_if2(R):
            return erfcinv(R) / root_coef
        return cls(f0=get_f0, f1=get_f1, f2=get_f2, parts=get_parts, if1=get_if1, if2=get_if2, Freq=Freq)

    @classmethod
    def brody(cls, Freq:float=1.0, w:float=0.0):
        """
        Sample Brody distribution without missing levels considered. The Brody distribution
        is an interpolation between Wigner distribution and Poisson distribution. Brody
        distribution is said to be more applicable for even-mass isotopes with an atomic mass
        number above 150. The Brody parameter can range from 0 to 1, where w=0 gives a Poisson
        distribution and w=1 gives a Wigner distribution.

        Inputs:
        ------
        Freq :: float
            Mean level-density. Default = 1.0.
        w    :: float
            Brody parameter. Default = 0.0.
        """
        w1i = 1.0 / (w+1)
        a = (Freq*w1i*gamma(w1i))**(w+1)

        def get_f0(X):
            aXw = a*X**w
            return (w+1) * aXw * np.exp(-aXw*X)
        def get_f1(X):
            return np.exp(-a*X**(w+1))
        def get_f2(X):
            return (w1i*a**(-w1i)) * gammaincc(w1i, a*X**(w+1))
        def get_parts(X):
            aXw = a*X**w
            aXw1 = aXw*X
            R1 = (w+1)*aXw
            F2 = (w1i * a**(-w1i)) * gammaincc(w1i, aXw1)
            R2 = np.exp(-aXw1) / F2
            return F2, R1, R2
        def get_if1(R):
            return (-np.log(R) / a) ** w1i
        def get_if2(R):
            return (gammainccinv(w1i, R) / a) ** w1i
        return cls(f0=get_f0, f1=get_f1, f2=get_f2, parts=get_parts, if1=get_if1, if2=get_if2, Freq=Freq)
    
    @classmethod
    def missing(cls, Freq:float=1.0, pM:float=0.0, err:float=0.005):
        """
        Sample Wigner distribution with missing resonances considered.

        Source: http://www.lib.ncsu.edu/resolver/1840.16/4284 (Eq. 4.1)

        Inputs:
        ------
        Freq :: float
            Mean level-density. Default = 1.0.
        pM   :: float
            Fraction of missing resonances. Default = 0.0.
        err  :: float
            Maximum error in PDF. Default = 0.005.

        Returns:
        -------
        distribution :: Distribution
            The distribution object for the missing-resonances level-spacing distribution.
        """
        
        # If we assume there are no missing resonances, the PDF converges to Wigner:
        if pM == 0.0:
            print(RuntimeWarning('Warning: the "missing" distribution has a zero missing resonance fraction.'))
            return cls.wigner(Freq)
        
        N_max = ceil(log(err, pM))
        coef = (pM**np.arange(N_max+1))[:,np.newaxis]
        mult_fact = Freq * (1-pM) / (1 - pM**(N_max+1))
        def get_f0(X):
            func = lambda _n: _high_order_level_spacing_parts(Freq*X, _n, orders=(0,))
            values = [func(n) for n in range(N_max+1)]
            return mult_fact * np.sum(coef * np.array([v[0] for v in values]), axis=0)
        def get_f1(X):
            func = lambda _n: _high_order_level_spacing_parts(Freq*X, _n, orders=(1,))
            values = [func(n) for n in range(N_max+1)]
            return mult_fact * np.sum(coef * np.array([v[0] for v in values]), axis=0)
        def get_f2(X):
            func = lambda _n: _high_order_level_spacing_parts(Freq*X, _n, orders=(2,))
            values = [func(n) for n in range(N_max+1)]
            return mult_fact * np.sum(coef * np.array([v[0] for v in values]), axis=0)
        def get_parts(X):
            func = lambda _n: _high_order_level_spacing_parts(Freq*X, _n, orders=(0,1,2))
            values = [func(n) for n in range(N_max+1)]
            V0, V1, V2 = zip(*values)
            F0 = mult_fact * np.sum(coef*V0, axis=0)
            F1 = mult_fact * np.sum(coef*V1, axis=0)
            F2 = mult_fact * np.sum(coef*V2, axis=0)
            R1 = F0 / F1
            R2 = F1 / F2
            return F2, R1, R2
        def get_if1(X):
            raise NotImplementedError('Inverse Function for f1 has not been implemented yet.')
        def get_if2(X):
            raise NotImplementedError('Inverse Function for f2 has not been implemented yet.')
        return cls(f0=get_f0, f1=get_f1, f2=get_f2, parts=get_parts, if1=get_if1, if2=get_if2, Freq=(1-pM)*Freq)

class Distributions:
    """
    A class that collects multiple `Distribution` objects together.
    """
    def __init__(self, *distributions:Distribution):
        'Initializing distributions'
        self.distributions = list(distributions)
        self.Freq = np.array([distr.Freq for distr in self.distributions])
    
    # Functions and properties:
    def f0(self, X):
        if not hasattr(X, '__iter__'):
            X = np.array([X])
        return np.array([distr.f0(X)    for distr in self.distributions]).T
    def f1(self, X):
        if not hasattr(X, '__iter__'):
            X = np.array([X])
        return np.array([distr.f1(X)    for distr in self.distributions]).T
    def f2(self, X):
        if not hasattr(X, '__iter__'):
            X = np.array([X])
        return np.array([distr.f2(X)    for distr in self.distributions]).T
    def if1(self, X):
        if not hasattr(X, '__iter__'):
            X = np.array([X])
        return np.array([distr.if1(X)   for distr in self.distributions]).T
    def if2(self, X):
        if not hasattr(X, '__iter__'):
            X = np.array([X])
        return np.array([distr.if2(X)   for distr in self.distributions]).T
    def parts(self, X):
        if not hasattr(X, '__iter__'):
            X = np.array([X])
        parts = np.array([distr.parts(X) for distr in self.distributions]).transpose(2,0,1)
        return parts[:,:,0], parts[:,:,1], parts[:,:,2]
    
    @property
    def FreqTot(self):
        return np.sum(self.Freq)
    @property
    def num_dists(self):
        return len(self.distributions)

    def __call__(self, _X):
        return self.f0(_X)
    @property
    def pdf(self):
        return self.f0
    @property
    def cdf(self):
        return lambda _X: 1.0 - self.f1(_X)

    def __getitem__(self, indices):
        if hasattr(indices, '__iter__'):
            distributions = [self.distributions[idx] for idx in indices]
            return self.__class__(*distributions)
        else:
            return self.distributions[indices]

    @classmethod
    def wigner(cls, Freq):
        'Sample Wigner distribution for each spingroup.'
        Freq = Freq.reshape(-1,)
        distributions = [Distribution.wigner(freq_g) for freq_g in Freq]
        return cls(*distributions)
    @classmethod
    def brody(cls, Freq, w=None):
        'Sample Brody distribution for each spingroup.'
        G = len(Freq)
        if w is None:
            w = np.zeros((G,))
        Freq = Freq.reshape(-1,)
        w    = w.reshape(-1,)
        distributions = [Distribution.brody(freq_g, w_g) for freq_g,w_g in zip(Freq,w)]
        return cls(*distributions)
    @classmethod
    def missing(cls, Freq, pM=None, err:float=5e-3):
        'Sample Missing distribution for each spingroup.'
        G = len(Freq)
        if pM is None:
            pM = np.zeros((G,))
        Freq = Freq.reshape(-1,)
        pM   = pM.reshape(-1,)
        distributions = [Distribution.missing(freq_g, pM_g, err) for freq_g,pM_g in zip(Freq,pM)]
        return cls(*distributions)
    
# =================================================================================================
#    Dyson-Mehta ∆3 Metric:
# =================================================================================================

def deltaMehta3(E, EB:tuple):
    """
    Finds the Dyson-Mehta ∆3 metric for the given data.

    Source: https://arxiv.org/pdf/2011.04633.pdf (Eq. 21 & 22)

    Let `L` be the number of recorded resonances in the ladder.

    Parameters:
    ----------
    E  : float [L]
        The recorded resonance energies.
    
    EB : float [2]
        The lower and upper energies for the resonance ladder.

    Returns:
    -------
    delta_3 : float
        The Dyson-Mehta ∆3 metric.
    """

    E = np.sort(E) # sort energies if not already sorted
    z = (E-EB[0])/(EB[1]-EB[0]) # renormalize energies
    s1 = np.sum(z)
    s2 = np.sum(z**2)
    a = np.arange( len(z)-1, -1, -1 )
    s3 = np.sum((2*a+1)*z)
    delta_3 = 6*s1*s2 - 4*s1**2 - 3*s2**2 + s3
    return delta_3

def deltaMehtaPredict(L:int, ensemble:str='GOE'):
    """
    A function that predicts the value of the Dyson-Mehta ∆3 metric based on the number of
    observed resonances and type of ensemble.

    Source: https://www.osti.gov/servlets/purl/1478482 (Eq. 31 & 32 & 33)

    Parameters:
    ----------
    L        : int
        The number of resonances.

    ensemble : 'GOE', 'Poisson', or 'picket'
        The ensemble to assumed under the calculation of the Dyson-Mehta ∆3 metric.

    Returns:
    -------
    delta_3 : float
        The prediction on the Dyson-Mehta ∆3 metric.
    """

    if   ensemble.lower() == 'goe':
        delta_3 = pi**(-2) * (log(L) - 0.0687)
    elif ensemble.lower() == 'poisson':
        delta_3 = L/15
    elif ensemble.lower() == 'picket':
        delta_3 = 1/12
    else:
        raise ValueError(f'Unknown ensemble, {ensemble}. Please choose from "GOE", "Poisson" or "picket".')
    return delta_3

# def _deltaMehta3AB(X, XB:tuple, A:float, B:float):
#     """
#     ...
#     """

#     N  = len(X)
#     H  = np.arange(N+1)
#     Y  = A*X+B
#     PB = (A*XB[0]+B, A*XB[1]+B-N)
#     P1 = Y-H[:-1]
#     P2 = Y-H[1:]
#     return (np.sum(P1**3 - P2**3) + (PB[1]**3 - PB[0]**3))/(3*A*(XB[1]-XB[0]))
#     # return (np.sum(P1**2 + P1*P2 + P2**2) + (A*(XB[1]-XB[0])-N)*(PB[1]**2+PB[1]*PB[0]+PB[0]**2))/(3*A*(XB[1]-XB[0]))

# def deltaMehta3(E, EB:tuple):
#     """
#     ...

#     Source: http://home.ustc.edu.cn/~zegang/pic/Mehta-Random-Matrices.pdf
#     """

#     N = len(E)
#     A0 = N/(EB[1]-EB[0])
#     def func(indvars):  return _deltaMehta3AB(E, EB, *indvars)
#     sol  = minimize(func, x0=(A0,0))
#     a,b = sol.x
#     D3 = _deltaMehta3AB(E, EB, a, b)
#     return D3, a, b

# =================================================================================================
#    Level-Spacing Ratio PDF:
# =================================================================================================

def levelSpacingRatioPDF(ratio:float, beta:int=1):
    """
    This function returns the probability density on the ensemble's nearest level-spacing ratio,
    evaluated at `ratio`. The ensemble can be chosen from GOE, GUE, and GSE for `beta` = 1, 2, or
    4, respectively.

    Source: https://arxiv.org/pdf/1806.05958.pdf (Eq. 1)

    Parameters:
    ----------
    ratio :: float or float array
        The nearest level-spacing ratio(s).
    beta  :: 1, 2, or 4
        The parameter that determines the assumed ensemble. For GOE, GUE, and GSE, `beta` = 1, 2,
        or 4, respectively. The default is 1 (GOE).

    Returns:
    -------
    level_spacing_ratio_pdf :: float or float array
        The probability density (or densities) evaluated at the the provided level-spacing
        ratio(s).
    """
    
    if   beta == 1:
        C_beta = 27/8
    elif beta == 2:
        C_beta = 81*sqrt(3)/(4*pi)
    elif beta == 4:
        C_beta = 729*sqrt(3)/(4*pi)
    else:
        raise ValueError('"beta" can only be 1, 2, or 4.')
    level_spacing_ratio_pdf = C_beta * (ratio+ratio**2)**beta / (1+ratio+ratio**2)**(1+(3/2)*beta)
    return level_spacing_ratio_pdf