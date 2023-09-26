import math
import numpy as np
from scipy.special import gamma, gammainc, gammaincc, gammainccinv, erfc, erfcx, erfcinv
from scipy.stats import norm

# from joblib import Parallel, delayed
# from multiprocessing import Pool

# __doc__ = """
# This file contains all level-spacing and width distributions.
# """

# # =================================================================================================
# #    Level-Spacing Probability Distributions
# # =================================================================================================

# def _gamma_ratio(x):
#     'Gamma(x/2) / Gamma((x-1)/2)'
#     rpii = 1.0 / math.sqrt(math.pi)
#     if hasattr(x,'__iter__'):
#         r = np.zeros(len(x))
#         for idx, w in enumerate(x):
#             q = rpii
#             for i in range(3,int(w)):       q = (i-2) / (2*q)
#             r[idx] = q
#     else:
#         r = rpii
#         for i in range(3,int(x)+1):     r = (i-2) / (2*r)
#     return r

# def _high_order_variance(n):
#     '...'
#     a = (n**2 + 5*n + 2)/2
#     B =(_gamma_ratio(a+2) / (n+1))**2
#     return (a+1)/(2*B) - (n+1)**2

# def _high_order_level_spacing_pdf(X, n):
#     """
#     ...

#     Source: https://journals.aps.org/pre/pdf/10.1103/PhysRevE.60.5371
#     """
#     if n <= 15: # Lower n --> Exact Calculation
#         a = (n**2 + 5*n + 2)/2
#         rB = _gamma_ratio(a+2) / (n+1)
#         C = 2 * rB / gamma((a+1)/2)
#         rBX = rB * X
#         return C * rBX**a * np.exp(-rBX**2)
#     else: # Higher n --> Gaussian Approximation:
#         sig = np.sqrt(_high_order_variance(n))
#         return norm.pdf(X, n+1, sig)

# def level_spacing_pdf(X, Freq:float=1.0, w:float=1.0, pM:float=0.0, err:float=5e-3):
#     """
#     ...
#     """
#     if pM == 0.0:    # "True" Level-spacings
#         if w == 1.0: # Wigner PDF
#             c = (math.pi/4) * Freq*Freq
#             Y = (2*c) * X * np.exp(-c * X*X)
#         else:        # Brody PDF
#             w1i = 1 / (w + 1)
#             a = (w1i * gamma(w1i) * Freq) ** (w+1)
#             Xw = X**w
#             Y = (w+1) * a * Xw * np.exp(-a * Xw*X)
#     else:            # Missing Level-spacings
#         if w == 1.0:
#             N_max = math.ceil(math.log(err, pM))
#             Y = Freq * (1-pM) * np.sum([pM**k * _high_order_level_spacing_pdf(X*Freq,k) for k in range(N_max+1)], axis=0)
#             Y /= 1 - pM**(N_max+1) # Renormalize to allieviate error
#         else:
#             raise NotImplementedError('Brody distribution with missing resonances has not been implemented yet')
#     return Y

# # =================================================================================================
# #    Width Probability Distributions
# # =================================================================================================

# def Porter_Thomas_pdf(G, Gm, trunc:float=0.0, DOF:int=1):
#     """
#     ...
#     """
#     if trunc == 0.0:
#         prob = DOF*(DOF*G/(2*Gm))**(DOF/2-1) * np.exp(-DOF*G/(2*Gm)) / (2 * Gm * gamma(DOF/2))
#     else:
#         prob = np.zeros(len(G))
#         prob[G >  trunc] = DOF*(DOF*G[G > trunc]/(2*Gm))**(DOF/2-1) * np.exp(-DOF*G[G > trunc]/(2*Gm)) / (2 * Gm * gamma(DOF/2) * gammaincc(DOF/2, DOF*trunc/(2*Gm)))
#         prob[G <= trunc] = 0.0
#     return prob

# def Porter_Thomas_cdf(G, Gnm:float=1.0, trunc:float=0.0, DOF:int=1):
#     """
#     ...
#     """
#     if trunc == 0.0:
#         prob = gammainc(DOF/2, DOF*G/(2*Gnm))
#     else:
#         prob = np.zeros(len(G))
#         prob[G >  trunc] = (gammainc(DOF/2, DOF*G[G>=trunc]/(2*Gnm)) - gammainc(DOF/2, DOF*trunc/(2*Gnm))) / gammaincc(DOF/2, DOF*trunc/(2*Gnm))
#         prob[G <= trunc] = 0.0
#     return prob

# def FractionMissing(trunc:float, Gnm:float=1.0, DOF:int=1):
#     'Gives the fraction of missing resonances due to the truncation in neutron width.'
#     return gammainc(DOF/2, DOF*trunc/(2*Gnm))





def _gamma_ratio(x):
    'Gamma(x/2) / Gamma((x-1)/2)'
    rpii = 1.0 / math.sqrt(math.pi)
    if hasattr(x,'__iter__'):
        r = np.zeros(len(x))
        for idx, w in enumerate(x):
            q = rpii
            for i in range(3,int(w)):       q = (i-2) / (2*q)
            r[idx] = q
    else:
        r = rpii
        for i in range(3,int(x)+1):     r = (i-2) / (2*r)
    return r

def _high_order_variance(n):
    '...'
    a = (n**2 + 5*n + 2)/2
    B =(_gamma_ratio(a+2) / (n+1))**2
    return (a+1)/(2*B) - (n+1)**2

def _high_order_level_spacing_parts(X, n):
    """
    ...

    Source: https://journals.aps.org/pre/pdf/10.1103/PhysRevE.60.5371
    """
    if n <= 15: # Lower n --> Exact Calculation
        a = (n**2 + 5*n + 2)/2
        rB = _gamma_ratio(a+2) / (n+1)
        C = 2 * rB / gamma((a+1)/2)
        rBX  = rB * X
        F0 = C * rBX**a * np.exp(-rBX**2)
        F1 = (C/2) * gammaincc((a+1)/2, rBX**2)
        F2 = (C/2) * gammaincc((a+2)/2, rBX**2)
    else: # Higher n --> Gaussian Approximation:
        sig = np.sqrt(_high_order_variance(n))
        F0 = norm.pdf(X, n+1, sig)
        F1 = (1/2) * erfc((X+n+1)/(sig*math.sqrt(2)))
        F2 = sig**2*F0 + (X+1)*F1 - 1
    return F0, F1, F2

def missing_pdf(X, Freq:float=1, pM:float=0, err:float=5e-3):
    """
    ...
    """
    N_max = math.ceil(math.log(err, pM))
    Y0, Y1, Y2 = 0, 0, 0
    for k in range(N_max+1):
        F0, F1, F2 = _high_order_level_spacing_parts(X*Freq, k)
        Y0 += pM**k * F0
        Y1 += pM**k * F1
        Y2 += pM**k * F2

    multfact = Freq * (1-pM) / (1 - pM**(N_max+1))
    Y0 *= multfact
    Y1 *= multfact
    Y2 *= multfact
    return Y0, Y1, Y2


















# def _gamma_ratio(x):
#     'Gamma(x/2) / Gamma((x-1)/2)'
#     rpii = 1.0 / math.sqrt(math.pi)
#     if hasattr(x,'__iter__'):
#         r = np.zeros(len(x))
#         for idx, w in enumerate(x):
#             q = rpii
#             for i in range(3,int(w)):       q = (i-2) / (2*q)
#             r[idx] = q
#     else:
#         r = rpii
#         for i in range(3,int(x)+1):     r = (i-2) / (2*r)
#     return r

# def _high_order_variance(n):
#     a = (n**2 + 5*n + 2)/2
#     B =(_gamma_ratio(a+2) / (n+1))**2
#     return (a+1)/(2*B) - (n+1)**2

# def _high_order_level_spacing_parts(X, n, orders:tuple=(0,1,2)):
#     """
#     Generates the `n+1`-th nearest neighbor level-spacing distribution as determined by the
#     Gaussian Orthogonal Ensemble (GOE). The distribution is calculated at each value in the numpy
#     array, `X`. Each order in `orders` request the order-th-integrated level-spacing distribution.

#     Source: https://journals.aps.org/pre/pdf/10.1103/PhysRevE.60.5371
#     """
#     out = []
#     if n <= 15: # Lower n --> Exact Calculation
#         a = (n**2 + 5*n + 2)/2
#         rB = _gamma_ratio(a+2) / (n+1)
#         coef = 2 * rB / gamma((a+1)/2)
#         rBX  = rB * X
#         for order in orders:
#             if   order == 0:
#                 F0 = coef * rBX**a * np.exp(-rBX**2)
#                 out.append(F0)
#             elif order == 1:
#                 F1 = (coef/2) * gammaincc((a+1)/2, rBX**2)
#                 out.append(F1)
#             elif order == 2:
#                 F2 = (coef/2) * gammaincc((a+2)/2, rBX**2)
#                 out.append(F2)
#     else: # Higher n --> Gaussian Approximation:
#         sig = np.sqrt(_high_order_variance(n))
#         for order in orders:
#             if   order == 0:
#                 F0 = norm.pdf(X, n+1, sig)
#                 out.append(F0)
#             elif order == 1:
#                 F1 = (1/2) * erfc((X+n+1)/(sig*math.sqrt(2)))
#                 out.append(F1)
#             elif order == 2:
#                 if 0 not in orders:      F0 = norm.pdf(X, n+1, sig)
#                 if 1 not in orders:      F1 = (1/2) * erfc((X+n+1)/(sig*math.sqrt(2)))
#                 F2 = sig**2*F0 + (X+1)*F1 - 1
#                 out.append(F2)
#     return tuple(out)


# class Distribution:
#     """
#     A class for level-spacing distributions, their integrals and inverses. Such distributions
#     have been defined for Wigner distribution, Brody distribution, and the missing distribution.
#     """
#     def __init__(self, f0, f1=None, f2=None, parts=None, if1=None, if2=None, Freq=None):
#         """
#         ...
#         """
#         self.f0 = f0

#         if f1 is None:
#             raise NotImplementedError('Integration for f1 has not been implemented yet.')
#         else:
#             self.f1 = f1

#         if f2 is None:
#             raise NotImplementedError('Integration for f2 has not been implemented yet.')
#         else:
#             self.f2 = f2

#         if parts is None:
#             def parts(x):
#                 F0 = self.f0(x)
#                 F1 = self.f1(x)
#                 F2 = self.f2(x)
#                 return F2, F0/F1, F1/F2
#         else:
#             self.parts = parts

#         if if1 is None:
#             raise NotImplementedError('The inverse of f1 has not been implemented yet.')
#         else:
#             self.if1 = if1

#         if if2 is None:
#             raise NotImplementedError('The inverse of f2 has not been implemented yet.')
#         else:
#             self.if2 = if2

#         self.Freq = Freq

#     @property
#     def pdf(self):
#         return self.f0

#     def sample_f0(self, size=None, seed=None):
#         'Inverse CDF Sampling on f0.'
#         if seed is not None:
#             rng = np.random.default_rng()
#         else:
#             rng = np.random.default_rng(seed)
#         return self.if1(rng.random(size))
    
#     def sample_f1(self, size=None, seed=None):
#         'Inverse CDF Sampling on f1.'
#         if seed is not None:
#             rng = np.random.default_rng()
#         else:
#             rng = np.random.default_rng(seed)
#         return self.if2(rng.random(size))

#     @classmethod
#     def wigner(cls, Freq:float=1.0):
#         'Sample Wigner distribution.'
#         pid4  = math.pi/4
#         coef = pid4*Freq**2
#         root_coef = math.sqrt(coef)

#         def get_f0(X):
#             return (2*coef) * X * np.exp(-coef * X*X)
#         def get_f1(X):
#             return np.exp(-coef * X*X)
#         def get_f2(X):
#             return erfc(root_coef * X) / Freq
#         def get_parts(X):
#             fX = root_coef * X
#             R1 = 2 * root_coef * fX
#             R2 = Freq / erfcx(fX)       # "erfcx(x)" is "exp(x^2)*erfc(x)"
#             F2 = np.exp(-fX*fX) / R2    # Getting the "erfc(x)" from "erfcx(x)"
#             return F2, R1, R2
#         def get_if1(X):
#             return np.sqrt(-np.log(X)) / root_coef
#         def get_if2(X):
#             return erfcinv(Freq * X) / root_coef
#         return cls(f0=get_f0, f1=get_f1, f2=get_f2, parts=get_parts, if1=get_if1, if2=get_if2, Freq=Freq)

#     @classmethod
#     def brody(cls, Freq:float=1.0, w:float=0.0):
#         'Sample Brody distribution.'
#         w1i = 1.0 / (w+1)
#         a = (Freq*w1i*gamma(w1i))**(w+1)

#         def get_f0(X):
#             aXw = a*X**w
#             return (w+1) * aXw * np.exp(-aXw*X)
#         def get_f1(X):
#             return np.exp(-a*X**(w+1))
#         def get_f2(X):
#             return (w1i*a**(-w1i)) * gammaincc(w1i, a*X**(w+1))
#         def get_parts(X):
#             aXw = a*X**w
#             aXw1 = aXw*X
#             R1 = (w+1)*aXw
#             F2 = (w1i * a**(-w1i)) * gammaincc(w1i,aXw1)
#             R2 = np.exp(-aXw1) / F2
#             return F2, R1, R2
#         def get_if1(X):
#             return (-np.log(X) / a) ** w1i
#         def get_if2(X):
#             return gammainccinv(w1i, ((w+1)*a**w1i) * X)
#         return cls(f0=get_f0, f1=get_f1, f2=get_f2, parts=get_parts, if1=get_if1, if2=get_if2, Freq=Freq)
    
#     @classmethod
#     def missing(cls, Freq:float=1.0, pM:float=0.0, err:float=5e-3):
#         'Sample Wigner distribution with missing resonances considered.'
        
#         # If we assume there are no missing resonances, the PDF converges to Wigner:
#         if pM == 0.0:
#             print(RuntimeWarning('Warning: the "missing" distribution has a zero missing resonance fraction.'))
#             return cls.wigner(Freq)
        
#         N_max = math.ceil(math.log(err, pM))
#         coef = (pM**np.arange(N_max+1))[:,np.newaxis]
#         mult_fact = Freq * (1-pM) / (1 - pM**(N_max+1))
#         def get_f0(X):
#             func = lambda _n: _high_order_level_spacing_parts(Freq*X, _n, orders=(0,))
#             values = [func(n) for n in range(N_max+1)]
#             return mult_fact * np.sum(coef * np.array([v[0] for v in values]), axis=0)
#         def get_f1(X):
#             func = lambda _n: _high_order_level_spacing_parts(Freq*X, _n, orders=(1,))
#             values = [func(n) for n in range(N_max+1)]
#             return mult_fact * np.sum(coef * np.array([v[0] for v in values]), axis=0)
#         def get_f2(X):
#             func = lambda _n: _high_order_level_spacing_parts(Freq*X, _n, orders=(2,))
#             values = [func(n) for n in range(N_max+1)]
#             return mult_fact * np.sum(coef * np.array([v[0] for v in values]), axis=0)
#         def get_parts(X):
#             func = lambda _n: _high_order_level_spacing_parts(Freq*X, _n, orders=(0,1,2))
#             values = [func(n) for n in range(N_max+1)]
#             V0, V1, V2 = zip(*values)
#             F0 = mult_fact * np.sum(coef*V0, axis=0)
#             F1 = mult_fact * np.sum(coef*V1, axis=0)
#             F2 = mult_fact * np.sum(coef*V2, axis=0)
#             R1 = F0 / F1
#             R2 = F1 / F2
#             return F2, R1, R2
#         def get_if1(X):
#             raise NotImplementedError('Inverse Function for f1 has not been implemented yet.')
#         def get_if2(X):
#             raise NotImplementedError('Inverse Function for f2 has not been implemented yet.')
#         return cls(f0=get_f0, f1=get_f1, f2=get_f2, parts=get_parts, if1=get_if1, if2=get_if2, Freq=(1-pM)*Freq)

# class Distributions:
#     """
#     A class that collects multiple `Distribution` objects together.
#     """
#     def __init__(self, *distributions:Distribution):
#         'Initializing distributions'
#         self.distributions = distributions
    
#     # Functions and properties:
#     def f0(self, X):
#         if not hasattr(X, '__iter__'):
#             X = np.array([X])
#         return np.array([distr.f0(X)    for distr in self.distributions]).T
#     def f1(self, X):
#         if not hasattr(X, '__iter__'):
#             X = np.array([X])
#         return np.array([distr.f1(X)    for distr in self.distributions]).T
#     def f2(self, X):
#         if not hasattr(X, '__iter__'):
#             X = np.array([X])
#         return np.array([distr.f2(X)    for distr in self.distributions]).T
#     def if1(self, X):
#         if not hasattr(X, '__iter__'):
#             X = np.array([X])
#         return np.array([distr.if1(X)   for distr in self.distributions]).T
#     def if2(self, X):
#         if not hasattr(X, '__iter__'):
#             X = np.array([X])
#         return np.array([distr.if2(X)   for distr in self.distributions]).T
#     def parts(self, X):
#         if not hasattr(X, '__iter__'):
#             X = np.array([X])
#         parts = np.array([distr.parts(X) for distr in self.distributions]).transpose(2,0,1)
#         return parts[:,:,0], parts[:,:,1], parts[:,:,2]
#     @property
#     def Freq(self):
#         return np.array([distr.Freq for distr in self.distributions]).reshape(1,-1)

#     def __getitem__(self, indices):
#         if hasattr(indices, '__iter__'):
#             return self.__class__(*self.distributions[indices])
#         else:
#             return self.distributions[indices]
    
#     @property
#     def num_dists(self):
#         return len(self.distributions)

#     @classmethod
#     def wigner(cls, Freq):
#         'Sample Wigner distribution for each spingroup.'
#         Freq = Freq.reshape(-1,)
#         distributions = [Distribution.wigner(freq_g) for freq_g in Freq]
#         return cls(*distributions)
#     @classmethod
#     def brody(cls, Freq, w=None):
#         'Sample Brody distribution for each spingroup.'
#         G = len(Freq)
#         if w is None:
#             w = np.zeros((G,))
#         Freq = Freq.reshape(-1,)
#         w    = w.reshape(-1,)
#         distributions = [Distribution.brody(freq_g, w_g) for freq_g,w_g in zip(Freq,w)]
#         return cls(*distributions)
#     @classmethod
#     def missing(cls, Freq, pM=None, err:float=5e-3):
#         'Sample Missing distribution for each spingroup.'
#         G = len(Freq)
#         if pM is None:
#             pM = np.zeros((G,))
#         Freq = Freq.reshape(-1,)
#         pM   = pM.reshape(-1,)
#         distributions = [Distribution.missing(freq_g, pM_g, err) for freq_g,pM_g in zip(Freq,pM)]
#         return cls(*distributions)