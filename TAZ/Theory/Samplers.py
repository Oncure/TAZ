from math import pi, sqrt
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal

from TAZ.Theory.WidthDists import ReduceFactor
from TAZ.Theory.LevelSpacingDists import WignerGen, BrodyGen
from TAZ.Theory.distributions import porter_thomas_dist, semicircle_dist

__doc__ = """
This module contains sampler codes for the neutron width, gamma (capture) width, and resonance
energies.
"""

# =================================================================================================
#    Partial Width Sampling:
# =================================================================================================

def SampleNeutronWidth(E, gn2m:float, dof:int, l:int, ac:float,
                       mass_targ:float, mass_proj:float,
                       rng=None, seed:int=None):
    """
    Samples neutron widths according to the chi-squared distribution.

    Parameters:
    ----------
    E         :: float [n]
        Resonance energies, where `n` is the number of resonances.

    gn2m      :: float
        Mean reduced neutron width.

    dof       :: int
        Chi-squared degrees of freedom.

    l         :: int
        Quantum angular momentum number for the spingroup.

    ac        :: float
        Nuclear radius of the target isotope.

    mass_targ :: float
        Mass of the target particle.

    mass_proj :: float
        Mass of the projectile particle.

    rng       :: default_rng
        A provided `default_rng`. Default is `None`.
    
    seed      :: int
        If no `rng` is provided, then a random number seed can be specified.

    Returns:
    -------
    Gn        :: float [n]
        Randomly sampled neutron widths, where `n` is the number of resonances.
    """

    if rng is None:
        rng = np.random.default_rng(seed)

    gn2 = porter_thomas_dist(mean=gn2m, df=dof, trunc=0.0).rvs((len(E),), rng)
    Gn = gn2 / ReduceFactor(np.array(E), l, ac=ac, mass_targ=mass_targ, mass_proj=mass_proj)
    return Gn

def SampleGammaWidth(L:int, gg2m:float, dof:int,
                     rng=None, seed:int=None):
    """
    Samples gamma (capture) widths according to the chi-squared distribution.

    Parameters:
    ----------
    L    :: int
        Number of gamma (capture) widths to sample.

    gg2m :: float
        Mean reduced gamma (capture) width.

    dof  :: int
        Chi-squared degrees of freedom.

    rng  :: default_rng
        A provided `default_rng`. Default is `None`.
    
    seed :: int
        If no `rng` is provided, then a random number seed can be specified.

    Returns:
    -------
    Gg   :: float [n]
        Randomly sampled gamma (capture) widths, where `n` is the number of resonances.
    """
    
    if rng is None:
        rng = np.random.default_rng(seed)

    gg2 = porter_thomas_dist(mean=gg2m, df=dof, trunc=0.0).rvs((L,), rng)
    Gg = 2 * gg2
    return Gg

# =================================================================================================
#    Energy Level Sampling:
# =================================================================================================

def sampleGEEigs(n:int, beta:int=1,
                 rng=None, seed:int=None):
    """
    Samples the eigenvalues of n by n Gaussian Ensemble random matrices efficiently using the
    tridiagonal representation. The time complexity of this method is `O( n**2 )` using scipy's
    `eigvalsh_tridiagonal` function. However, there exist `O( n log(n) )` algorithms that have more
    low `n` cost and higher error. Unfortunately, no implementation of that algorithm has been made
    in Python.

    Source: https://people.math.wisc.edu/~valko/courses/833/2009f/lec_8_9.pdf

    Parameters:
    ----------
    n    :: int
        The rank of the random matrix. This is also the number of eigenvalues to sample.
    
    beta :: 1, 2, or 4
        The ensemble to consider, corresponding to GOE, GUE, and GSE respectively.

    rng  :: default_rng
        A provided `default_rng`. Default is `None`.
    
    seed :: int
        If no `rng` is provided, then a random number seed can be specified.

    Returns:
    -------
    eigs :: float [n]
        The eigenvalues of the random matrix.
    """

    if rng is None:
        rng = np.random.default_rng(seed)

    # Tridiagonal Matrix Coefficients:
    # Using Householder transformations (orthogonal transformations), we can
    # transform GOE-sampled matrices into a tridiagonal symmetric matrix.
    # Instead of performing the transformation, we can sample the transformed matrix directly.
    # Let `a` be the central diagonal elements and `b` be the offdiagonal diagonal elements.
    # We can define the following:
    a = sqrt(2) * rng.normal(size=(n,))
    b = np.sqrt(rng.chisquare(beta*np.arange(1,n)))

    # Now we sample the eigenvalues of the tridiagonal symmetric matrix:
    eigs = eigvalsh_tridiagonal(a, b)
    eigs /= sqrt(beta)
    eigs.sort()
    return eigs

def sampleGEEnergies(EB:tuple, lvl_dens:float=1.0, beta:int=1,
                     rng=None, seed:int=None):
    """
    Samples GOE (β = 1), GUE (β = 2), or GSE (β = 4) resonance energies within a given energy
    range, `EB` and with a specified mean level-density, `lvl_dens`.

    Parameters:
    ----------
    EB       :: float [2]
        The energy range for sampling.

    lvl_dens :: float
        The mean level-density.
    
    beta     :: 1, 2, or 4
        The ensemble parameter, where β = 1 is GOE, β = 2 is GUE, and β = 4 is GSE.

    rng      :: default_rng
        A provided `default_rng`. Default is `None`.
    
    seed     :: int
        If no `rng` is provided, then a random number seed can be specified.

    Returns:
    -------
    E    :: float [n]
        The sampled resonance energies, where `n` is the number of resonances.
    """

    if rng is None:
        rng = np.random.default_rng(seed)

    margin = 0.1 # a margin of safety where we consider the GOE samples to properly follow the semicircle law. This removes the uncooperative tails.
    N_res_est = (EB[1]-EB[0]) * lvl_dens # estimate number of resonances
    N_Tot = round((1 + 2*margin) * N_res_est) # buffer number of resonances

    eigs = sampleGEEigs(N_Tot, beta=beta, rng=rng)
    eigs /= 2*sqrt(N_Tot)
    eigs = eigs[eigs > -1.0+margin]
    # eigs = eigs[eigs <  1.0-margin]

    # Using semicircle law CDF to make the resonances uniformly spaced:
    # Source: https://github.com/LLNL/fudge/blob/master/brownies/BNL/restools/level_generator.py
    E = EB[0] + (N_Tot / lvl_dens) * (semicircle_dist.cdf(eigs) - semicircle_dist.cdf(-1.0+margin))
    E = E[E < EB[1]]
    E = np.sort(E)
    return E

def sampleNNEEnergies(EB:tuple, lvl_dens:float, w:float=1.0, rng=None, seed:int=None):
    """
    Sampler for the resonance energies according to the Nearest Neighbor Ensemble.

    Parameters:
    ----------
    EB       :: float [2]
        The energy range for sampling.

    lvl_dens :: float
        The mean level-density.

    w        :: float
        The brody parameter. Default is 1.0, giving a Wigner distribution.

    rng      :: default_rng
        A provided `default_rng`. Default is `None`.
    
    seed     :: int
        If no `rng` is provided, then a random number seed can be specified.

    Returns:
    -------
    E        :: float [n]
        Sampled resonance energies, where `n` is the number of resonances.
    """

    MULTIPLIER = 5 # a multiplication factor for conservative estimate of the number of resonances

    if rng is None:
        rng = np.random.default_rng(seed)

    L_Guess = round( lvl_dens * (EB[1] - EB[0]) * MULTIPLIER )
    LS = np.zeros(L_Guess+1, dtype='f8')
    if w == 1.0:
        distribution = WignerGen(lvl_dens=lvl_dens)
    else:
        distribution = BrodyGen(lvl_dens=lvl_dens, w=w)
    LS[0]  = EB[0] + distribution.sample_f1(rng=rng)
    LS[1:] = distribution.sample_f0(size=(L_Guess,), rng=rng)
    E = np.cumsum(LS)
    E = E[E < EB[1]]
    return E

def SampleEnergies(EB:tuple, lvl_dens:float, w:float=1.0, ensemble:str='NNE',
                   rng=None, seed:int=None):
    """
    Sampler for the resonance energies according to the selected ensemble.

    Parameters:
    ----------
    EB       :: float [2]
        The energy range for sampling.

    lvl_dens :: float
        The mean level-density.

    w        :: float
        The brody parameter. Default is 1.0, giving a Wigner distribution.

    ensemble :: NNE, GOE, GUE, GSE, or Poisson
        The level-spacing distribution to sample from:
        NNE     : Nearest Neighbor Ensemble
        GOE     : Gaussian Orthogonal Ensemble
        GUE     : Gaussian Unitary Ensemble
        GSE     : Gaussian Symplectic Ensemble
        Poisson : Poisson Ensemble

    rng      :: default_rng
        A provided `default_rng`. Default is `None`.
    
    seed     :: int
        If no `rng` is provided, then a random number seed can be specified.

    Returns:
    -------
    E        :: float [n]
        Sampled resonance energies, where `n` is the number of resonances.
    """

    MULTIPLIER = 5 # a multiplication factor for conservative estimate of the number of resonances
    
    if rng is None:
        rng = np.random.default_rng(seed)

    # Error Checking:
    if (ensemble in ('GOE','GUE','GSE')) and (w != 1.0):
        raise NotImplementedError(f'Cannot sample "{ensemble}" with Brody parameters')

    # Sampling based on ensemble:
    if   ensemble == 'NNE': # Nearest Neighbor Ensemble
        E = sampleNNEEnergies(EB, lvl_dens, w=w, rng=rng)
    elif ensemble == 'GOE': # Gaussian Orthogonal Ensemble
        E = sampleGEEnergies(EB, lvl_dens, beta=1, rng=rng)
    elif ensemble == 'GUE': # Gaussian Unitary Ensemble
        E = sampleGEEnergies(EB, lvl_dens, beta=2, rng=rng)
    elif ensemble == 'GSE': # Gaussian Symplectic Ensemble
        E = sampleGEEnergies(EB, lvl_dens, beta=4, rng=rng)
    elif ensemble == 'Poisson': # Poisson Ensemble
        num_samples = rng.poisson(lvl_dens * (EB[1]-EB[0]))
        E = rng.uniform(*EB, size=num_samples)
    else:
        raise NotImplementedError(f'The {ensemble} ensemble has not been implemented yet.')
    E.sort()
    return E