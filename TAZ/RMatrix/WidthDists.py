import numpy as np
from scipy.special import gammainc, iv
from scipy.stats import chi2

from ..DataClasses import MASS_NEUTRON
from .RMatrix import PenetrationFactor, Rho

__doc__ = """
This module contains partial width probability distributions.
"""

# =================================================================================================
#    Width Probability Distributions
# =================================================================================================

def FractionMissing(trunc:float, Gnm:float=1.0, dof:int=1):
    """
    Gives the fraction of missing resonances due to the truncation in neutron width.

    Parameters:
    ----------
    trunc :: float
        The lower limit on the reduced neutron width.
    Gnm   :: float
        The mean reduced neutron width. Default = 1.0.
    dof   :: int
        The number of degrees of freedom for the chi-squared distribution.

    Returns:
    -------
    fraction_missing :: float
        The fraction of missing resonances within the spingroup.
    """
    fraction_missing = gammainc(dof/2, dof*trunc/(2*Gnm))
    return fraction_missing

def PorterThomasPDF(G, Gm:float, trunc:float=0.0, dof:int=1):
    """
    The probability density function (PDF) for Porter-Thomas distribution on the width. There is
    an additional width truncation factor, `trunc`, that ignores widths below the truncation
    threshold (meant for missing resonances hidden in the statistical noise).

    Parameters:
    ----------
    G     :: float [n]
        Partial resonance widths.
    Gm    :: float
        Mean partial resonance width.
    trunc :: float
        Width truncation factor for the distribution. Resonance widths below `trunc` are not
        considered in the distribution. Default = 0.0 (no truncation).
    dof   :: int
        Chi-squared degrees of freedom for the partial widths, `G`.

    Returns:
    -------
    prob  :: float [n]
        Probability density at the specified partial resonance widths.
    """
    
    if trunc == 0.0:
        prob = chi2.pdf(G, df=dof, scale=Gm/dof)
    else:
        prob = np.zeros(len(G))
        fraction_missing = FractionMissing(trunc, Gm, dof)
        prob[G >  trunc] = chi2.pdf(G[G > trunc], df=dof, scale=Gm/dof) / (1 - fraction_missing)
        prob[G <= trunc] = 0.0
    return prob

def PorterThomasCDF(G, Gm:float=1.0, trunc:float=0.0, dof:int=1):
    """
    The cumulative density function (CDF) for Porter-Thomas Distribution on the width. There is
    an additional width truncation factor, `trunc`, that ignores widths below the truncation
    threshold (meant for missing resonances hidden in the statistical noise).

    Parameters:
    ----------
    G     :: float [n]
        Partial resonance widths.
    Gm    :: float
        Mean partial resonance width.
    trunc :: float
        Width truncation factor for the distribution. Resonance widths below `trunc` are not
        considered in the distribution. Default = 0.0 (no truncation).
    dof   :: int
        Chi-squared degrees of freedom for the partial widths.

    Returns:
    -------
    prob  :: float [n]
        Cumulative probability at the specified partial resonance widths, `G`.
    """
    
    if trunc == 0.0:
        prob = chi2.cdf(G, df=dof, scale=Gm/dof)
    else:
        fraction_missing = FractionMissing(trunc, Gm, dof)
        prob = np.zeros(len(G))
        prob[G >  trunc] = (chi2.cdf(G, df=dof, scale=Gm/dof) - fraction_missing) / (1 - fraction_missing)
        prob[G <= trunc] = 0.0
    return prob
    
def ReduceFactor(E, l:int, mass_targ:float, ac:float,
                 mass_proj:float=MASS_NEUTRON,
                 mass_targ_after:float=None,
                 mass_proj_after:float=None,
                 E_thres:float=None):
    """
    Multiplication factor to convert from neutron width to reduced neutron width.

    Parameters:
    ----------
    E               :: float, array-like
        Resonance energies.
    l               :: int, array-like
        Orbital angular momentum number.
    mass_proj       :: float
        Mass of the projectile. Default = 1.008665 amu (neutron mass).
    mass_targ_after :: float
        Mass of the target after the reaction. Default = mass_targ.
    mass_proj_after :: float
        Mass of the target before the reaction. Default = mass_proj.
    E_thres         :: float
        Threshold energy for the reaction. Default is calculated from Q-value.

    Returns:
    -------
    reduce_factor :: float, array-like
        A multiplication factor that converts neutron widths into reduced neutron widths.
    """

    rho = Rho(mass_targ, ac, E,
              mass_proj, mass_targ_after, mass_proj_after, E_thres)
    reduce_factor = 1.0 / (2.0*PenetrationFactor(rho,l))
    return reduce_factor

def FissionDistPDF(G,
                   GmfA:float     , dofA:int=1,
                   GmfB:float=None, dofB:int=1,
                   trunc:float=0.0):
    """
    ...
    """

    if GmfB == None:
        return PorterThomasPDF(G, GmfA, trunc, dofA)
    else:
        if trunc != 0.0:
            raise NotImplementedError('Truncation has not been implemented yet.')
        elif not (dofA == dofB == 1):
            raise NotImplementedError('The fission distribution with more than 1 degree of freedom has not been implemented yet.')
        
        # Source:   https://www.publish.csiro.au/ph/pdf/PH750489
        a = G * (1/GmfB - 1/GmfA)/4
        b = G * (1/GmfB + 1/GmfA)/4
        prob = np.exp(-b) * iv(0, a) / np.sqrt(4*GmfA*GmfB)
        return prob