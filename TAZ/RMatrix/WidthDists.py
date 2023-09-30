import numpy as np
from scipy.special import gammainc
from scipy.stats import chi2

from .RMatrix import PenetrationFactor, Rho

__doc__ = """
This module is for partial width probability distributions.
"""

# =================================================================================================
#    Width Probability Distributions
# =================================================================================================

def FractionMissing(trunc:float, Gm:float=1.0, dof:int=1):
    """
    Gives the fraction of missing resonances due to the truncation in neutron width.
    """
    return gammainc(dof/2, dof*trunc/(2*Gm))

def PorterThomasPDF(G, Gm:float, trunc:float=0.0, dof:int=1):
    """
    The probability density function (PDF) for Porter-Thomas Distribution on the width. There is
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
        prob[G >  trunc] = chi2.pdf(G[G > trunc], df=dof, scale=Gm/dof) / (1 - FractionMissing(trunc, Gm, dof))
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
    
def ReduceFactor(E, l:float, A:float, ac:float):
    """
    Multiplication factor to convert from neutron width to reduced neutron width.
    """

    rho = Rho(A, ac, E)
    return 1.0 / (2.0*PenetrationFactor(rho,l))