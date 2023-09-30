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

    Inputs:
    ------
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

    Inputs:
    ------
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

def PTBayes(Res, MeanParam, FalseWidthDist=None, Prior=None, GammaWidthOn:bool=False):
    """
    Performs a Bayesian update on the spingroup probabilities of resonances, based on Porter-Thomas
    Distribution on the neutron widths (and gamma widths if specified).

    Let `L` be the number of resonances and `G` be the number of (true) spingroups.

    Inputs:
    ------
    Res            : Resonances
        The resonance data.
    MeanParam      : MeanParameters
        The mean parameters for the reaction.
    FalseWidthDist : function
        The PDF for the neutron widths of false resonances. If none are given, false widths are
        sampled from the joint neutron width PDF of all spingroups. Default is `None`.
    Prior          : float [L,G+1]
        Optional prior spingroup probabilities. Such probabilities may include information on
        statistical fits. However, the probabilities must be independent of the information
        provided by the width distributions. Default is `None`.
    GammaWidthOn   : bool
        Determines whether the gamma-width probabilities are calculated based on the theoretical
        distribution. Many RRR evaluations assume the gamma widths are more or less constant. This
        theory is controversial, which is why the gamma width distribution is not considered by
        default. Default is `None`.

    Returns:
    -------
    Posterior         : float [L,G+1]
        The posterior spingroup probabilities.
    Total_Probability : float [L]
        Likelihoods for each resonance. These likelihoods are used for the log-likelihoods.
    """
    
    if Prior == None:
        prob = MeanParam.FreqAll #/ np.sum(MeanParam.FreqAll)
        Prior = np.repeat(prob, repeats=Res.E.size, axis=0)
    Posterior = Prior

    mult_factor = (MeanParam.nDOF/MeanParam.Gnm) * ReduceFactor(Res.E, MeanParam.L, MeanParam.A, MeanParam.ac)
    Posterior[:,:-1] *= mult_factor * chi2.pdf(mult_factor * Res.Gn.reshape(-1,1), MeanParam.nDOF)

    if GammaWidthOn:
        mult_factor = MeanParam.gDOF/MeanParam.Ggm
        Posterior[:,:-1] *= mult_factor * chi2.pdf(mult_factor * Res.Gg.reshape(-1,1), MeanParam.gDOF)

    if (MeanParam.FreqF != 0.0) and (FalseWidthDist is not None):
        Posterior[:,-1] *= FalseWidthDist(Res.E, Res.Gn, Res.Gg)
    else:
        Posterior[:,-1] *= np.sum(Posterior[:,:-1], axis=1) / np.sum(prob[0,:-1])

    Total_Probability = np.sum(Posterior, axis=1)
    Posterior /= Total_Probability.reshape(-1,1)
    return Posterior, Total_Probability