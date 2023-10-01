import numpy as np
from scipy.stats import chi2

from .RMatrix import ReduceFactor
from . import MeanParameters, Resonances

__doc__ = """
This module contains Bayes' update for the probabilistic distribution on the neutron widths (and
gamma widths if specified).
"""

def PTBayes(res:Resonances, mean_params:MeanParameters, false_width_dist=None, prior=None, gamma_width_on:bool=False):
    """
    Performs a Bayesian update on the spingroup probabilities of resonances, based on Porter-Thomas
    Distribution on the neutron widths (and gamma widths if specified).

    Let `L` be the number of resonances and `G` be the number of (true) spingroups.

    Parameters:
    ----------
    res              :: Resonances
        The resonance data object.
    
    mean_params      :: MeanParameters
        The mean parameters for the reaction.
    
    false_width_dist :: function
        The PDF for the neutron widths of false resonances. If none are given, false widths are
        sampled from the joint neutron width PDF of all spingroups. Default is `None`.
    
    prior            :: float [L,G+1]
        Optional prior spingroup probabilities. Such probabilities may include information on
        statistical fits. However, the probabilities must be independent of the information
        provided by the width distributions. Default is `None`.
    
    gamma_width_on   :: bool
        Determines whether the gamma-width probabilities are calculated based on the theoretical
        distribution. Many RRR evaluations assume the gamma widths are more or less constant. This
        theory is controversial, which is why the gamma width distribution is not considered by
        default. Default is `None`.

    Returns:
    -------
    posterior        :: float [L,G+1]
        The posterior spingroup probabilities.

    log_likelihood   :: float
        Calculated log-likelihood.
    """

    # Error Checking:
    if type(res) is not Resonances:
        raise TypeError('The "Res" argument must be a "Resonances" object.')
    if type(mean_params) is not MeanParameters:
        raise TypeError('The "MeanParam" argument must be a "MeanParameters" object.')
    
    # Setting prior:
    if prior == None:
        prob = mean_params.FreqAll / np.sum(mean_params.FreqAll)
        prior = np.repeat(prob, repeats=res.E.size, axis=0)
    posterior = prior

    # Neutron widths:
    mult_factor = (mean_params.nDOF/mean_params.Gnm) * ReduceFactor(res.E, mean_params.L, mean_params.A, mean_params.ac)
    posterior[:,:-1] *= mult_factor * chi2.pdf(mult_factor * res.Gn.reshape(-1,1), mean_params.nDOF)

    # Gamma widths: (if gamma_width_on is True)
    if gamma_width_on:
        mult_factor = mean_params.gDOF/mean_params.Ggm
        posterior[:,:-1] *= mult_factor * chi2.pdf(mult_factor * res.Gg.reshape(-1,1), mean_params.gDOF)

    # False distribution:
    if (mean_params.FreqF != 0.0) and (false_width_dist is not None):
        posterior[:,-1] *= false_width_dist(res.E, res.Gn, res.Gg)
    else:
        posterior[:,-1] *= np.sum(posterior[:,:-1], axis=1) / np.sum(prob[0,:-1])

    # Normalization:
    total_probability = np.sum(posterior, axis=1)
    posterior /= total_probability.reshape(-1,1)

    # Log likelihood:
    log_likelihood = np.sum(np.log(total_probability))

    return posterior, log_likelihood