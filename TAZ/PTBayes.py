import numpy as np
from numpy import newaxis as NA
from scipy.stats import chi2

from TAZ.Theory import ReduceFactor
from TAZ import Reaction, Resonances

__doc__ = """
This module contains Bayes' update for the probabilistic distribution on the neutron widths (and
gamma widths if specified).
"""

def PTBayes(res:Resonances, reaction:Reaction, false_width_dist=None, prior=None, gamma_width_on:bool=False):
    """
    Performs a Bayesian update on the spingroup probabilities of resonances, based on Porter-Thomas
    Distribution on the neutron widths (and gamma widths if specified).

    Let `L` be the number of resonances and `G` be the number of (true) spingroups.

    Parameters:
    ----------
    res              :: Resonances
        The resonance data object.
    reaction         :: Reaction
        A Reaction object that holds the mean parameters for the reaction.
    false_width_dist :: function
        The PDF for the neutron widths of false resonances. If none are given, false widths are
        sampled from the joint neutron width PDF of all spingroups. Default is None.
    prior            :: float [L,G+1]
        Optional prior spingroup probabilities. Such probabilities may include information on
        statistical fits. However, the probabilities must be independent of the information
        provided by the width distributions. Default is None.
    gamma_width_on   :: bool
        Determines whether the gamma-width probabilities are calculated based on the theoretical
        distribution. Many RRR evaluations assume the gamma widths are more or less constant. This
        theory is controversial, which is why the gamma width distribution is not considered by
        default. Default is False.

    Returns:
    -------
    posterior        :: float [L,G+1]
        The posterior spingroup probabilities.
    log_likelihood   :: float
        Calculated log-likelihood.
    """

    # Error Checking:
    if type(res) is not Resonances:
        raise TypeError('The "res" argument must be a "Resonances" object.')
    if type(reaction) is not Reaction:
        raise TypeError('The "mean_param" argument must be a "Reaction" object.')
    
    # Setting prior:
    if prior == None:
        prob = reaction.lvl_dens_all / np.sum(reaction.lvl_dens_all)
        prior = np.tile(prob, (res.E.size,1))
    posterior = prior

    # Neutron widths:
    mult_factor = (reaction.nDOF/reaction.gn2m)[NA,:] * ReduceFactor(res.E, reaction.L, ac=reaction.ac,
                                                                                  mass_targ=reaction.targ.mass, mass_proj=reaction.proj.mass)
    posterior[:,:-1] *= mult_factor * chi2.pdf(mult_factor * res.Gn[:,NA], reaction.nDOF)

    # Gamma widths: (if gamma_width_on is True)
    if gamma_width_on:
        mult_factor = (reaction.gDOF/reaction.gg2m)[NA,:]
        gg2 = 2 * res.Gg
        posterior[:,:-1] *= mult_factor * chi2.pdf(mult_factor * gg2[:,NA], reaction.gDOF)

    # False distribution:
    if (reaction.false_dens != 0.0) and (false_width_dist is not None):
        posterior[:,-1] *= false_width_dist(res.E, res.Gn, res.Gg)
    else:
        posterior[:,-1] *= np.sum(posterior[:,:-1], axis=1) / np.sum(prob[:-1])

    # Normalization:
    total_probability = np.sum(posterior, axis=1)
    posterior /= total_probability[:,NA]

    # Log likelihood:
    log_likelihood = np.sum(np.log(total_probability))

    return posterior, log_likelihood