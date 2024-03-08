from pandas import DataFrame
import numpy as np
from numpy import newaxis as NA

from TAZ.Theory import porter_thomas_dist
from TAZ import Reaction

__doc__ = """
This module contains Bayes' update for the probabilistic distribution on the neutron widths (and
gamma widths if specified).
"""

def PTBayes(resonances:DataFrame, reaction:Reaction, false_width_dist=None, prior=None, gamma_width_on:bool=False):
    """
    Performs a Bayesian update on the spingroup probabilities of resonances, based on Porter-Thomas
    Distribution on the neutron widths (and gamma widths if specified).

    Let `L` be the number of resonances and `G` be the number of (true) spingroups.

    Parameters
    ----------
    resonancs        : DataFrame
        The dataframe of resonance energies, widths, etc.
    reaction         : Reaction
        A Reaction object that holds the mean parameters for the reaction.
    false_width_dist : function, optional
        The PDF for the neutron widths of false resonances. If none are given, false widths are
        sampled from the joint neutron width PDF of all spingroups.
    prior            : float [L,G+1], optional
        Optional prior spingroup probabilities. Such probabilities may include information on
        statistical fits. However, the probabilities must be independent of the information
        provided by the width distributions.
    gamma_width_on   : bool
        Determines whether the gamma-width probabilities are calculated based on the theoretical
        distribution. Many RRR evaluations assume the gamma widths are more or less constant. This
        theory is controversial, which is why the gamma width distribution is not considered by
        default. Default is False.

    Returns
    -------
    posterior        : float [L,G+1]
        The posterior spingroup probabilities.
    log_likelihood   : float
        Calculated log-likelihood.
    """

    # Error Checking:
    if type(resonances) is not DataFrame:
        raise TypeError('The "resonances" argument must be a DataFrame')
    if type(reaction) is not Reaction:
        raise TypeError('The "mean_param" argument must be a "Reaction" object.')
    
    E  = resonances['E'].to_numpy()
    Gg = resonances['Gg'].to_numpy()
    Gn = resonances['Gn1'].to_numpy()
    
    # Setting prior:
    if prior == None:
        prob = reaction.lvl_dens_all / np.sum(reaction.lvl_dens_all)
        prior = np.tile(prob, (E.size,1))
    posterior = prior

    # Neutron widths:
    Gnms = reaction.Gnm
    for g, Gnm in enumerate(Gnms):
        posterior[:,g] *= porter_thomas_dist.pdf(Gn, mean=Gnm(E), df=reaction.nDOF[g], trunc=0.0)

    # Gamma widths: (if gamma_width_on is True)
    if gamma_width_on:
        for g, Gnm in enumerate(Gnms):
            posterior[:,g] *= porter_thomas_dist.pdf(Gg, mean=reaction.Ggm[g], df=reaction.gDOF[g], trunc=0.0)

    # False distribution:
    if (reaction.false_dens != 0.0) and (false_width_dist is not None):
        posterior[:,-1] *= false_width_dist(E, Gn, Gg)
    else:
        posterior[:,-1] *= np.sum(posterior[:,:-1], axis=1) / np.sum(prob[:-1])

    # Normalization:
    total_probability = np.sum(posterior, axis=1)
    posterior /= total_probability[:,NA]

    # Log likelihood:
    log_likelihood = np.sum(np.log(total_probability))

    return posterior, log_likelihood