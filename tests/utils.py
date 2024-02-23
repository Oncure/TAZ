import numpy as np
from scipy.stats import chisquare

__doc__ = """
A file containing utility functions for unit tests.
"""

def chi2_test(dist, data, num_bins:int):
    """
    Performs a Pearson's Chi-squared test with the provided distribution and data.
    """

    data_len = len(data)
    quantiles = np.linspace(0.0, 1.0, num_bins+1)
    with np.errstate(divide='ignore'):
        edges = dist.ppf(quantiles)
    obs_counts, edges = np.histogram(data, edges)
    exp_counts = (data_len / num_bins) * np.ones((num_bins,))
    chi2, p = chisquare(f_obs=obs_counts, f_exp=exp_counts)
    chi2_bar = chi2 / num_bins
    return chi2_bar, p

def chi2_uniform_test(data, num_bins:int):
    """
    Performs a Pearson's Chi-squared test on the data, assuming that the underlying distribution
    is uniform.
    """

    obs_counts, bin_edges = np.histogram(data, num_bins)
    exp_counts = (len(data)/num_bins) * np.ones((num_bins,))
    chi2, p = chisquare(f_obs=obs_counts, f_exp=exp_counts)
    chi2_bar = chi2 / num_bins
    return chi2_bar, p