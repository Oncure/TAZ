from copy import copy
import math
import numpy as np
from numpy import newaxis as NA

import warnings

__doc__ = """
This file is contains the spingroup classification algorithm. `RunMaster` is used as a pre- and
post-processor for Encore.
"""

def shape(vec, g:int, G:int):
    """
    Shapes a provided 1-dimensional vector to a `G` dimensional array along axis `g`.

    Parameters
    ----------
    vec : array-like
        The vector to reshape.
    g : int
        The dimension to reshape into.
    G : int
        The number of dimensions.
    
    Returns
    -------
    array : array-like
        The G-dimensional array reshaped from array
    """

    shape_ = [1]*G
    shape_[g] = -1
    array = vec.reshape(*shape_)
    return array

class Encore:
    """
    A class spingroup classification. This class has many features:

    1. WigBayes: spingroup probabilities for each resonance, given the resonance ladder and mean
    parameters.

    2. WigSample: random sample of possible spingroup assignments. Likelihood of ladder sample
    correlates with the ladder spingroup assignment probability.
    
    3. ProbOfSample: probability of sampling the exact ladder of spingroup assignments.

    4. LogTotProb: log of the probability of sampling the resonance ladder regardless
    of spingroup, given the mean parameters. This is used for log-likelihood estimation for mean
    parameter estimation.

    5. WigMaxLikelihood: finds the maximum likelihood spingroup assignments.

    Let "L" be the number of resonances in the ladder and "G" be the number of spingroups
    (excuding the false group).

    Initialization Variables
    ------------------------
    prior : float [L+2, G+1]
        Prior probabilities on each resonance (with boundaries) and spingroup (with false group).

    level_spacing_probs : float [L+2, L+2, G]
        All of the level spacings from position index 0 to index 1 with spingroup, index 2.

    iMax : int [L+2, 2, G]
        Maximum/minimum index for approximation threshold for each resonance (index 0) in each
        direction (index 1) for each spingroup (index 2).
    
    Internally Used Variables
    -------------------------
    CPL : float [L+2, ..., L+2]
        'Cumulative Probabilities' for left-to-right propagation which are the building blocks
        for Bayesian probability calculations and probabilistic sampling.
        
    CPR : float [L+2, ..., L+2]
        'Cumulative Probabilities' for right-to-left propagation which are the building blocks
        for Bayesian probability calculations and probabilistic sampling.

    PW : float [L+2]
        'Placement Weights' which ensure CP does not grow or decay exponententially beyond the
        limits of floating-point numbers. PW is effectively a list of probability normalization
        factors for each resonance.

    TP : float
        'Total Probability' which is the probability density for the current arrangements of
        resonance energies (with PW removed for floating point limitation reasons).
    """

    # Error and debugging text:
    PW_INF_ERROR = """
Infinite placement weight ("PW") found. This implies a local probability of zero.
This should not happen unless the floating point limitations have been exceeded.

DEBUGGING VALUES:
Direction: {side}
Index:     {idx}
"""

    CP_BAD_ERROR = 'A cumulative probability ("CP") has a "{}" value.'

    TP_ERROR_THRESHOLD = 1.0 # % (percent)

    # ==================================================================================
    # Constructor
    # ==================================================================================
    def __init__(self, prior, level_spacing_probs, iMax):
        """
        Initializes Encore.

        Let `G` be the number of spingroups, and let `L` be the number of resonances.

        Parameters
        ----------
        prior               : float [L+2, G+1]
            Prior probabilities on each resonance (with boundaries) and spingroup (with false group).
        level_spacing_probs : float [L+2, L+2, G]
            All of the level spacings from position index 0 to index 1 with spingroup, index 2.
        iMax                : int [L+2, 2, G]
            Maximum/minimum index for approximation threshold for each resonance (index 0) in each
            direction (index 1) for each spingroup (index 2).
        """

        # Initialization:
        self.G = level_spacing_probs.shape[2]
        self.L = np.int32(prior.shape[0])
        self.Prior = np.ones((self.L+2,self.G+1), dtype='f8')
        self.Prior[1:-1,:] = prior
        self.LSP = level_spacing_probs
        self.iMax = iMax

        # Calculating "CP" and "TP":
        self.__makeCP()
        self.__makeTP()

    # ==================================================================================
    # Make CP
    # ==================================================================================
    def __makeCP(s):
        """
        `__MakeCP` calculates `CP` -- the 'Cumulative Probabilities'. `CP` is a G+1 dimensional
        array that takes the form, `CP(i1,i2,...,iG,dir)`, where `i1, i2, ..., iG` are resonance
        indices for spingroups, `1, 2, ..., G`, and `dir` is the direction. `CP(i1,i2,...,iG,dir)`
        is the likelihood of all spingroup assignments, where `i1, i2, ..., iG` are the resonance
        indices of the latest occurance of spingroups `1, 2, ..., G` from the left if `dir = 0` or
        from the right if `dir = 1`.

        `CP` is calculated through an iterative process, hence the name. The true value of `CP`
        has the tendency to exponentially grow or decay through each iteration, therefore, we use
        'Placement Weights' (`PW`) to renormalize the value to an acceptable floating-point range.
        Without the `PW` normalization, the `CP` values will quickly exceed the domain of floating-
        point numbers.

        Let `G` be the number of spingroups, and let `L` be the number of resonances. `m` is the
        expected number of resonances before `xMax` is exceeded.

        Time complexity: O( G⋅m^G⋅L )
        """
            
        L = s.L;    G = s.G
        s.CPL = np.zeros([L+2]*G, dtype='f8')
        s.CPR = np.zeros([L+2]*G, dtype='f8')
        s.PW = np.zeros(L+2, dtype='f8')

        # Trivial CP values:
        zeros = [0]*G       ;   negs = [-1]*G
        s.CPL[*zeros] = 1.0 ;   s.CPR[*negs] = 1.0
        s.PW[0] = 1.0       ;   s.PW[-1] = 1.0

        np.seterr(divide='raise', over='raise', invalid='raise')

        # Leading index (left iteration):
        for lnL in range(1,L+2): # next lead index (left-to-right)
            lnR = L+1-lnL # next lead index (right-to-left)

            # =====================================================================================
            # Left to Right:

            iMax_all = np.max(s.iMax[lnL,1,:])
            mult_chain = 1.0
            for ll in range(lnL-1, iMax_all-1, -1): # last lead index
                mult_chain *= s.PW[ll]
                slices = [slice(s.iMax[lnL,1,g], max(ll,1)) for g in range(G)]
                for gl in range(G): # last lead index group
                    slicesl = (*slices[:gl], slice(ll,ll+1), *slices[gl+1:])
                    for gn in range(G): # next lead index group
                        slicesn = (*slicesl[:gn], slice(lnL,lnL+1), *slicesl[gn+1:])
                        lsp = s.LSP[slicesl[gn],lnL,gn]
                        lsp = shape(lsp, gn, G)
                        s.CPL[*slicesn] += mult_chain * s.Prior[ll,gl] * np.sum(lsp * s.CPL[*slicesl], axis=gn, keepdims=True)
                mult_chain *= s.Prior[ll,-1]
                if mult_chain == 0.0:
                    break   # potential computation time improvement

            # =====================================================================================
            # Right to Left:

            iMax_all = np.min(s.iMax[lnR,0,:])
            mult_chain = 1.0
            for ll in range(lnR+1, iMax_all+1): # last lead index
                mult_chain *= s.PW[ll]
                slices = [slice(s.iMax[lnR,0,g], min(ll,L), -1) for g in range(G)]
                for gl in range(G): # last lead index group
                    slicesl = (*slices[:gl], slice(ll,ll+1), *slices[gl+1:])
                    for gn in range(G): # next lead index group
                        slicesn = (*slicesl[:gn], slice(lnR,lnR+1), *slicesl[gn+1:])
                        lsp = s.LSP[lnR,slicesl[gn],gn]
                        lsp = shape(lsp, gn, G)
                        s.CPR[*slicesn] += mult_chain * s.Prior[ll,gl] * np.sum(lsp * s.CPR[*slicesl], axis=gn, keepdims=True)
                mult_chain *= s.Prior[ll,-1]
                if mult_chain == 0.0:
                    break   # potential computation time improvement

            # =====================================================================================
            # Finding PW:
                
            # Left-hand Side PW Calculation:
            if 2*lnL <= L+1:
                slices = [slice(s.iMax[lnL,1,g], max(lnL,1)) for g in range(G)]
                denom = 0.0
                for gn in range(G):
                    slicesn = (*slices[:gn], slice(lnL,lnL+1), *slices[gn+1:])
                    denom += np.sum(s.CPL[*slicesn])
                if denom == 0.0:
                    raise RuntimeError(s.PW_INF_ERROR.format(side='left', idx=lnL))
                s.PW[lnL] = 1.0 / denom
                
            # Right-hand Side PW Calculation:
            if 2*lnL <= L:
                slices = [slice(s.iMax[lnR,0,g], min(lnR,L), -1) for g in range(G)]
                denom = 0.0
                for gn in range(G):
                    slicesn = (*slices[:gn], slice(lnR,lnR+1), *slices[gn+1:])
                    denom += np.sum(s.CPR[*slicesn])
                if denom == 0.0:
                    raise RuntimeError(s.PW_INF_ERROR.format(side='right', idx=lnR))
                s.PW[lnR] = 1.0 / denom

        # =====================================================================================
        # Finding end-point CPLs and CPRs:
        # needed because the "outside" resonance takes all spingroup assignments, not just one.
                
        lnL = L+1;      lnR = 0

        #                             the new part
        #                                  v
        slices = [slice(s.iMax[lnL,1,g], None) for g in range(G)]
        slicesl = (slice(lnL,lnL+1), *slices[1:])
        for gn in range(1, G):
            sliceg = slice(s.iMax[lnL,1,gn], lnL)
            slicesn = (*slicesl[:gn], slice(lnL,lnL+1), *slicesl[gn+1:])
            slicesnp = (*slicesl[:gn], sliceg, *slicesl[gn+1:])
            lsp = s.LSP[sliceg, lnL, gn]
            lsp = shape(lsp, gn, G)
            s.CPL[*slicesn] += np.sum(lsp * s.CPL[*slicesnp], axis=gn, keepdims=True)
        s.CPL[*negs] *= G # I don't understand this correction factor

        #                             the new part
        #                                  v
        slices = [slice(s.iMax[lnR,0,g], None, -1) for g in range(G)]
        slicesl = (slice(lnR,lnR+1), *slices[1:])
        for gn in range(1, G):
            sliceg = slice(s.iMax[lnR,0,gn], lnR, -1)
            slicesn = (*slicesl[:gn], slice(lnR,lnR+1), *slicesl[gn+1:])
            slicesnp = (*slicesl[:gn], sliceg, *slicesl[gn+1:])
            lsp = s.LSP[lnR, sliceg, gn]
            lsp = shape(lsp, gn, G)
            s.CPR[*slicesn] += np.sum(lsp * s.CPR[*slicesnp], axis=gn, keepdims=True)
        s.CPR[*zeros] *= G # I don't understand this correction factor

        # # =====================================================================================
        # # Finding end-point CPLs and CPRs:
        # # needed because the "outside" resonance takes all spingroup assignments, not just one.
                
        # lnL = L+1;      lnR = 0

        # #                             the new part
        # #                                  v
        # slicesw = [slice(s.iMax[lnL,1,g], None) for g in range(G)]
        # sliceso = [slice(s.iMax[lnL,1,g], lnL) for g in range(G)]
        # for gl in range(G):
        #     slicesl = (*sliceso[:gl], slice(lnL,lnL+1), *slicesw[gl+1:])
        #     for gn in range(gl+1, G):
        #         sliceg = slice(s.iMax[lnL,1,gn], lnL)
        #         slicesn = (*slicesl[:gn], slice(lnL,lnL+1), *slicesl[gn+1:])
        #         slicesnp = (*slicesl[:gn], sliceg, *slicesl[gn+1:])
        #         lsp = s.LSP[sliceg, lnL, gn]
        #         lsp = shape(lsp, gn, G)
        #         s.CPL[*slicesn] += np.sum(lsp * s.CPL[*slicesnp], axis=gn, keepdims=True)

        # #                             the new part
        # #                                  v
        # slicesw = [slice(s.iMax[lnR,0,g], None, -1) for g in range(G)]
        # sliceso = [slice(s.iMax[lnR,0,g], lnR, -1) for g in range(G)]
        # for gl in range(G):
        #     slicesl = (*sliceso[:gl], slice(lnR,lnR+1), *slicesw[gl+1:])
        #     for gn in range(gl+1, G):
        #         sliceg = slice(s.iMax[lnR,0,gn], lnR, -1)
        #         slicesn = (*slicesl[:gn], slice(lnR,lnR+1), *slicesl[gn+1:])
        #         slicesnp = (*slicesl[:gn], sliceg, *slicesl[gn+1:])
        #         lsp = s.LSP[lnR, sliceg, gn]
        #         lsp = shape(lsp, gn, G)
        #         s.CPR[*slicesn] += np.sum(lsp * s.CPR[*slicesnp], axis=gn, keepdims=True)

    # ==================================================================================
    # Find Total Probability Factor
    # ==================================================================================
    def __makeTP(s):
        """
        `__makeTP` is a method that calculates the total probability factor, `TP` using `CP`
        and `PW` found in the method, `__makeCP`. `TP` can be found by calculating
        `CPL[-1, ..., -1]` or `CPR[0, ..., 0]`. We accept the mean of the two calculation as the
        total probability and find the percent error between each for error-checking purposes.
        """

        # CPL[-1, ..., -1] = TPL and CPR[0, ..., 0] = TPR:
        negs = [-1]*s.G     ;   zeros = [0]*s.G
        TPL = s.CPL[*negs]  ;   TPR = s.CPR[*zeros]
        
        # Taking the average of the two sides for the total probability factor:
        s.TP = (TPL + TPR) / 2

        # If the two sides are past a threshold percent difference, raise a warning:
        percent_error = 100.0 * abs(TPL - TPR) / s.TP
        if percent_error > s.TP_ERROR_THRESHOLD:
            warnings.warn(f"""
In calculating the total probability ("TP"), there was a percent error of {percent_error:.3f}%.
This exceeds the set limit of {s.TP_ERROR_THRESHOLD:.3f}%. This error could be attributed to a bug or
numerical instability.
""", RuntimeWarning)

    # ==================================================================================
    # Level Spacing Bayes Theorem
    # ==================================================================================
    def WigBayes(s):
        """
        `WigBayes` finds the probability on each spingroup assignment of each resonance, informed
        by the level-spacing probability distributions. This is performed using the calculated `CP`
        and `TP` probabilities and Bayes' Theorem.

        Let `G` be the number of spingroups, and let `L` be the number of resonances. `m` is the
        expected number of resonances before `xMax` is exceeded.

        Time complexity: O( G⋅m^(2G-2)⋅L )

        Returns
        -------
        posterior : float [L,G+1]
            The posterior probabilities.
        """

        L = s.L; G = s.G
        sp = np.zeros((L,G), dtype='f8') # No longer need the edge cases, so the bounds are 0:L instead of 0:L+2

        for i in range(1,L+1):
            NAs = [NA]*G
            NAs2 = [NA]*(2*G-2)
            slicesL = [slice(s.iMax[i,1,g], max(i,1)    ) for g in range(G)]
            slicesR = [slice(s.iMax[i,0,g], min(i,L), -1) for g in range(G)]
            for gt in range(G):
                lsps = np.array(1.0, ndmin=2*G)
                for g in range(G):
                    if g == gt:     continue
                    lspsg = s.LSP[slicesL[g],slicesR[g],g]
                    Dims = np.arange(2, 2*G)
                    Dims = np.insert(Dims,   g, 0)
                    Dims = np.insert(Dims, G+g, 1)
                    lsps = lsps * lspsg[:,:,*NAs2].transpose(*Dims)
                slicesiL = (*slicesL[:gt], slice(i,i+1), *slicesL[gt+1:])
                slicesiR = (*slicesR[:gt], slice(i,i+1), *slicesR[gt+1:])
                sp[i-1,gt] = s.Prior[i,gt] * s.PW[i] * np.sum(lsps * s.CPL[*slicesiL,*NAs] * s.CPR[*NAs,*slicesiR])

        # Dividing by total probability, "TP":
        sp /= s.TP

        # False probabilities appended by taking one minus the other probabilities:
        posterior = np.append(sp, (1.0 - np.sum(sp, axis=1))[:,NA], 1)
        return posterior

    # ==================================================================================
    # Sample Spacing-Updated Spin-Groups
    # ==================================================================================
    def WigSample(s, num_trials:int=1,
                  rng:np.random.Generator=None, seed:int=None):
        """
        `WigSample` randomly samples spingroup assignments based its Bayesian probability. This is
        performed by sampling spingroup assignments for each resonance one at a time from low-to-
        high energy, indexed by `i`. Using the lower energy resonaces with determined spingroup
        assignments and the `CP` probabilities, representing the possibilities for the the higher
        energy resonances, we can get a probability on resonance `i` for each spingroup.

        Let `G` be the number of spingroups, `L` be the number of resonances and `T` be the number
        of trials. `m` is the expected number of resonances before `xMax` is exceeded.

        Time complexity: O( T⋅m^(G-1)⋅L )

        Parameters
        ----------
        num_trials : int
            The number of assignment ensembles to return. Default = 1.
        rng : np.random.Generator, optional
            The random number generator.
        seed : int, optional
            The random number seed if `rng` is not provided.

        Returns
        -------
        sampled_groups : int [L,T]
            The sampled spingroup assignments.
        """
        
        # Setting random number generator:
        if rng is None:
            rng = np.random.default_rng(seed)
        
        L = s.L;    G = s.G
        sampled_groups = G * np.ones((L,num_trials), dtype='u1') # the sampled spingroups (default as false)

        for tr in range(num_trials):
            last_seen = np.zeros((G,), dtype='u4') # last seen indices for each spingroup, for each trial
            last_seen_all = 0
            while True: # iterate procedure until the resonance ladder is filled
                iMax = [s.iMax[last_seen[g],0,g] for g in range(G)]
                iMaxmin = np.min(iMax)
                likelihoods = np.zeros((iMaxmin-last_seen_all,G), dtype='f8')
                mult_chain = 1.0
                lls = np.arange(last_seen_all+1, iMaxmin+1, dtype='i4')
                for lidx, ll in enumerate(lls): # last leader
                    mult_chain *= s.PW[ll]
                    if mult_chain == 0.0:   break   # potential computation time improvement
                    slices = [slice(ll+1, iMax[g]+1) for g in range(G)]
                    for gn in range(G): # next leader group
                        slicesn = (*slices[:gn], slice(ll, ll+1), *slices[gn+1:])
                        lsps = 1.0
                        for g in range(G):
                            lsps = lsps * shape(s.LSP[last_seen[g],slicesn[g],g], g, G)
                        likelihoods[lidx,gn] = s.LSP[last_seen[gn],ll,0] * s.Prior[ll,gn] * mult_chain \
                                                * np.sum(lsps * s.CPR[*slicesn])
                    mult_chain *= s.Prior[ll,G]
                prob_sgs = np.sum(likelihoods, axis=0) / np.sum(likelihoods)
                g = rng.choice(G, p=prob_sgs)
                sample_probs = likelihoods[:,g] / np.sum(likelihoods[:,g])
                last_seen[g] = rng.choice(lls, p=sample_probs)
                last_seen_all = last_seen[g]
                if last_seen_all == L+1:
                    break # ladder complete
                i_new = last_seen_all - 1
                sampled_groups[i_new,tr] = g
                if last_seen_all == L:
                    break # ladder complete

        return sampled_groups
    
    # ==================================================================================
    # Sample Probability
    # ==================================================================================
    def ProbOfSample(s, spingroup_assignments) -> float:
        """
        `ProbOfSample` finds the probability of sampling the given spingroup assignments.

        ...

        Parameters
        ----------
        spingroup_assignments : int [L]
            An array of spingroup assignments.
        
        Returns
        -------
        prob_of_sampling : float
            The probability of sampling the given spingroup assignments given the resonance ladder.
        """

        # NOTE: This needs checking!!!

        G = s.G

        likelihood = 1.0
        last_res = np.zeros((G,),'i4')
        for i, g in enumerate(spingroup_assignments):
            i1 = i + 1
            if g == G:
                likelihood *= s.Prior[i1,G] * s.PW[i1]
            else:
                j1 = last_res[g]
                likelihood *= s.Prior[i1,g] * s.PW[i1] * s.LSP[j1,i1,g]
                last_res[g] = i1

        # Right boundary level-spacings:
        for g in range(G):
            likelihood *= s.LSP[last_res[g],-1,g]
        
        # Normalize by total probability (TP):
        prob_of_sampling = likelihood / s.TP
        return prob_of_sampling

    # ==================================================================================
    # Log Likelihood
    # ==================================================================================
    def LogLikelihood(self, EB:tuple, lvl_dens_false:float=0.0, log_likelihood_prior:float=None) -> float:
        """
        `LogLikelihood` gives a criterion for the likelihood of sampling the given energies and
        widths, assuming correct information was given (i.e. mean level spacing, mean neutron
        width, etc.). This can be used with gradient descent to optimize mean parameters.

        Parameters
        ----------
        EB : tuple(float, float)
            The ladder energy limits.
        lvl_dens_false : float
            The false level-density. Default = 0.0.
        log_likelihood_prior : float, optional
            A log-likelihood provided by the prior.

        Returns
        -------
        log_likelihood : float
            The logarithm of the likelihood of sampling the given energies and widths.
        """

        dE = EB[1] - EB[0]
        log_likelihood = math.log(self.TP) - np.sum(np.log(self.PW)) - lvl_dens_false * dE

        # Prior log likelihood:
        if log_likelihood_prior is not None:
            log_likelihood += log_likelihood_prior

        return log_likelihood

    @staticmethod
    def LikelihoodString(log_likelihood:float, precision:int=3) -> str:
        """
        Prints the likelihood given the log likelihood. The likelihood sometimes cannot be
        given directly since the exponential term could exceed floating-point limitations;
        for this reason, likelihood is given by a custom scientific notation.

        Parameters
        ----------
        log_likelihood : float
            The logarithm of the likelihood of sampling the given energies and widths.
        precision : int
            The number of digits after the decimal point to report. Default = 3.
        
        Returns
        -------
        likelihood_str : str
            The likelihood as a string.
        """

        # Turn to into log10:
        log_likelihood /= math.log(10)

        likelihood_str = f'{{0:.{precision}f}}e{{1}}'
        exponent    = math.floor(log_likelihood)
        significand = 10 ** (log_likelihood % 1.0)
        return likelihood_str.format(significand, exponent)
    
    # ==================================================================================
    # Maximum-Likelihood Assignments
    # ==================================================================================
    @staticmethod
    def WigMaxLikelihood(prior, level_spacing_probs, iMax):
        """
        Returns the maximum likelihood spingroup assignments using branching and pruning methods.

        WigMaxLikelihood starts with the left-most (lowest-energy) resonance and branches into
        spingroup ladders for following resonances. When unrestricted, there are `G` additional
        ladders each iteration. Several pruning methods are applied to make this procedure
        computationally feasible. First, since level-spacing distributions only account for the
        nearest reasonances, only the right-most assigned resonances for each spingroup matter in
        calculating level-spacing likelihoods. Therefore, ladders with the same right-most
        assigned resonances for each spingroup will share the maximum likelihood ladder following
        the last assigned resonance. Therefore, only the ladder with the highest likelihood up to
        the last assigned resonances could be the maximum-likelihood ladder. All other ladders with
        matching right-most assigned resonances can be pruned. Additionally, if the resonances with
        the same spingroup become too large, the likelihood may be too low to be the maximum
        likelihood ladder.

        Let `G` be the number of spingroups, and let `L` be the number of resonances. `m` is the
        expected number of resonances before `xMax` is exceeded.

        Time complexity: O( G⋅m^G⋅L )

        Parameters
        ----------
        prior : float [L,G+1]
            The prior probabilities.
        level_spacing_probs : float [L+2,L+2,G]
            The level-spacing probability densities between each resonance for each spingroup.
        iMax : int [L+2,2,G]
            Maximum/minimum index for approximation threshold for each resonance (index 0) in each
            direction (index 1) for each spingroup (index 2).

        Returns
        -------
        max_spingroups : int [L]
            An array of spingroup assignment IDs with maximal likelihood.
        """

        L = level_spacing_probs.shape[0] - 2
        G = level_spacing_probs.shape[2]

        # Logification:
        with np.errstate(divide='ignore'):
            log_prior = np.log(prior)
            log_level_spacing_probs = np.log(level_spacing_probs)

        # Initialization:
        contenders = {} # list of spingroup assignments and likelihood, indexed by a G-tuple of last indices
        for g in range(G):
            spingroups = [g]
            log_likelihood = log_prior[0,g] + log_level_spacing_probs[0,1,g]
            last_indices = [0]*G
            last_indices[g] = 1
            contenders[tuple(last_indices)] = (spingroups, log_likelihood)
        spingroups = [G]
        log_likelihood = log_prior[0,G]
        contenders[tuple([0]*G)] = (spingroups, log_likelihood)

        # Loop for each additional resonance:
        for i in range(1,L):
            i1 = i + 1
            # Generate new contenders:
            new_contenders = {}
            for last_indices, (spingroups, log_likelihood) in contenders.items():
                # True Groups:
                for g in range(G):
                    new_last_indices = list(last_indices)
                    new_last_indices[g] = i1
                    new_last_indices = tuple(new_last_indices)
                    if np.any(new_last_indices < iMax[i1,1,:]):
                        continue # branch has forgotten about a spingroup. It is quite unlikely that this branch will be of maximal likelihood.
                    new_spingroups = spingroups + [g]
                    new_log_likelihood = log_likelihood + log_prior[i,g] + log_level_spacing_probs[last_indices[g],i1,g]
                    # Add assignment to the list:
                    if new_last_indices in new_contenders:
                        prev_log_likelihood = new_contenders[new_last_indices][1]
                        if new_log_likelihood > prev_log_likelihood:
                            new_contenders[new_last_indices] = (new_spingroups, new_log_likelihood)
                    else:
                        new_contenders[new_last_indices] = (new_spingroups, new_log_likelihood)
                # False Group:
                new_spingroups = spingroups + [G]
                new_log_likelihood = log_likelihood + log_prior[i,G]
                new_contenders[last_indices] = (new_spingroups, new_log_likelihood)
            contenders = copy(new_contenders)
            del new_contenders

            # Raise error if there are no more contenders left:
            if len(contenders) == 0:
                raise RuntimeError('The number of maximum likelihood contenders has dropped to zero unexpectedly.')
        
        # When finished, find the best contenders and return:
        max_log_likelihood = -np.inf
        for last_indices, (spingroups, log_likelihood) in contenders.items():
            for g in range(G):
                log_likelihood += log_level_spacing_probs[last_indices[g],-1,g]
            if log_likelihood > max_log_likelihood:
                max_log_likelihood = log_likelihood
                max_spingroups = spingroups
        return max_spingroups