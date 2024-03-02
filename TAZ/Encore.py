from copy import copy
import math
import numpy as np
from numpy import newaxis as NA
# import autograd.numpy as np

__doc__ = """
This file is responsible for the 1 and 2 spingroup classification algorithm. For more than 2
spingroups, the `Merger` class must be used to combine spingroups.
"""

class Encore:
    """
    A class for 1 or 2 spingroup classification. This class has many features:

    1. WigBayes: spingroup probabilities for each resonance, given the resonance ladder and mean
    parameters.

    2. WigSample: random sample of possible spingroup assignments. Likelihood of ladder sample
    correlates with the ladder spingroup assignment probability.
    
    3. ProbOfSample: probability of sampling the exact ladder of spingroup assignments.

    4. LogTotProb: log of the probability of sampling the resonance ladder regardless
    of spingroup, given the mean parameters. This is used for log-likelihood estimation for mean
    parameter estimation.

    Let "L" be the number of resonances in the ladder and "G" be the number of spingroups
    (excuding the false group).

    Initialization Variables
    ------------------------
    prior               : float [L+2, G+1]
        Prior probabilities on each resonance (with boundaries) and spingroup (with false group).

    level_spacing_probs : float [L+2, L+2, G]
        All of the level spacings from position index 0 to index 1 with spingroup, index 2.

    iMax                : int [L+2, 2, G]
        Maximum/minimum index for approximation threshold for each resonance (index 0) in each
        direction (index 1) for each spingroup (index 2).
    
    Internally Used Variables
    -------------------------
    CP : float [L+2, L+2, 2]
        'Cumulative Probabilities' which are the building blocks for Bayesian probability
        calculations and probabilistic sampling.

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
List A:    {AList}
List B:    {BList}
"""

    CP_BAD_ERROR = 'A cumulative probability ("CP") has a "{}" value.'

    TP_PERCENT_ERROR = """
In calculating the total probability ("TP"), there was a percent error of {p_error:.3f}%.
This exceeds the set limit of {p_thres:.3f}%. This error could be attributed to a bug or
numerical instability.
"""

    TP_ERROR_THRESHOLD = 1.0 # % (percent)

    # ==================================================================================
    # Constructor
    # ==================================================================================
    def __init__(self, prior, level_spacing_probs, iMax):
        """
        Initialization ...

        ...
        """

        # Initialization:
        self.G = level_spacing_probs.shape[2]
        if self.G not in (1,2):
            raise NotImplementedError('ENCORE can only handle 1 or 2 spingroups and a false group at the moment.')
        self.L = np.int32(prior.shape[0])
        self.Prior = np.ones((self.L+2,self.G+1), dtype='f8')
        self.Prior[1:-1,:] = prior
        self.LSP = level_spacing_probs
        self.iMax = iMax

        # Calculating "CP" and "TP":
        if   self.G == 1:       self.__makeCP1()
        elif self.G == 2:       self.__makeCP2()
        self.__makeTP()

    # ==================================================================================
    # Make CP (1 spingroups and false group)
    # ==================================================================================
    def __makeCP1(s):
        """
        `__MakeCP1` calculates `CP` -- the 'Cumulative Probabilities' for the 1 spingroup case.
        `CP` is a three dimensional array that takes the form, `CP(i,dir)`, where `i` is the
        indices of the true spingroup and `dir` is the direction that the probabilities are
        accumulating. ...

        ...
        """

        L = s.L
        s.CP = np.zeros((L+2,2), dtype='f8')
        s.PW = np.zeros(L+2, dtype='f8')

        # Trivial CP & PW values:
        s.CP[ 0, 0] = 1.0
        s.CP[-1, 1] = 1.0
        s.PW[[0,-1]] = 1.0

        for iL in range(1,L+2):
            iR = L+1-iL

            # =================================================================================
            # Left-Hand Side:

            s.CP[iL,0] = s.CP[iL-1,0] * s.PW[iL-1] * s.Prior[iL-1,0] * s.LSP[iL-1,iL,0]
            mult_chain = 1.0
            for j in range(iL-2, s.iMax[iL,1,0]-1, -1):
                mult_chain *= s.Prior[j+1,1] * s.PW[j+1]
                if mult_chain == 0.0:   break   # Potential computation time improvement
                s.CP[iL,0] += s.CP[j,0] * mult_chain * s.PW[j] * s.Prior[j,0] * s.LSP[j,iL,0]
            
            # =================================================================================
            # Right-Hand Side:

            s.CP[iR,1] = s.CP[iR+1,1] * s.PW[iR+1] * s.Prior[iR+1,0] * s.LSP[iR,iR+1,0]
            mult_chain = 1.0
            for j in range(iR+2, s.iMax[iR,0,0]+1, 1):
                mult_chain *= s.Prior[j-1,1] * s.PW[j-1]
                if mult_chain == 0.0:   break   # Potential computation time improvement
                s.CP[iR,1] += s.CP[j,1] * mult_chain * s.PW[j] * s.Prior[j,0] * s.LSP[iR,j,0]
            
            # =====================================================================================
            # Finding PW:

            if 2*iL <= L+1:
                if s.CP[iL,0] == 0.0:
                    raise RuntimeError(s.PW_INF_ERROR.format(side='left', idx=iL, AList=s.CP[iL,0], BList='NaN'))
                s.PW[iL] = 1.0 / s.CP[iL,0]

            if 2*iL <= L:
                if s.CP[iR,1] == 0.0:
                    raise RuntimeError(s.PW_INF_ERROR.format(side='right', idx=iR, AList=s.CP[iR,1], BList='NaN'))
                s.PW[iR] = 1.0 / s.CP[iR,1]

    # ==================================================================================
    # Make CP (2 spingroups and false group)
    # ==================================================================================
    def __makeCP2(s):
        """
        `__MakeCP2` calculates `CP` -- the 'Cumulative Probabilities'. `CP` is a three dimensional
        array that takes the form, `CP(iA,iB,dir)`, where `iA` and `iB` are the indices of 'A'
        and 'B' and `dir` is the direction. `CP(iA,iB,dir)` is the probability of the spingroup
        assignment distribution on the set of resonances with indices less that or equal to
        `max(iA,iB)` for `dir = 1` or greater than or equal to `min(iA,iB)` for `dir = 2` that
        meet the following conditions:
            (1) `iA` is the index of the right-most (`dir = 1`)
                or left-most (`dir = 2`) resonance with spingroup A.
            (2) `iB` is the index of the right-most (`dir = 1`)
                or left-most (`dir = 2`) resonance with spingroup B.

        `CP` is calculated through an iterative process, hince the name. The true value of `CP`
        has the tendency to exponentially grow or decay through each iteration, therefore, we use
        'Placement Weights' (`PW`) to renormalize the value to an acceptable floating-point range.
        """

        L = s.L
        s.CP = np.zeros((L+2,L+2,2), dtype='f8')
        s.PW = np.zeros(L+2, dtype='f8')

        # Trivial CP values:
        s.CP[ 0, 0, 0] = 1.0 ;      s.CP[-1, -1, 1] = 1.0
        # s.CP[ 1, 0, 0], s.CP[ 0, 1, 0] = s.LSP[ 0, 1, :]
        # s.CP[-2,-1, 1], s.CP[-1,-2, 1] = s.LSP[-2,-1, :]

        s.PW[[0,-1]] = 1.0
        # denom = s.CP[ 1, 0, 0] + s.CP[ 0, 1, 0]
        # s.PW[1]  = 1.0 / denom
        # denom = s.CP[-2,-1, 1] + s.CP[-1,-2, 1]
        # s.PW[-2] = 1.0 / denom


        jMax  = np.array([[0, L+1], [0, L+1]], dtype='u4')
        for i1L in range(1,L+2):
            i1R = L+1-i1L

            jMax[:,0] = s.iMax[i1L,1,:]
            jMax[:,1] = s.iMax[i1R,0,:]
            i2Max = jMax[1::-1,:]
            for g in range(2):
                # =================================================================================
                # Left-Hand Side Iteration:

                idx = range(i1L-1, jMax[g,0]-1, -1)
                ls = s.LSP[idx, i1L, g]
                for i2L in range(i1L-1, i2Max[g,0]-1, -1):

                    # Finding J1 and J2:
                    if (jMax[g,0] > i2L) or (i2L == 0):
                        j1 = idx ;      ls1 = ls
                        j2 = []  ;      ls2 = []
                    else:
                        j1 = range(i1L-1, i2L, -1)         ;    ls1 = ls[:i1L-i2L-1]
                        j2 = range(i2L-1, jMax[g,0]-1, -1) ;    ls2 = ls[i1L-i2L:]
                        
                    # New CP Iteration:
                    s.__CPIter(j1,j2,ls1,ls2,i1L,i2L,g,1)

                # =================================================================================
                # Right-Hand Side Iteration:

                idx = range(i1R+1, jMax[g,1]+1)
                ls  = s.LSP[i1R, idx, g]
                for i2R in range(i1R+1, i2Max[g,1]+1):

                    # Finding J1 and J2:
                    if (jMax[g,1] < i2R) or (i2R == L+1):
                        j1 = idx ;      ls1 = ls
                        j2 = []  ;      ls2 = []
                    else:
                        j1 = range(i1R+1, i2R)         ;    ls1 = ls[:i2R-i1R-1]
                        j2 = range(i2R+1, jMax[g,1]+1) ;    ls2 = ls[i2R-i1R:]

                    # New CP Iteration:
                    s.__CPIter(j1,j2,ls1,ls2,i1R,i2R,g,-1)

            # =====================================================================================
            # Finding PW:

            # Left-Hand Side PW Calculation:
            if 2*i1L <= L+1:
                idxB = range(i2Max[0,0], i1L)
                idxA = range(i2Max[1,0], i1L)
                denom = np.sum(s.CP[i1L,idxB,0]) + np.sum(s.CP[idxA,i1L,0])
                if denom == 0.0:
                    raise RuntimeError(s.PW_INF_ERROR.format(side='left', idx=i1L, AList=s.CP[idxA,i1L,0], BList=s.CP[i1L,idxB,0]))
                s.PW[i1L] = 1.0 / denom

            # Right-Hand Side PW Calculation:
            if 2*i1L <= L:
                idxB = range(i1R+1, i2Max[0,1]+1)
                idxA = range(i1R+1, i2Max[1,1]+1)
                denom = np.sum(s.CP[i1R,idxB,1]) + np.sum(s.CP[idxA,i1R,1])
                if denom == 0.0:
                    raise RuntimeError(s.PW_INF_ERROR.format(side='right', idx=i1R, AList=s.CP[idxA,i1R,1], BList=s.CP[i1R,idxB,1]))
                s.PW[i1R] = 1.0 / denom

    # ==================================================================================
    # CP Iteration Function
    # ==================================================================================
    def __CPIter(s, J1, J2, ls1, ls2, i1, i2, g, sgn):
        """
        `__CPIter` is a function that calculates the next `CP` element using an iterative process.
        
        ...
        """

        dir = int(sgn == -1)
        cp_out = 0.0

        # Preceeding i2:
        if J1:
            if g:   prod1 = ls1 * s.CP[i2,J1,dir]
            else:   prod1 = ls1 * s.CP[J1,i2,dir]
            
            j = J1[0]
            mult_chain = s.PW[j]
            cp_out = mult_chain * prod1[0] * s.Prior[j,g]
            for j, p1 in zip(J1[1:], prod1[1:]):
                mult_chain *= s.PW[j] * s.Prior[j+sgn,2]
                if mult_chain == 0.0:   break   # Potential computation time improvement
                cp_out += mult_chain * p1 * s.Prior[j,g]

        # Behind i2:
        if J2:
            if i2+sgn == i1:    mult_chain  = s.PW[i2] * s.Prior[i2,1-g]
            elif i1 == i2:      mult_chain  = 1.0
            else:               mult_chain *= s.PW[i2] * s.Prior[i2,1-g] * s.Prior[i2+sgn,2]

            if g:   cp_out += mult_chain * np.sum(ls2 * s.CP[i2,J2,dir])
            else:   cp_out += mult_chain * np.sum(ls2 * s.CP[J2,i2,dir])

        # Error Checks:
        if   cp_out == math.inf:
            raise RuntimeWarning(s.CP_BAD_ERROR.format('Inf'))
        elif cp_out == math.nan:
            raise RuntimeWarning(s.CP_BAD_ERROR.format('NaN'))
        
        # Assigning CP:
        if g:   s.CP[i2,i1,dir] = cp_out
        else:   s.CP[i1,i2,dir] = cp_out

    # ==================================================================================
    # Find Total Probability Factor
    # ==================================================================================
    def __makeTP(s):
        """
        `__makeTP` is a function that calculates the total probability factor, `TP` using `CP`
        and `PW` found in the function, `__makeCP1` or `__makeCP2`. `TP` can be found by
        calculating `CP[ -1, -1, 0]` or `CP[ 0, 0, 1]`. Each can be performed by summing `CP`
        across the first or second spingroup. Therefore, there are a four ways to calculate `TP`.
        We accept the mean of the four calculation as the total probability and find the percent
        error for error-checking purposes.
        """

        # =====================================================================================
        # Finding CP[ -1, -1, 0] = CP[ 0, 0, 1] = TP:

        # Alternative solution:
        # for t in range(2):
        #     J2  = range(s.L, s.iMax[s.L+1,1,t]-1, -1)
        #     ls2 = s.LSP[J2, s.L+1, t]
        #     s.__CPIter([], J2, [], ls2, s.L+1, s.L+1, t, 1)

        #     J2  = range(1, s.iMax[0,0,t]-1)
        #     ls2 = s.LSP[0, J2, t]
        #     s.__CPIter([], J2, [], ls2, 0, 0, t, -1)

        if s.G == 2:
            TPs = np.empty((2,2), dtype='f8')
            TPs[0,0] = np.sum(s.LSP[:-1, -1, 0] * s.CP[:-1, -1, 0])
            TPs[1,0] = np.sum(s.LSP[:-1, -1, 1] * s.CP[-1, :-1, 0])
            TPs[0,1] = np.sum(s.LSP[  0, 1:, 0] * s.CP[ 1:,  0, 1])
            TPs[1,1] = np.sum(s.LSP[  0, 1:, 1] * s.CP[ 0,  1:, 1])
            s.CP[-1,-1,0], s.CP[0,0,1] = np.mean(TPs, axis=0)

        # =====================================================================================

        if   s.G == 1:      TP1 = s.CP[0,1];    TP2 = s.CP[-1,0]
        elif s.G == 2:      TP1 = s.CP[0,0,1];  TP2 = s.CP[-1,-1,0]
        
        # Taking the average of the two sides for the total probability factor:
        s.TP = (TP1 + TP2) / 2

        # If the two sides are past a threshold percent difference, raise a warning:
        percent_error = 100.0 * abs(TP1 - TP2) / s.TP
        if percent_error > s.TP_ERROR_THRESHOLD:
            print(RuntimeWarning(s.TP_PERCENT_ERROR.format(p_error=percent_error, p_thres=s.TP_ERROR_THRESHOLD)))

    # ==================================================================================
    # Level Spacing Bayes Theorem
    # ==================================================================================
    def WigBayes(s):
        """
        `WigBayes` finds the probability on each spingroup assignment of each resonance, informed
        by the level-spacing probability distributions. This is performed using the calculated `CP`
        and `TP` probabilities and Bayes' Theorem.

        ...
        """

        # Error checking:
        if s.G not in (1, 2):
            raise NotImplementedError('Currently "WigBayes" can only handle 1 or 2 spingroups and a False group. \
                                        Merge spingroups to get probabilities on more spingroups.')

        L = s.L
        sp = np.zeros((L,s.G), dtype='f8') # No longer need the edge cases, so the bounds are 0:L instead of 0:L+2

        if s.G == 1:
            sp[:,0] = s.CP[1:-1,0] * s.CP[1:-1,1]

        elif s.G == 2:
            for g in range(2):
                for iL in range(L):
                    iRMax = s.iMax[iL,0,1-g]
                    lsp = s.LSP[iL, iL+2:iRMax+1, 1-g]
                    for i in range(iL+1, iRMax):
                        idxR = range(i+1, iRMax+1)
                        if g:   sp[i-1,1] += s.CP[iL,i,0] * np.sum(lsp[i-iL-1:] * s.CP[idxR,i,1])
                        else:   sp[i-1,0] += s.CP[i,iL,0] * np.sum(lsp[i-iL-1:] * s.CP[i,idxR,1])
                    # if t:   sp[iL:iRMax-1,1] += np.array([s.CP[iL,i,0] * np.sum(ls[i-iL-1:] * s.CP[i+1:iRMax+1,i,1]) for i in range(iL+1, iRMax)])
                    # else:   sp[iL:iRMax-1,0] += np.array([s.CP[i,iL,0] * np.sum(ls[i-iL-1:] * s.CP[i,i+1:iRMax+1,1]) for i in range(iL+1, iRMax)])

        # Factors shared throughout the sum including the normalization factor, "TP":
        sp *= s.Prior[1:-1,:s.G] * s.PW[1:-1][:,NA] / s.TP

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
        
        ...
        """
        
        # Setting random number generator:
        if rng is None:
            rng = np.random.default_rng(seed)
        
        L = s.L
        sampled_groups = s.G * np.ones((L,num_trials), dtype='u1') # the sampled spingroups (default as false)
        
        if s.G == 1:
            for tr in range(num_trials):
                last_seen = 0 # last seen indices for each spingroup, for each trial
                while True: # iterate procedure until the resonance ladder is filled
                    iMax = s.iMax[last_seen,0,0]
                    likelihoods = np.zeros((iMax-last_seen,), dtype='f8')
                    mult_chain = 1.0
                    Js = np.arange(last_seen+1, iMax+1, dtype='i4')
                    for lidx, j in enumerate(Js):
                        mult_chain *= s.PW[j]
                        if mult_chain == 0.0:   break # Potential computation time improvement
                        likelihoods[lidx] = s.LSP[last_seen,j,0] * s.Prior[j,0] * mult_chain * s.CP[j,1]
                        mult_chain *= s.Prior[j,1]
                    sample_probs = likelihoods / np.sum(likelihoods)
                    last_seen = rng.choice(Js, p=sample_probs)
                    if last_seen == L+1:
                        break
                    i_new = last_seen - 1
                    sampled_groups[i_new,tr] = 0

        elif s.G == 2:
            for tr in range(num_trials):
                last_seen = np.zeros((2,), dtype='u4') # last seen indices for each spingroup, for each trial
                while True: # iterate procedure until the resonance ladder is filled
                    last_seen_all = np.max(last_seen)
                    iMaxA = s.iMax[last_seen[0],0,0]
                    iMaxB = s.iMax[last_seen[1],0,1]
                    iMax  = min(iMaxA, iMaxB)
                    likelihoods = np.zeros((iMax-last_seen_all,2), dtype='f8')
                    mult_chain = 1.0
                    Js = np.arange(last_seen_all+1, iMax+1, dtype='i4')
                    for lidx, j in enumerate(Js):
                        mult_chain *= s.PW[j]
                        if mult_chain == 0.0:
                            break   # Potential computation time improvement
                        IdxA = slice(j+1, iMaxA+1)
                        IdxB = slice(j+1, iMaxB+1)
                        likelihoods[lidx,0] = s.LSP[last_seen[0],j,0] * s.Prior[j,0] * mult_chain * np.sum(s.LSP[last_seen[1],IdxB,1]*s.CP[j,IdxB,1])
                        likelihoods[lidx,1] = s.LSP[last_seen[1],j,1] * s.Prior[j,1] * mult_chain * np.sum(s.LSP[last_seen[0],IdxA,0]*s.CP[IdxA,j,1])
                        mult_chain *= s.Prior[j,2]
                    prob_sgs = np.sum(likelihoods, axis=0) / np.sum(likelihoods)
                    g = rng.choice(2, p=prob_sgs)
                    sample_probs = likelihoods[:,g] / np.sum(likelihoods[:,g])
                    last_seen[g] = rng.choice(Js, p=sample_probs)
                    last_seen_all = last_seen[g]
                    i_new = last_seen_all - 1
                    sampled_groups[i_new,tr] = g
                    if last_seen_all == L:
                        break

        return sampled_groups
    
    # ==================================================================================
    # Sample Probability
    # ==================================================================================
    def ProbOfSample(s, spingroup_assignments) -> float:
        """
        `ProbOfSample` finds the probability of sampling the given spingroup assignments.

        ...
        """

        # NOTE: This needs checking!!!
        likelihood = 1.0
        if   s.G == 1:
            raise NotImplementedError('Single spingroup sampling probability has not been implemented yet.')
        
        elif s.G == 2:
            last_res = np.zeros(2,'i4')
            for i, sg in enumerate(spingroup_assignments):
                j = last_res[sg]
                likelihood *= s.Prior[i+1,sg] * s.PW[i+1] * s.LSP[j,i+1,sg]
                last_res[sg] = i+1
            likelihood *= s.LSP[last_res[0],-1,0] * s.LSP[last_res[1],-1,1]
        
        prob_of_sampling = likelihood / s.TP
        return prob_of_sampling

    # ==================================================================================
    # Log Likelihood
    # ==================================================================================
    def LogLikelihood(self, EB:tuple, lvl_dens_false:float, log_likelihood_prior:float=None) -> float:
        """
        `LogLikelihood` gives a criterion for the likelihood of sampling the given energies and
        widths, assuming correct information was given (i.e. mean level spacing, mean neutron
        width, etc.). This can be used with gradient descent to optimize mean parameters.
        """

        dE = EB[1] - EB[0]
        log_likelihood = math.log(self.TP) - np.sum(np.log(self.PW)) - lvl_dens_false * dE

        # Prior log likelihood:
        if log_likelihood_prior is not None:
            log_likelihood += log_likelihood_prior

        return log_likelihood

    @staticmethod
    def LikelihoodString(log_likelihood:float, sigfigs:int=3) -> str:
        """
        Prints the likelihood given the log likelihood. The likelihood sometimes cannot be
        given directly since the exponential term could exceed floating-point limitations;
        for this reason, likelihood is given by a custom scientific notation.
        """

        # Turn to into log10:
        log_likelihood /= math.log(10)

        out_str = f'{{0:.{sigfigs}f}}e{{1}}'
        exponent    = math.floor(log_likelihood)
        significand = 10 ** (log_likelihood % 1.0)
        return out_str.format(significand, exponent)
    
    # ==================================================================================
    # Maximum-Likelihood Assignments
    # ==================================================================================
    @staticmethod
    def WigMaxLikelihood(prior, level_spacing_probs, iMax):
        """
        Returns the maximum likelihood spingroup assignments using branching and pruning methods.

        ...
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