import math
import numpy as np
from scipy.integrate import quad as integrate
from scipy.optimize import root_scalar
import scipy.special as spf

from Encore import Encore
import RMatrix as Distributions

from os.path import basename
THIS_FILE = basename(__file__)
ENCORE_FILE = 'Encore.py'

__doc__ = f"""
This module serves as a preprocessor and postprocessor for the 2-spingroup assignment algorithm,
"{ENCORE_FILE}". "{THIS_FILE}" extends the 2-spingroup algorithm to multiple-spingroups by
"merging" spingroups. {THIS_FILE} finds the probabilities for various merge cases and combines
the expected probabilities. Unlike the 2-spingroup case that gives the best answer given its
information (i.e. mean level spacings, reduced widths, false resonance probabilities, etc.), the
multiple-spingroup case is an approximation.
"""

# ==================================================================================
# Merger:
# ==================================================================================
class Merger:
    """
    This class combines properties of spin-groups such as level-spacings and mean
    parameters. ...

    Initialization Attributes:
    -------------------------
    N                :: int
        Number of possible spin-groups.
    levelSpacingFunc :: 'Wigner', 'Brody', or 'Missing'
        Determines the type of level-spacing distribution used.
    Freq             :: float [N]
        Level densities for each spingroup. Inverse of mean level spacing.
    w                :: float [N]  (optional)
        Brody Parameter for each spingroup.
    err              :: float      (default = 1e-9)
        Threshold for ignoring Wigner probabilities.
    pM               :: float [N]  (optional)
        Fraction of resonances that are missing for each spingroup.
        
    Internally Used Attributes:
    --------------------------
    a     :: float [N]
        Coefficient for level spacing distributions.
    Z     :: float [N,N]
        Matrix used to calculate the normalization factor for the merged group level-spacing
        distribution. See the `Z` variable in the merged level-spacing equation.
    ZB    :: float [N]
        A special variation of `Z` used for the level-spacing with one resonance that exists
        outside the ladder energy bounds. As such the resonance's position is assumed to be
        unknown except that we know it is not inside the ladder energy bounds.
    xMax  :: float
        The maximum level spacings that give a probability above a threshold error, `err`.
    xMaxB :: float
        A special variation of `xMax` used for the level-spacing with one resonance that exists
        outside the ladder energy bounds. As such the resonance's position is assumed to be
        unknown except that we know it is not inside the ladder energy bounds.
    """

    pid4       = math.pi/4
    sqrtpid4   = math.sqrt(pid4)

    def __init__(self, Freq, levelSpacingFunc:str='Wigner', BrodyParam=None, err:float=1e-9, pM=None):
        """
        ...
        """

        self.Freq = Freq.reshape(1,-1)
        self.N    = self.Freq.shape[1]

        self.LevelSpacingDist = levelSpacingFunc
        if levelSpacingFunc not in ('Wigner','Brody','Missing'):
            raise NotImplementedError(f'Level-spacing function, "{levelSpacingFunc}", is not implemented yet.')

        if levelSpacingFunc == 'Brody':
            if BrodyParam is None:
                self.w = np.ones((1,self.N), dtype='f8')
            else:
                self.w = BrodyParam

            w1i = 1/(self.w+1)
            self.a = (w1i*spf.gamma(w1i)*self.Freq)**(self.w+1)

        if pM is None:
            self.pM = np.zeros((1,self.N), dtype='f8')
        else:
            if levelSpacingFunc != 'Missing':
                raise ValueError('Need to use "Missing" function for missing factors')
            self.pM = pM

        xMax_limit  = self.__xMax_limit(err) # xMax must be bounded by the error for spin-group alone

        if self.N != 1:
            self.Z, self.ZB = self.__findZ(xMax_limit)

        self.xMax, self.xMaxB = self.__findxMax(err, xMax_limit)

    @property
    def FreqTot(self):
        'Gives the total frequency for the combined group.'
        return np.sum(self.Freq)
    @property
    def MeanLS(self):
        'Gives the combined mean level-spacing for the group.'
        return 1.0 / np.sum(self.Freq)

    def FindLevelSpacings(self, E, EB, Prior, verbose=False):
        """
        Finds the level-spacing probabilities, `level_spacing_probs` between each resonance,
        including resonance outside the left and right ladder energy bounds. It is unnecessary
        to calculate all `(L+2)**2` possibilities since the probabilities from most level-spacings
        would be negligible, so we calculate the left and right cutoffs using `xMax` and `xMaxB`
        and store the left and right bounds in `iMax`. `level_spacing_probs` stores all of the
        level-spacing probabilities between those bounds.

        ...
        """

        L  = E.shape[0]
        
        # Calculating the approximation thresholds:
        iMax = np.full((L+2,2), -1, dtype='i4')
        for j in range(L):  # Edge Case (Left)
            if E[j] - EB[0] >= self.xMaxB:
                iMax[0,0]    = j
                iMax[:j+1,1] = 0
                break
        for i in range(L-1):
            for j in range(iMax[i,0]+1,L):
                if E[j] - E[i] >= self.xMax:
                    iMax[i+1,0] = j
                    iMax[iMax[i-1,0]:j+1,1] = i+1
                    break
            else:
                iMax[i:,0] = L+1
                iMax[iMax[i-1,0]:,1] = i+1
                break
        for j in range(L-1,-1,-1):  # Edge Case (Right)
            if EB[1] - E[j] >= self.xMaxB:
                iMax[-1,1] = j
                iMax[j:,0] = L+1
                break
        
        # Level-spacing calculation:
        level_spacing_probs = np.zeros((L+2,L+2),'f8')
        if self.N == 1:
            for i in range(L-1):
                X = E[i+1:iMax[i+1,0]-1] - E[i]
                level_spacing_probs[i+1,i+2:iMax[i+1,0]] = self.levelSpacingFunc(X)
            level_spacing_probs[0,1:-1]  = self.levelSpacingFunc(E - EB[0], True)
            level_spacing_probs[1:-1,-1] = self.levelSpacingFunc(EB[1] - E, True)
        else:
            for i in range(L-1):
                X = E[i+1:iMax[i+1,0]-1] - E[i]
                PriorL = np.tile(Prior[i,:], (iMax[i+1,0]-i-2, 1))
                PriorR = Prior[i+1:iMax[i+1,0]-1,:]
                level_spacing_probs[i+1,i+2:iMax[i+1,0]] = self.__levelSpacingMerge(X, PriorL, PriorR)
            level_spacing_probs[0,1:-1]  = self.__levelSpacingMergeBounds(E - EB[0], Prior)
            level_spacing_probs[1:-1,-1] = self.__levelSpacingMergeBounds(EB[1] - E, Prior)

        # Error checking:
        if   (level_spacing_probs == np.nan).any():
            raise RuntimeError('Level-spacings have "NaN" values.')
        elif (level_spacing_probs == np.inf).any():
            raise RuntimeError('Level-spacings have "Inf" values.')
        
        # Verbose 
        if verbose: print('Finished level-spacing calculations')

        # The normalization factor is duplicated in the prior. One must be removed: FIXME!!!!!
        level_spacing_probs /= self.FreqTot

        return level_spacing_probs, iMax

    def __levelSpacingMerge(s, X, PriorL, PriorR):
        """
        ...
        """
        c, r1, r2 = s.__findParts(X.reshape(-1,1))
        d = r2 * (r1 - r2)

        Norm = np.matmul(PriorL.reshape(-1,1,s.N), np.matmul(s.Z.reshape(1,s.N,s.N), PriorR.reshape(-1,s.N,1))).reshape(-1,)
        return ((c / Norm) * ( \
                    np.sum(PriorL*r2, axis=1) * \
                    np.sum(PriorR*r2, axis=1) + \
                    np.sum(PriorL*PriorR*d, axis=1)
                )).reshape(X.shape)

    def __levelSpacingMergeBounds(s, X, Prior):
        """
        ...
        """
        c, r1, r2 = s.__findParts(X.reshape(-1,1))
        Norm = np.sum(s.ZB * Prior, axis=1)
        return ((c / Norm) * \
                    np.sum(Prior*r2, axis=1) \
                ).reshape(X.shape)

    def levelSpacingFunc(s, X, Boundary=False):
        if s.LevelSpacingDist == 'Wigner':
            coef = (s.sqrtpid4 * s.Freq)**2
            if Boundary:    return np.exp(-coef * X**2)
            else:           return 2 * coef * X * np.exp(-coef * X*X)
        elif s.LevelSpacingDist == 'Brody':
            Xw = X**s.w
            if Boundary:    return np.exp(-s.a * Xw*X)
            else:           return (s.w+1) * s.a * Xw * np.exp(-s.a * Xw*X)
        elif s.LevelSpacingDist == 'Missing':
            if Boundary:    return Distributions.missing_pdf(X, s.Freq, s.pM)[1]
            else:           return Distributions.missing_pdf(X, s.Freq, s.pM)[0]
    
    def __findParts(s, X):
        """
        ...
        """
        if s.LevelSpacingDist == 'Wigner':
            fX = s.sqrtpid4 * s.Freq * X
            R1 = 2 * s.sqrtpid4 * s.Freq * fX
            R2 = s.Freq / spf.erfcx(fX)     # "erfcx(x)" is "exp(x^2)*erfc(x)"
            F2 = np.exp(-fX*fX) / R2        # Getting the "erfc(x)" from "erfcx(x)"
        elif s.LevelSpacingDist == 'Brody':
            # FIXME: THIS NEEDS CHECKING:
            w1i = 1.0 / (s.w+1)
            aXw = s.a*X**s.w
            aXw1 = aXw*X
            R1 = (s.w+1)*aXw
            F2 = (w1i*(s.a)**w1i)*spf.gamma(w1i)*spf.gammaincc(w1i,aXw1)
            R2 = np.exp(-aXw1) / F2
        else:
            raise NotImplementedError(f'Have not implemented "{s.LevelSpacingDist}" yet')
        C  = np.prod(F2, axis=1)
        return C, R1, R2
   
    def __findZ(s, xMax_limit):
        """
        Function that calculates normalization matrices using the spingroups.
        """
        def offDiag(x, i:int, j:int):
            C, R1, R2 = s.__findParts(x)
            return C[0] * R2[0,i] * R2[0,j]
        def mainDiag(x, i:int):
            C, R1, R2 = s.__findParts(x)
            return C[0] * R1[0,i] * R2[0,i]
        def boundaries(x, i:int):
            C, R1, R2 = s.__findParts(x)
            return C[0] * R2[0,i]

        # Level Spacing Normalization Matrix:
        Z = np.zeros((s.N,s.N), dtype='f8')
        for i in range(s.N):
            for j in range(i):
                Z[i,j] = integrate(lambda _x: offDiag(_x,i,j), a=0.0, b=min(*xMax_limit[0,[i,j]]))[0]
                Z[j,i] = Z[i,j]
            Z[i,i] = integrate(lambda _x: mainDiag(_x,i), a=0.0, b=xMax_limit[0,i])[0]
        
        # Level Spacing Normalization at Boundaries:
        ZB = np.zeros((1,s.N), dtype='f8')
        ZB[0,:] = [integrate(lambda _x: boundaries(_x,i), a=0.0, b=xMax_limit[0,i])[0] for i in range(s.N)]
        return Z, ZB
    
    def __findxMax(self, err:float, xMax_limit):
        """
        ...
        """
        mthd = 'brentq' # method for root-finding
        bounds = [self.MeanLS, max(*xMax_limit)]
        if self.N == 1:
            xMax  = root_scalar(lambda x: self.levelSpacingFunc(x) - err     , method=mthd, bracket=bounds).root
            xMaxB = root_scalar(lambda x: self.levelSpacingFunc(x,True) - err, method=mthd, bracket=bounds).root
        else:
            # Bounding the equation from above:
            def upperBoundLevelSpacing(x):
                c, r1, r2 = self.__findParts(x)
                # Minimum normalization possible. This is the lower bound on the denominator with regards to the priors:
                Norm_LB = np.amin(self.Z)
                # Finding maximum numerator for the upper bound on the numerator:
                c1 = np.amax(r1*r2, axis=1)
                c2 = np.amax(r2**2)
                return (c / Norm_LB) * max(c1, c2)
            def upperBoundLevelSpacingBoundary(x):
                c, r1, r2 = self.__findParts(x)
                # Minimum normalization possible for lower bound:
                Norm_LB = np.amin(self.ZB)
                return (c / Norm_LB) * np.max(r2, axis=1) # Bounded above by bounding the priors from above
            
            xMax  = root_scalar(lambda x: upperBoundLevelSpacing(x)-err        , method=mthd, bracket=bounds).root
            xMaxB = root_scalar(lambda x: upperBoundLevelSpacingBoundary(x)-err, method=mthd, bracket=bounds).root
        return xMax, xMax

    def __xMax_limit(self, err:float):
        """
        In order to estimate `xMax` and `xMaxB`, we first must calculate `Z` and `ZB`. `xMax_limit`
        is used to get a simple upper bound on `xMax` and `xMaxB`. This limit assumes that the
        xMax for the level spacing distribution is atleast less than the xMax for Poisson
        distribution.

        ...
        """
        pM = self.pM if self.LevelSpacingDist == 'Missing' else 0.0
        xMax = -np.log(err)/(self.Freq*(1-pM))
        return xMax
        # if s.LevelSpacingDist == 'Wigner':
        #     xMax = np.sqrt(-np.log(err)) / (s.sqrtpid4 * s.Freq)
        # elif s.LevelSpacingDist == 'Brody':
        #     xMax = (-np.log(err) / s.a) ** (1.0 / (s.w+1))
        # elif s.LevelSpacingDist == 'Missing':
        #     # FIXME: THIS IS NOT CORRECT DISTRIBUTION!!!
        #     xMax = 5*np.sqrt(-np.log(err)) / (s.sqrtpid4 * s.Freq)
        # # FIXME: FUDGE FACTOR IN USE!!!
        # return 2*xMax

# =================================================================================================
#     WigBayes Partition / Run Master:
# =================================================================================================

class RunMaster:
    f"""
    A wrapper for {ENCORE_FILE} responsible for partitioning spingroup merging, and combining the spingroups.

    ...
    """

    def __init__(self, E, EB, Prior, TPPrior, Freq, LevelSpacingDist:str='Wigner', err:float=1e-9, BrodyParam=None, MissingFrac=None):
        self.E       = E
        self.EB      = EB
        self.Prior   = Prior
        self.TPPrior = TPPrior
        self.Freq    = Freq

        self.LevelSpacingDist = LevelSpacingDist
        self.err     = err
        self.w       = BrodyParam
        self.pM      = MissingFrac

        self.L = E.shape[0]
        self.N = Freq.shape[1] - 1

    def mergePartitioner(s, partitions:list):
        """
        ...
        """

        n = len(partitions)

        # Merged level-spacing calculation:
        level_spacing_probs = np.zeros((s.L+2, s.L+2, n), 'f8')
        iMax = np.zeros((s.L+2,     2, n), 'i4')
        for g, group in enumerate(partitions):
            if s.w is None:     BP = None
            else:               BP = s.w[:,group]
            if s.pM is None:    PM = None
            else:               PM = s.pM[:,group]
            merge = Merger(s.Freq[:,group], s.LevelSpacingDist, BrodyParam=BP, err=s.err, pM=PM)
            level_spacing_probs[:,:,g], iMax[:,:,g] = merge.FindLevelSpacings(s.E, s.EB, s.Prior[:,group])

        # Merged prior calculation:
        prior_merged = np.zeros((s.L, n+1), 'f8')
        for g, group in enumerate(partitions):
            if hasattr(group, '__iter__'):
                prior_merged[:,g] = np.sum(s.Prior[:,group], axis=1)
            else:
                prior_merged[:,g] = s.Prior[:,group]
        prior_merged[:,-1] = s.Prior[:,-1]

        return level_spacing_probs, iMax, prior_merged

    def WigBayes(s, return_log_tot_prob=False, verbose=False):
        """
        Spingroup partitioner for merging spingroups for the WigBayes algorithm.

        ...
        """
        
        # Only 2 spin-groups (merge not needed):
        if s.N == 2:
            if verbose: print(f'Preparing level-spacings')
            level_spacing_probs_g, iMax_g, prior_g = s.mergePartitioner([[0], [1]])
            if verbose: print(f'Finished level-spacing calculations')
            ENCORE = Encore(prior_g, level_spacing_probs_g, iMax_g)
            if verbose: print(f'Finished CP calculation')
            sg_probs = ENCORE.WigBayes()
            if verbose: print(f'Finished WigBayes calculation')

            if return_log_tot_prob:
                log_tot_prob = ENCORE.LogTotProb(s.EB, s.Freq, s.TPPrior)
                return sg_probs, log_tot_prob
            else:
                return sg_probs

        # More than 2 spin-groups (merge needed):
        else:
            sg_probs = np.zeros((s.L,3,s.N),dtype='f8')
            if return_log_tot_prob:
                log_tot_prob = np.zeros(s.N, dtype='f8')

            # Partitioning:
            for g in range(s.N):
                partition = [[g], [g_ for g_ in range(s.N) if g_ != g]]
                if verbose: print(f'Preparing for Merge group, {g}')
                level_spacing_probs_g, iMax_g, prior_g = s.mergePartitioner(partition)
                if verbose: print(f'Finished spin-group {g} level-spacing calculation')
                ENCORE = Encore(prior_g, level_spacing_probs_g, iMax_g)
                if verbose: print(f'Finished spin-group {g} CP calculation')
                sg_probs[:,:,g] = ENCORE.WigBayes()
                if verbose: print(f'Finished spin-group {g} WigBayes calculation')

                if return_log_tot_prob:
                    # FIXME: I DON'T KNOW LOG TOT PROB CORRECTION FACTOR FOR MERGED CASES! 
                    Freq_comb = np.array([s.Freq[0,g], np.sum(s.Freq[0,:])-s.Freq[0,g]]).reshape(1,-1)
                    log_tot_prob[g] = ENCORE.LogTotProb(s.EB, Freq_comb, s.TPPrior)

            # Combine probabilities for each merge case:
            combined_sg_probs = s.probCombinator(sg_probs)
            if return_log_tot_prob:
                if verbose: print(f'Preparing for Merge group, 1000!!!')
                level_spacing_probs_1, iMax_1, prior_1 = s.mergePartitioner([tuple(range(s.N))])
                if verbose: print(f'Finished spin-group 1000 level-spacing calculation')
                ENCORE = Encore(prior_1, level_spacing_probs_1, iMax_1)
                if verbose: print(f'Finished spin-group {g} CP calculation')
                base_LogProb = ENCORE.LogTotProb(s.EB, s.Freq, s.TPPrior)
                combined_log_tot_prob = s.logTotProbCombinator(log_tot_prob, base_LogProb)
                if verbose: print('Finished!')
                return combined_sg_probs, combined_log_tot_prob
            else:
                if verbose: print('Finished!')
                return combined_sg_probs

    def probCombinator(self, sg_probs):
        """
        Combines probabilities from various spingroup partitions.

        ...
        """

        combined_sg_probs = np.zeros((self.L,self.N+1), dtype='f8')
        for g in range(self.N):
            combined_sg_probs[:,g] = sg_probs[:,1,g]
        combined_sg_probs[:,-1] = np.prod(sg_probs[:,1,:], axis=1) * self.Prior[:,-1] ** (1-self.N)
        combined_sg_probs[self.Prior[:,-1]==0.0,  -1] = 0.0
        combined_sg_probs /= np.sum(combined_sg_probs, axis=1).reshape((-1,1))
        return combined_sg_probs

    def WigSample(s, Trials:int=1, verbose=False):
        """
        Spingroup partitioner for merging spingroups for the WigSample algorithm.

        ...
        """
        
        # Only 2 spin-groups (merge not needed):
        if s.N == 2:
            if verbose: print(f'Preparing level-spacings')
            level_spacing_probs_g, iMax_g, prior_g = s.mergePartitioner([[0], [1]])
            if verbose: print(f'Finished level-spacing calculations')
            ENCORE = Encore(prior_g, level_spacing_probs_g, iMax_g)
            if verbose: print(f'Finished CP calculation')
            Samples = ENCORE.WigSample(Trials)
            if verbose: print(f'Finished WigBayes calculation')
            return Samples

        # More than 2 spin-groups (Merge needed):
        else:
            raise NotImplementedError('WigSample for more than two spingroups has not been implemented yet.')
            Probs = np.zeros((s.L,3,s.N),dtype='f8')
            if log_total_probability:
                ltp = np.zeros(s.N, dtype='f8')
            for g in range(s.N):
                partition = [[g], [g_ for g_ in range(s.N) if g_ != g]]
                if verbose: print(f'Preparing for Merge group, {g}')
                level_spacing_probs_g, iMax_g, prior_g = s.mergePartitioner(partition)
                if verbose: print(f'Finished spin-group {g} level-spacing calculation')
                ENCORE = Encore(prior_g, level_spacing_probs_g, iMax_g)
                if verbose: print(f'Finished spin-group {g} CP calculation')
                Probs[:,:,g] = ENCORE.WigBayes()
                if verbose: print(f'Finished spin-group {g} WigBayes calculation')

                if log_total_probability:
                    ltp[g] = ENCORE.LogTotProb(s.EB, s.Freq, s.TPPrior)

            # Combine probabilities for each merge case:
            NewProbs = s.probCombinator(Probs)
            if log_total_probability:
                if verbose: print(f'Preparing for Merge group, 1000!!!')
                level_spacing_probs_1, iMax_1, prior_1 = s.mergePartitioner([tuple(range(s.N))])
                if verbose: print(f'Finished spin-group 1000 level-spacing calculation')
                ENCORE = Encore(prior_1, level_spacing_probs_1, iMax_1)
                if verbose: print(f'Finished spin-group {g} CP calculation')
                base_LogProb = ENCORE.LogTotProb(s.EB, s.Freq, s.TPPrior)
                NewLTP = s.ltp_combinator(ltp, base_LogProb)
                if verbose: print('Finished!')
                return NewProbs, NewLTP
            else:
                if verbose: print('Finished!')
                return NewProbs

    def logTotProbCombinator(self, LogProbs, base_LogProb):
        """
        Combines log total probabilities from from various partitions.
        """

        return np.sum(LogProbs) - (self.N-1)*base_LogProb

# ==================================================================================
# Missing/False Resonances:
# ==================================================================================

# ...