import math
import numpy as np
from scipy.integrate import quad as integrate
from scipy.optimize import root_scalar
import scipy.special as spf

from Encore import Encore
import Distributions

# ==================================================================================
# Merger:
# ==================================================================================
class Merger:
    """
    This class combines properties of spin-groups such as level-spacings and mean
    parameters. ...

    Variables:
    ...
    Freq  :: 64b float
    N     ::  8b   int       | ...

    Z     :: 64b float [N,N] | ...
    ZB    :: 64b float [N]   | ...
    xMax  :: 64b float       | ...
    xMaxB :: 64b float       | ...
    """

    pid4       = math.pi/4
    sqrtpid4   = math.sqrt(pid4)

    def __init__(self, Freq, LevelSpacingDist, err:float=1e-9):
        """
        ...
        """

        self.Freq = Freq

        self.N = LevelSpacingDist.num
        self.LevelSpacingDist = LevelSpacingDist

        xMaxLimit  = self.__xMaxLimit(err) # xMax must be bounded by the error for spin-group alone

        if self.N != 1:
            self.Z, self.ZB = self.__FindZ(xMaxLimit)

        self.xMax, self.xMaxB = self.__FindxMax(err, xMaxLimit)

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
        LS = np.zeros((L+2,L+2),'f8')
        if self.N == 1:
            for i in range(L-1):
                X = E[i+1:iMax[i+1,0]-1] - E[i]
                LS[i+1,i+2:iMax[i+1,0]] = self.LevelSpacingDist.f0(X).reshape(-1)
            LS[0,1:-1]  = self.LevelSpacingDist.f1(E - EB[0]).reshape(-1)
            LS[1:-1,-1] = self.LevelSpacingDist.f1(EB[1] - E).reshape(-1)
        else:
            for i in range(L-1):
                X = E[i+1:iMax[i+1,0]-1] - E[i]
                PriorL = np.tile(Prior[i,:], (iMax[i+1,0]-i-2, 1))
                PriorR = Prior[i+1:iMax[i+1,0]-1,:]
                LS[i+1,i+2:iMax[i+1,0]] = self.__LSMerge(X, PriorL, PriorR)
            LS[0,1:-1]  = self.__LSMergeBoundaries(E - EB[0], Prior)
            LS[1:-1,-1] = self.__LSMergeBoundaries(EB[1] - E, Prior)

        if   (LS == np.nan).any():
            raise RuntimeError('Level-spacings have "NaN" values.')
        elif (LS == np.inf).any():
            raise RuntimeError('Level-spacings have "Inf" values.')
        if verbose: print('Finished level-spacing calculations')
        # return LS, iMax

        # The normalization factor is duplicated in the prior. One must be removed:
        LS /= self.FreqTot

        return LS, iMax

    def __LSMerge(s, X, PriorL, PriorR):
        """
        ...
        """

        c, r1, r2 = s.LevelSpacingDist.parts(X)
        d = r2 * (r1 - r2)

        Norm = np.matmul(PriorL.reshape(-1,1,s.N), np.matmul(s.Z.reshape(1,s.N,s.N), PriorR.reshape(-1,s.N,1))).reshape(-1,)
        return ((c / Norm) * ( \
                    np.sum(PriorL*r2, axis=1) * \
                    np.sum(PriorR*r2, axis=1) + \
                    np.sum(PriorL*PriorR*d, axis=1)
                )).reshape(X.shape)

    def __LSMergeBoundaries(s, X, Prior):
        """
        ...
        """

        c, r2 = s.LevelSpacingDist.parts(X)[::2]
        Norm = np.sum(s.ZB * Prior, axis=1)
        return ((c / Norm) * \
                    np.sum(Prior*r2, axis=1) \
                ).reshape(X.shape)
   
    def __FindZ(s, xMaxLimit):
        """
        ...
        """

        def OffDiag(x, i:int, j:int):
            F2, R2 = s.LevelSpacingDist.parts(x)[::2]
            C = np.prod(F2)
            return C * R2[i] * R2[j]
        def MainDiag(x, i:int):
            F2, R1, R2 = s.LevelSpacingDist.parts(x)
            C = np.prod(F2)
            return C * R1[i] * R2[i]
        def Boundaries(x, i:int):
            F2, R2 = s.LevelSpacingDist.parts(x)[::2]
            C = np.prod(F2)
            return C * R2[i]

        # Level Spacing Normalization Matrix:
        Z = np.zeros((s.N,s.N), dtype='f8')
        for i in range(s.N):
            for j in range(i):
                Z[i,j] = integrate(lambda _x: OffDiag(_x,i,j), a=0.0, b=min(*xMaxLimit[[i,j]]))[0]
                Z[j,i] = Z[i,j]
            Z[i,i] = integrate(lambda _x: MainDiag(_x,i), a=0.0, b=xMaxLimit[i])[0]
        
        # Level Spacing Normalization at Boundaries:
        ZB = np.zeros((1,s.N), dtype='f8')
        ZB[0,:] = [integrate(lambda _x: Boundaries(_x,i), a=0.0, b=xMaxLimit[i])[0] for i in range(s.N)]
        return Z, ZB

    def __xMaxLimit(s, err:float):
        """
        ...
        """

        # xMax = np.array([distr.if1(err) for distr in s.LevelSpacingDist])
        xMax = s.LevelSpacingDist.if1(err)
        
        # FIXME: FUDGE FACTOR!!!
        return 2*xMax
    
    def __FindxMax(s, err:float, xMaxLimit):
        """
        ...
        """

        mthd = 'brentq'

        bounds = [s.MeanLS, np.max(xMaxLimit)]
        if s.N == 1:
            xMax  = root_scalar(lambda x: s.LevelSpacingDist.f0(x) - err, method=mthd, bracket=bounds).root
            xMaxB = root_scalar(lambda x: s.LevelSpacingDist.f1(x)  - err, method=mthd, bracket=bounds).root
        else:
            # Bounding the equation from above:
            def UpperBoundLS(x):
                c, r1, r2 = s.LevelSpacingDist.parts(x)

                # Minimum normalization possible. This is the lower bound on the denominator with regards to the priors:
                Norm_LB = np.amin(s.Z)

                # Finding maximum numerator for the upper bound on the numerator:
                c1 = np.amax(r1*r2)
                c2 = np.amax(r2**2)
                return (c / Norm_LB) * max(c1, c2)

            def UpperBoundLSBoundary(x):
                c, r2 = s.LevelSpacingDist.parts(x)[::2]

                # Minimum normalization possible for lower bound:
                Norm_LB = np.amin(s.ZB)
                return (c / Norm_LB) * np.max(r2) # Bounded above by bounding the priors from above
            
            xMax  = root_scalar(lambda x: UpperBoundLS(x)-err        , method=mthd, bracket=bounds).root
            xMaxB = root_scalar(lambda x: UpperBoundLSBoundary(x)-err, method=mthd, bracket=bounds).root
        return xMax, xMaxB

# =================================================================================================
#     WigBayes Partition / Run Master:
# =================================================================================================

class RunMaster:

    def __init__(self, E, EB, Prior, TPPrior, Freq, LevelSpacingDist, err:float=1e-9):
        self.E       = E
        self.EB      = EB
        self.Prior   = Prior
        self.TPPrior = TPPrior
        self.Freq    = Freq

        self.LevelSpacingDist = LevelSpacingDist
        self.err     = err

        self.L = E.shape[0]
        self.N = Freq.shape[1] - 1

    def MergePartitioner(s, Partitions:list):
        """
        ...
        """

        n = len(Partitions)

        # Merged level-spacing calculation:
        LS   = np.zeros((s.L+2, s.L+2, n), 'f8')
        iMax = np.zeros((s.L+2,     2, n), 'i4')
        for g, group in enumerate(Partitions):
            merge = Merger(s.Freq[:,group], s.LevelSpacingDist[group])
            LS[:,:,g], iMax[:,:,g] = merge.FindLevelSpacings(s.E, s.EB, s.Prior[:,group])

        # Merged prior calculation:
        PriorMerge = np.zeros((s.L, n+1), 'f8')
        for g, group in enumerate(Partitions):
            if hasattr(group, '__iter__'):
                PriorMerge[:,g] = np.sum(s.Prior[:,group], axis=1)
            else:
                PriorMerge[:,g] = s.Prior[:,group]
        PriorMerge[:,-1] = s.Prior[:,-1]

        return LS, iMax, PriorMerge

    def WigBayesPartitionMaster(s, log_total_probability=False, verbose=False):
        """
        ...
        """
        
        # Only 2 spin-groups (merge not needed):
        if s.N == 2:
            if verbose: print(f'Preparing level-spacings')
            LS_g, iMax_g, Prior_g = s.MergePartitioner([[0], [1]])
            if verbose: print(f'Finished level-spacing calculations')
            ENCORE = Encore(Prior_g, LS_g, iMax_g)
            if verbose: print(f'Finished CP calculation')
            Probs = ENCORE.WigBayes()
            if verbose: print(f'Finished WigBayes calculation')
            if log_total_probability:
                ltp = ENCORE.LogTotProb(s.EB, s.Freq, s.TPPrior)
                return Probs, ltp
            else:
                return Probs

        # More than 2 spin-groups (Merge needed):
        else:
            Probs = np.zeros((s.L,3,s.N),dtype='f8')
            if log_total_probability:
                ltp = np.zeros(s.N, dtype='f8')
            for g in range(s.N):
                partition = [[g], [g_ for g_ in range(s.N) if g_ != g]]
                if verbose: print(f'Preparing for Merge group, {g}')
                LS_g, iMax_g, Prior_g = s.MergePartitioner(partition)
                if verbose: print(f'Finished spin-group {g} level-spacing calculation')
                ENCORE = Encore(Prior_g, LS_g, iMax_g)
                if verbose: print(f'Finished spin-group {g} CP calculation')
                Probs[:,:,g] = ENCORE.WigBayes()
                if verbose: print(f'Finished spin-group {g} WigBayes calculation')

                if log_total_probability:
                    # FIXME: I DON'T KNOW THE LOGTOTPROB FUDGE FACTOR FOR MERGED CASES!!!
                    Freq_comb = np.array([s.Freq[0,g], np.sum(s.Freq[0,:])-s.Freq[0,g]]).reshape(1,-1)
                    ltp[g] = ENCORE.LogTotProb(s.EB, Freq_comb, s.TPPrior)

            # Combine probabilities for each merge case:
            NewProbs = s.prob_combinator(Probs)
            if log_total_probability:
                if verbose: print(f'Preparing for Merge group, 1000!!!')
                LS1, iMax1, Prior1 = s.MergePartitioner([tuple(range(s.N))])
                if verbose: print(f'Finished spin-group 1000 level-spacing calculation')
                ENCORE = Encore(Prior1, LS1, iMax1)
                if verbose: print(f'Finished spin-group {g} CP calculation')
                base_LogProb = ENCORE.LogTotProb(s.EB, s.Freq, s.TPPrior)
                NewLTP = s.ltp_combinator(ltp, base_LogProb)
                if verbose: print('Finished!')
                return NewProbs, NewLTP
            else:
                if verbose: print('Finished!')
                return NewProbs

    def prob_combinator(self, Probs):
        """
        ...
        """

        CombProbs = np.zeros((self.L,self.N+1), dtype='f8')
        for g in range(self.N):
            CombProbs[:,g] = Probs[:,1,g]
        CombProbs[:,-1] = np.prod(Probs[:,1,:], axis=1) * self.Prior[:,-1] ** (1-self.N)
        CombProbs[self.Prior[:,-1]==0.0,  -1] = 0.0
        CombProbs /= np.sum(CombProbs, axis=1).reshape((-1,1))
        return CombProbs

    def WigSample(s, Trials:int=1, verbose=False):
        """
        ...
        """
        
        # Only 2 spin-groups (merge not needed):
        if s.N == 2:
            if verbose: print(f'Preparing level-spacings')
            LS_g, iMax_g, Prior_g = s.MergePartitioner([[0], [1]])
            if verbose: print(f'Finished level-spacing calculations')
            ENCORE = Encore(Prior_g, LS_g, iMax_g)
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
                LS_g, iMax_g, Prior_g = s.MergePartitioner(partition)
                if verbose: print(f'Finished spin-group {g} level-spacing calculation')
                ENCORE = Encore(Prior_g, LS_g, iMax_g)
                if verbose: print(f'Finished spin-group {g} CP calculation')
                Probs[:,:,g] = ENCORE.WigBayes()
                if verbose: print(f'Finished spin-group {g} WigBayes calculation')

                if log_total_probability:
                    ltp[g] = ENCORE.LogTotProb(s.EB, s.Freq, s.TPPrior)

            # Combine probabilities for each merge case:
            NewProbs = s.prob_combinator(Probs)
            if log_total_probability:
                if verbose: print(f'Preparing for Merge group, 1000!!!')
                LS1, iMax1, Prior1 = s.MergePartitioner([tuple(range(s.N))])
                if verbose: print(f'Finished spin-group 1000 level-spacing calculation')
                ENCORE = Encore(Prior1, LS1, iMax1)
                if verbose: print(f'Finished spin-group {g} CP calculation')
                base_LogProb = ENCORE.LogTotProb(s.EB, s.Freq, s.TPPrior)
                NewLTP = s.ltp_combinator(ltp, base_LogProb)
                if verbose: print('Finished!')
                return NewProbs, NewLTP
            else:
                if verbose: print('Finished!')
                return NewProbs

    def ltp_combinator(self, LogProbs, base_LogProb):
        """
        ...
        """

        return np.sum(LogProbs) - (self.N-1)*base_LogProb

# ==================================================================================
# Missing/False Resonances:
# ==================================================================================

# ...