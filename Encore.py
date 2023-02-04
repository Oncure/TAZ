#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np

class Encore:
    """
    ...

    Variables:
    L     :: 32b   int              | Total number of resonances
    N     ::  8b   int              | Total number of spin groups (excluding false group)
    Prior :: 64b float [L+2 N+1]    | Prior probabilities on each resonance (with boundaries) and
                                    |   spin group (with false group)
    F     :: 64b float [L+2 L+2 N]  | All of the level spacings from index 0 to index 1 with spin
                                    |   group, index 2
    iMax  :: 32b   int [L+2  2  N]  | Maximum index for approximation threshold in each direction
    
    CP    :: 64b float [L+2 L+2 2]  | 'Cumulative Probabilities' which are the building blocks for
                                    |   Bayesian probability calculations and probabilistic
                                    |   sampling
    PW    :: 64b float [L+2]        | 'Placement Weights' which ensure CP does not grow or decay
                                    |   exponententially beyond the limits of floating-point
                                    |   numbers
    TP    :: 64b float              | 'Total Probability' which is the probability density for the
                                    |   current arrangements of resonance energies (with PW
                                    |   removed)
    """

    TP_error_threshold = 1.0 # % (percent)

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

    # ==================================================================================
    # Constructor
    # ==================================================================================
    def __init__(self, Prior, F, iMax):
        """
        Initialization ...

        ...
        """

        # Initialization:
        self.N = F.shape[2]
        if self.N not in (1,2):
            raise NotImplementedError("ENCORE can only handle 1 or 2 spin-groups and a false group at the moment.")
        self.L = np.int32(Prior.shape[0])
        self.Prior = np.ones((self.L+2,self.N+1))
        # self.Prior = np.ones((self.L+2,self.N+1), dtype='f8')
        self.Prior[1:-1,:] = Prior
        self.F    = F
        self.iMax = iMax

        # Calculating "CP" and "TP":
        if   self.N == 1:   self.__MakeCP1()
        elif self.N == 2:   self.__MakeCP2()
        self.__MakeTP()

    # ==================================================================================
    # Make CP (1 spin groups and false group)
    # ==================================================================================
    def __MakeCP1(s):
        """
        ...
        """

        raise NotImplementedError('MakeCP1 has not been fully tested yet.')

        L = s.L
        s.CP = np.zeros((L+2,2), dtype='f8')
        s.PW = np.zeros(L+2, dtype='f8')

        #FIXME: these "s.CP" and "s.PW" need checks. Do the numbers even matter?

        # Trivial CP values:
        s.CP[ 0, 0] = 1.0
        s.CP[ 1, 0] = 1.0
        s.CP[-1, 1] = 1.0
        s.CP[-2, 1] = 1.0

        s.PW[[0,-1]] = 1.0
        s.PW[[1,-2]] = 2.0

        for iL in range(2,L+2):
            iR = L+1-iL

            # =================================================================================
            # Left-Hand Side:

            s.CP[iL,0] = s.CP[iL-1,0] * s.PW[iL-1] * s.Prior[iL-1,0] * s.F[iL-1,iL,0]
            Mchain = 1.0
            for j in range(iL-2,s.iMax[iL,0,0]-1,-1):
                Mchain *= s.Prior[j+1,1]
                if Mchain == 0.0:   break   # Potential computation time improvement
                s.CP[iL,0] += s.CP[j,0] * Mchain * s.PW[j] * s.Prior[j,0] * s.F[j,iL,0]
            
            # =================================================================================
            # Right-Hand Side:

            s.CP[iR,1] = s.CP[iR+1,1] * s.PW[iR+1] * s.Prior[iR+1,0]* s.F[iL-1,iL,0]
            Mchain = 1.0
            for j in range(iR+2,s.iMax[iR,1,0]+1):
                Mchain *= s.Prior[j+1,1]
                if Mchain == 0.0:   break   # Potential computation time improvement
                s.CP[iR,1] += s.CP[j,1] * Mchain * s.PW[j] * s.Prior[j,0] * s.F[iR,j,0]
            
            # =====================================================================================
            # Finding PW:

            if 2*iL <= L+1:
                if s.CP[iL,0] == 0.0:
                    raise RuntimeError(s.PW_INF_ERROR)
                s.PW[iL] = 1.0 / s.CP[iL,0]

            if 2*iL <= L:
                if s.CP[iR,1] == 0.0:
                    raise RuntimeError(s.PW_INF_ERROR)
                s.PW[iR] = 1.0 / s.CP[iR,1]

    # ==================================================================================
    # Make CP (2 spin groups and false group)
    # ==================================================================================
    def __MakeCP2(s):
        """
        'MakeCP2' calculates 'CP' -- the 'Cumulative Probabilities'. 'CP' is a
        three dimensional array that takes the form, 'CP(iA,iB,dir)', where
        'iA' and 'iB' are the indices of 'A' and 'B' and 'dir' is the direction.
        'CP(iA,iB,dir)' is the probability of the spin group assignment
        distribution on the set of resonances with indices less that or equal to
        'max(iA,iB)' for 'dir = 1' or greater than or equal to 'min(iA,iB)' for
        'dir = 2' that meet the following conditions:
            (1) 'iA' is the index of the right-most (dir = 1)
                or left-most (dir = 2) resonance with spin group A.
            (2) 'iB' is the index of the right-most (dir = 1)
                or left-most (dir = 2) resonance with spin group B.

        'CP' is calculated through an iterative process, hince the name. The true
        value of 'CP' has the tendency to exponentially grow or decay through
        each iteration, therefore, we use 'Placement Weights' ('PW') to
        renormalize the value to an acceptable floating-point range.

        ...
        """

        L = s.L
        s.CP = np.zeros((L+2,L+2,2), dtype='f8')
        s.PW = np.zeros(L+2, dtype='f8')

        # Trivial CP values:
        s.CP[ 0, 0, 0] = 1.0 ;      s.CP[-1, -1, 1] = 1.0
        # s.CP[ 1, 0, 0], s.CP[ 0, 1, 0] = s.F[ 0, 1, :]
        # s.CP[-2,-1, 1], s.CP[-1,-2, 1] = s.F[-2,-1, :]

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
            for t in range(2):
                # =================================================================================
                # Left-Hand Side:

                idx = range(i1L-1, jMax[t,0]-1, -1)
                ls = s.F[idx, i1L, t]
                for i2L in range(i1L-1, i2Max[t,0]-1, -1):

                    # Finding J1 and J2:
                    if (jMax[t,0] > i2L) or (i2L == 0):
                        j1 = idx ;      ls1 = ls
                        j2 = []  ;      ls2 = []
                    else:
                        j1 = range(i1L-1, i2L, -1)         ;    ls1 = ls[:i1L-i2L-1]
                        j2 = range(i2L-1, jMax[t,0]-1, -1) ;    ls2 = ls[i1L-i2L:]
                        
                    # New CP Iteration:
                    s.__CPIter(j1,j2,ls1,ls2,i1L,i2L,t,1)

                # =================================================================================
                # Right-Hand Side:

                idx = range(i1R+1, jMax[t,1]+1)
                ls  = s.F[i1R, idx, t]
                for i2R in range(i1R+1, i2Max[t,1]+1):

                    # Finding J1 and J2:
                    if (jMax[t,1] < i2R) or (i2R == L+1):
                        j1 = idx ;      ls1 = ls
                        j2 = []  ;      ls2 = []
                    else:
                        j1 = range(i1R+1, i2R)         ;    ls1 = ls[:i2R-i1R-1]
                        j2 = range(i2R+1, jMax[t,1]+1) ;    ls2 = ls[i2R-i1R:]

                    # New CP Iteration:
                    s.__CPIter(j1,j2,ls1,ls2,i1R,i2R,t,-1)

            # =====================================================================================
            # Finding PW:

            # Left-Hand Side PW Calculation:
            if 2*i1L <= L+1:
                idxB = range(i2Max[0,0], i1L)
                idxA = range(i2Max[1,0], i1L)
                denom = np.sum(s.CP[i1L,idxB,0]) + np.sum(s.CP[idxA,i1L,0])
                if denom == 0.0:
                    raise RuntimeError()
                    # raise RuntimeError(s.PW_INF_ERROR.format(side='left', idx=i1L, AList=s.CP[idxA,i1L,0], BList=s.CP[i1L,idxB,0]))
                s.PW[i1L] = 1.0 / denom

            # Right-Hand Side PW Calculation:
            if 2*i1L <= L:
                idxB = range(i1R+1, i2Max[0,1]+1)
                idxA = range(i1R+1, i2Max[1,1]+1)
                denom = np.sum(s.CP[i1R,idxB,1]) + np.sum(s.CP[idxA,i1R,1])
                if denom == 0.0:
                    raise RuntimeError()
                    # raise RuntimeError(s.PW_INF_ERROR.format(side='right', idx=i1R, AList=s.CP[idxA,i1R,1], BList=s.CP[i1R,idxB,1]))
                s.PW[i1R] = 1.0 / denom

    # ==================================================================================
    # CP Iteration Function
    # ==================================================================================
    def __CPIter(s, J1, J2, ls1, ls2, i1, i2, t, sgn):
        """
        CPIter is a function that calculates the next 'CP' element using an
        iterative process. ...
        
        ...
        """

        Dir = int(sgn == -1)
        cpOut = 0.0

        # Preceeding i2:
        if J1:
            if t:   prod1 = ls1 * s.CP[i2,J1,Dir]
            else:   prod1 = ls1 * s.CP[J1,i2,Dir]
            
            j = J1[0]
            Mchain = s.PW[j]
            cpOut = Mchain * prod1[0] * s.Prior[j,t]
            for j, p1 in zip(J1[1:], prod1[1:]):
                Mchain *= s.PW[j] * s.Prior[j+sgn,2]
                if Mchain == 0.0:   break   # Potential computation time improvement
                cpOut += Mchain * p1 * s.Prior[j,t]

        # Behind i2:
        if J2:
            if i2+sgn == i1:    Mchain  = s.PW[i2] * s.Prior[i2,1-t]
            elif i1 == i2:      Mchain  = 1.0
            else:               Mchain *= s.PW[i2] * s.Prior[i2,1-t] * s.Prior[i2+sgn,2]

            if t:   cpOut += Mchain * np.sum(ls2 * s.CP[i2,J2,Dir])
            else:   cpOut += Mchain * np.sum(ls2 * s.CP[J2,i2,Dir])

        # Error Checks:
        if   cpOut == math.inf:
            raise RuntimeWarning(s.CP_BAD_ERROR.format('Inf'))
        elif cpOut == math.nan:
            raise RuntimeWarning(s.CP_BAD_ERROR.format('NaN'))
        
        # Assigning CP:
        if t:   s.CP[i2,i1,Dir] = cpOut
        else:   s.CP[i1,i2,Dir] = cpOut

    # ==================================================================================
    # Find Total Probability Factor
    # ==================================================================================
    def __MakeTP(s):
        """
        MakeTP is a function that calculates the total probability, 'TP' using
        'CP' and 'PW' found in the function, 'MakeCP'. TP can be found by
        calculating 'CP[ -1, -1, 0]' or 'CP[ 0, 0, 1]'. Each can be performed
        by summing 'CP' across the first or second spin-group. Therefore there
        are a four ways to calculate TP. We accept the mean of the four
        calculation as the total probability and find the percent error for
        error checking purposes.
        """

        # =====================================================================================
        # Finding CP[ -1, -1, 0] = CP[ 0, 0, 1] = TP:

        # for t in range(2):
        #     J2  = range(s.L, s.iMax[s.L+1,1,t]-1, -1)
        #     ls2 = s.F[J2, s.L+1, t]
        #     s.__CPIter([], J2, [], ls2, s.L+1, s.L+1, t, 1)

        #     J2  = range(1, s.iMax[0,0,t]-1)
        #     ls2 = s.F[0, J2, t]
        #     s.__CPIter([], J2, [], ls2, 0, 0, t, -1)

        if s.N == 2:
            TPs = np.empty((2,2), dtype='f8')
            TPs[0,0] = np.sum(s.F[:-1, -1, 0] * s.CP[:-1, -1, 0])
            TPs[1,0] = np.sum(s.F[:-1, -1, 1] * s.CP[-1, :-1, 0])
            TPs[0,1] = np.sum(s.F[  0, 1:, 0] * s.CP[ 1:,  0, 1])
            TPs[1,1] = np.sum(s.F[  0, 1:, 1] * s.CP[ 0,  1:, 1])
            s.CP[-1,-1,0], s.CP[0,0,1] = np.mean(TPs, axis=0)

        # =====================================================================================

        if   s.N == 1:      TP1 = s.CP[0,1];    TP2 = s.CP[-1,0]
        elif s.N == 2:      TP1 = s.CP[0,0,1];  TP2 = s.CP[-1,-1,0]

        # print(np.argmax(s.Prior[[1,-2],:], axis=1))
        # print(TPs)
        
        # Taking the average of the two sides for the Total Probability:
        s.TP = (TP1 + TP2) / 2

        # If the two sides are past a threshold percent difference, make a warning:
        percent_error = 100.0 * abs(TP1 - TP2) / s.TP
        if percent_error > s.TP_error_threshold:
            print(RuntimeWarning(s.TP_PERCENT_ERROR.format(p_error=percent_error, p_thres=s.TP_error_threshold)))

        # print(s.TP)

    # ==================================================================================
    # Level Spacing Bayes Theorem
    # ==================================================================================
    def WigBayes(s):
        """
        WigBayes finds the probability on the spin group assignment of
        each resonance. This is done using 'CP' probabilities and
        Bayes' Theorem. ...

        ...
        """

        # Error checking:
        if s.N not in (1, 2):
            raise NotImplementedError('Currently "WigBayes" can only handle 1 or 2 spin-groups and a False group. \
                                        Merge spin-groups to get probabilities on more spin-groups.')

        L = s.L
        sp = np.zeros((L,s.N), dtype='f8') # No longer need the edge cases, so the bounds are 0:L instead of 0:L+2
        if s.N == 1:
            sp = s.CP[1:-1,0] * s.CP[1:-1,1]
        elif s.N == 2:
            for t in range(2):
                for iL in range(L):
                    iRMax = s.iMax[iL,0,1-t]
                    ls = s.F[iL, iL+2:iRMax+1, 1-t]
                    for i in range(iL+1, iRMax):
                        idxR = range(i+1, iRMax+1)
                        if t:   sp[i-1,1] += s.CP[iL,i,0] * np.sum(ls[i-iL-1:] * s.CP[idxR,i,1])
                        else:   sp[i-1,0] += s.CP[i,iL,0] * np.sum(ls[i-iL-1:] * s.CP[i,idxR,1])
                    # if t:   sp[iL:iRMax-1,1] += np.array([s.CP[iL,i,0] * np.sum(ls[i-iL-1:] * s.CP[i+1:iRMax+1,i,1]) for i in range(iL+1, iRMax)])
                    # else:   sp[iL:iRMax-1,0] += np.array([s.CP[i,iL,0] * np.sum(ls[i-iL-1:] * s.CP[i,i+1:iRMax+1,1]) for i in range(iL+1, iRMax)])

        # print(np.mean(np.sum(sp *  s.Prior[1:-1,:2] * s.PW[1:-1].reshape(-1,1), axis=1)))
        # print()
        # print(s.CP[:7,:7,0])

        # Factors shared throughout the sum including the normalization factor, "TP":
        sp *= s.Prior[1:-1,:2] * s.PW[1:-1].reshape(-1,1) / s.TP

        # False probabilities appended by taking one minus the other probabilities:
        Probs = np.append(sp, (1.0 - np.sum(sp, axis=1)).reshape(-1,1), 1)
        return Probs

    # ==================================================================================
    # Sample Spacing-Updated Spin-Groups
    # ==================================================================================
    def WigSample(s, Trials:int):
        """
        WigSample randomly samples spin group assignments based its
        bayesian probability. This is done by sampling spin group
        assignments for each resonance one at a time from low to high
        energy, indexed by 'i'. Using the lower energy resonaces with
        determined spin group assignments and the 'CP' probabilities
        representing the possibilities for the the higher energy
        resonances, we can get a probability on resonance 'i' for each
        spin group.
        
        ...
        """

        # Error checking:
        if s.N != 2:
            raise NotImplementedError("Currently sampling only works with 2 spin-groups.")
        # elif (s.Prior[:,-1] == 0.0).any:
        #     raise NotImplementedError("Currently spin groups cannot be sampled with false resonance possibilities.")

        L = s.L
        ssg  = np.zeros((L,Trials),dtype='u1') # The sampled spin-groups

        Last = np.zeros((2,Trials),dtype='u4') # Latest indices for each spingroup, for each trial
        Rand = np.random.rand(L,Trials)        # Random numbers used to sample the spingroups

        iMax = int(3)
        sp = np.zeros((Trials, 3),dtype='f8')
        for i in range(L):
            i1 = i + 1

            # FIXME: THE APROXIMATION THRESHOLD NEEDS UPDATING!!!
            while any(s.F[i1,iMax,:]) and (iMax <= L):
                iMax += 1
            Idx = range(i1+1,iMax+1)
            l = iMax - i1

            ls1 = np.zeros((1,Trials,2),dtype='f8')
            ls2 = np.zeros((l,Trials,2),dtype='f8')
            for t in range(2):
                ls1[0,:,t] = s.F[Last[t,:],i1,t].T
                for tr in range(Trials):
                    ls2[:,tr,t] = s.F[Last[1-t,tr],Idx,1-t]
            cp       = np.concatenate((s.CP[i1,Idx,1].reshape(-1,1,1), s.CP[Idx,i1,1].reshape(-1,1,1)), axis=2)
            sp[:,:2] = (s.Prior[i1,:2].reshape(1,1,2)*ls1*np.sum(ls2*cp, axis=0))[0,:,:]
            
            sp[:,2]  = np.array([s.Prior[i1,2] * np.sum(ls2[:,tr,0].reshape(-1,1) \
                                                      * ls2[:,tr,1].reshape(1,-1) \
                                                      * s.CP[Idx,Idx,1], axis=(0,1)) \
                                                        for tr in range(Trials)])

            sp_sum = np.sum(sp, axis=1)
            p2       = sp[:,2] / sp_sum
            p1       = sp[:,1] / sp_sum + p2

            # Sampling New Spin-groups:
            ssg[i,:] =  np.int_(Rand[i,:] <= p2) + np.int_(Rand[i,:] <= p1)

            # Rewriting the last used resonance of spin group sg:
            for tr,sg in enumerate(ssg[i,:]):
                if sg != 2:     Last[sg,tr] = i1

        return ssg
    
    # ==================================================================================
    # Sample Probability
    # ==================================================================================
    def ProbOfSample(s, SGs) -> float:
        """
        Find the probability of sampling the given spin-group assignments.
        """

        # NOTE: This needs checking!!!
        Prob = 1.0
        if s.N == 2:
            LastRes = np.zeros(2,'i4')
            for i, sg in enumerate(SGs):
                j = LastRes[sg]
                Prob *= s.Prior[i+1,sg] * s.PW[i+1] * s.F[j,i+1,sg]
                LastRes[sg] = i+1
            Prob *= s.F[LastRes[0],-1,0] * s.F[LastRes[1],-1,1]
        
        elif s.N == 1:
            raise NotImplementedError('This feature has not been implemented yet.')
        
        return Prob / s.TP

    # ==================================================================================
    # Log Total Probability
    # ==================================================================================
    def LogTotProb(self, EB: tuple, Freq: float, TPPrior = None) -> float:
        """
        LogTotProb gives a criterion for the likelihood of sampling the given energies
        and widths, assuming correct information was given (i.e. mean level spacing,
        mean neutron width, etc.). This can be used with gradient descent to optimize
        mean parameters.
        """

        dE = EB[1]-EB[0]
        log_total_probability = math.log(self.TP) - np.sum(np.log(self.PW)) - Freq[0,-1] * dE
        
        # TP Prior component:
        if TPPrior is not None:
            log_total_probability += np.sum(np.log(TPPrior))

        # Turn to into log10:
        log_total_probability / math.log(10)

        return log_total_probability

    @staticmethod
    def LogTotProbString(log_total_probability:float, sigfigs:int=3) -> str:
        """
        Prints the total probability given the log_total probabilty. This sometimes cannot be done
        directly since the exponential term can sometimes exceed floating-point limitations.
        """
        out_str = f'{{0:.{sigfigs}f}}e{{1}}'
        exponent    = math.floor(log_total_probability)
        significand = 10 ** (log_total_probability % 1.0)
        return out_str.format(significand, exponent)