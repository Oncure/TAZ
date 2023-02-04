#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from copy import copy
import math

import Levels
import FileReader
import Results
from Encore import Encore
from RMatrix import PTBayes
import Resonances
from SpinGroups import SpinGroups
# import Statistics

np.set_printoptions(precision=6, edgeitems=9, linewidth=130)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # verbosity controls
    parser.add_argument('-v', dest='verbose', default=False, action='store_true', help="Enable verbose output.")
    
    # Mean parameter information:
    parser.add_argument('--meanParams', type=str, default=None, help='JSON file with mean parameter information.')
    
    # Resonance infomration:
    parser.add_argument('--ENDFreader', type=str, default=None, help='ENDF file with the resonance information.')
    parser.add_argument('--sample', type=bool, default=False, action='store_true', help='Samples resonances from the mean parameter file.')

    # Running:
    parser.add_argument('--run', type=str, default=None, choices=['score', 'sample'], help='What sequence to run on the resonances.')

    # Results:
    parser.add_argument('--saveResults', type=str, default=None, help='Saves the probability scores or resampling to the given file.')
    parser.add_argument('--corrPlot', type=str, default=None, help='Gives a correlation plot on the probabilities for checking.')

    # ...

    return parser.parse_args()

#######################################################################################################################
# 							Main								#
#######################################################################################################################
if __name__ == "__main__":
    #args = parse_args()

    A    = 181
    EB   = (0, 2000)

    FreqF = 0
    Freq = [0.125484916901, 0.157559977484]
    Gnm  = [48.74477, 48.74477]
    Ggm  = [55.00000, 56.51515]
    dfn  = [1, 1]
    dfg  = [572.45309, 572.45309]
    l    = [0, 0]
    j    = [3.0, 4.0]

    # Freq = [1.5, 3, 2, 0]
    # Gnm  = [5, 4, 3]
    # Ggm  = [1, 1, 1]
    # dfn  = [1, 1, 1]
    # dfg  = [100, 100, 100]
    # l    = [0, 1, 1]
    # j    = [0.5, 0.5, 1.5]

    SGs = SpinGroups.make(l, j)
    MP_True  = Resonances.MeanParameters(Freq = Freq, Gnm = Gnm, nDOF = dfn, Ggm = Ggm, gDOF = dfg, A = A, sg = SGs, EB = EB, FreqF=0.0)
    MP_Guess = Resonances.MeanParameters(Freq = Freq, Gnm = Gnm, nDOF = dfn, Ggm = Ggm, gDOF = dfg, A = A, sg = SGs, EB = EB, FreqF=FreqF)

    N = 1
    PosteriorAll = np.zeros((0,3))
    TypesAll     = np.zeros((0,))
    for itr in range(N):
        Res, Types, Missed_Res, Missed_Types = MP_True.sample()
        Prior, TPPrior = PTBayes(Res, MP_Guess)
        LS0, Imax0 = Levels.Merger(MP_Guess.Freq[0,0]).FindLevelSpacings(Res.E, MP_Guess.EB, Prior[:,0].reshape(-1,1))
        LS1, Imax1 = Levels.Merger(MP_Guess.Freq[0,1]).FindLevelSpacings(Res.E, MP_Guess.EB, Prior[:,1].reshape(-1,1))
        LS = np.concatenate((LS0.reshape(*LS0.shape,1), LS1.reshape(*LS1.shape,1)), axis=2)
        Imax = np.concatenate((Imax0.reshape(-1,2,1),Imax1.reshape(-1,2,1)), axis=1)
        encore = Encore(Prior, LS, Imax)
        Sample = encore.WigSample(60)
        # runMaster = Levels.RunMaster(Res.E, MP_True.EB, Prior, TPPrior, MP_True.FreqAll)
        # Posterior = runMaster.WigBayesPartitionMaster(True, verbose=True)[0]
        # PosteriorAll = np.concatenate((PosteriorAll, Posterior), axis=0)
        # TypesAll = np.concatenate((TypesAll, Types), axis=0)

    LSs = np.array([ls for k in range(60) for ls in np.diff(Res.E[Sample[:,k] == 0])])
    print(LSs.shape)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(LSs, bins=50)
    plt.tight_layout()
    plt.show()
    
    # print(f'Trivial Score = {100*np.max(Freq)/np.sum(Freq):.2f}%')
    # Results.PrintScore(Prior, Types, 'Prior')
    # Results.PrintScore(Posterior, Types, 'Posterior')
    # Results.ProbCorrPlot(Posterior, Types, ('+3.0', '+4.0'))

    # ============================================================================================

    # Current issues:
    #   * Merging has not been fully implemented
    #   * "WigSample" has not been tested
    #   * Plots and results have not been fully implemented


