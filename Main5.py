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
from Distributions import Distribution, Distributions
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

    A    = 238
    EB   = (1000, 1000+50)

    FreqF = 1
    Freq = [3, 4]
    # Freq = [1, 1]
    Gnm  = [3, 3]
    Ggm  = [1, 1]
    dfn  = [1, 1]
    dfg  = [100, 100]
    l    = [0, 0]
    j    = [0.5, 0.5]

    # FreqF = 0
    # Freq = [2, 1]
    # Gnm  = [1, 1]
    # Ggm  = [1, 1]
    # dfn  = [1, 1]
    # dfg  = [100, 100]
    # l    = [0, 0]
    # j    = [0.5, 0.5]

    # FreqF = 0
    # Freq = [1, 5, 2]
    # Gnm  = [5, 4, 3]
    # Ggm  = [1, 1, 1]
    # dfn  = [1, 1, 1]
    # dfg  = [100, 100, 100]
    # l    = [0, 0, 0]
    # j    = [0.5, 0.5, 1.5]

    distr = Distributions.wigner(Freq)


    SGs = SpinGroups.make(l, j)
    MP_True = Resonances.MeanParameters(Freq=Freq, Gnm=Gnm, nDOF=dfn, Ggm=Ggm, gDOF=dfg, A=A, sg=SGs, EB=EB, FreqF=FreqF)
    Res, Types, Missed_Res, Missed_Types = MP_True.sample()

   
    MP_Guess = Resonances.MeanParameters(Freq = Freq, Gnm = Gnm, nDOF = dfn, Ggm = Ggm, gDOF = dfg, A = A, sg = SGs, EB = EB, FreqF = FreqF)
    Prior, TPPrior = PTBayes(Res, MP_Guess)
    
    # for g in range(len(Freq)+1):
    #     Prior[:,g] = abs((Types == g))

    # print(np.argmax(Prior[[0,-1],:], axis=1))
    # Prior[0,:] = [0.5,0.5,0]
    # Prior[-1,:] = [1,0,0]
    # Prior[0,:] = [1,0,0]
    # Prior[:,:-1] = 0.5
    # Prior[0,-1] = 0;        Prior[0,:] /= np.sum(Prior[0,:])
    # Prior[-1,-1] = 0;       Prior[-1,:] /= np.sum(Prior[-1,:])

    runMaster = Levels.RunMaster(Res.E, MP_Guess.EB, Prior, TPPrior, MP_Guess.FreqAll)
    Posterior, LTP = runMaster.WigBayesPartitionMaster(True, verbose=True)
    print()
    # print(Posterior)
    print()
    print(f'L = {Res.len}')

    # Results.PrintScore(Prior, Types, 'Prior')
    # Results.PrintScore(Posterior, Types, 'ENCORE')
    # Results.ProbCorrPlot(Posterior, Types, ['A', 'B', 'F'])








    # print(Probs)

    # Results.PrintScore(Prior, Types, 'Best-Guess Porter-Thomas', metric='best guess')
    # Results.PrintScore(Posterior, Types, 'Best-Guess ENCORE', metric='best guess')

    # Results.PrintScore(Prior, Types, 'Prob-Score Porter-Thomas', metric='probability score')
    # Results.PrintScore(Posterior, Types, 'Prob-Score ENCORE', metric='probability score')
    # Results.ProbCorrPlot(Posterior, Types, ['A', 'B', 'C', 'False'])

    # Merge = Levels.Merger(MP_True.Freq[:,:-1])
    # PriorM, iMaxM = Merge.FindLevelSpacings(Res.E, Res.EB, Prior[:,:-1], True)





    # Levels = Levels.Levels(Res.E, Res.EB, Prior, MP_True.Freq)

    # LS = Levels.LevelSpacings(1e-9)

    # ENCORE = Encore(Prior, LS)

    # if any(ENCORE.PW == np.nan):
    #     raise RuntimeError('ENCORE failed to run properly')
    # Probs = ENCORE.WigBayes()

    # Sample = ENCORE.WigSample(2)

    # ltp = ENCORE.LogTotProb(EB, Freq[-1])
    # print(f'Log Total Probability = {ltp:.5e}')

    # Results.PrintScore(Probs, Types)
    # Results.ProbCorrPlot(Probs, Types, ['A', 'B', 'F'])

    # print(np.array([Probs[j,Types[j]] for j in range(ENCORE.Length)]))

    # print(f'L = {ENCORE.Length}')
    # # print(f'SP = {Probs}')
    # # print(f'TP = {ENCORE.TP}')
    # print(f'RATIO = {1-Probs[:,-1]}')

    # print(f'Probs | Answers =\n{np.concatenate((Probs[:,:-1], Types.reshape(-1,1)), axis=1)}')


    # ============================================================================================

    # Current issues:
    #   * WigBayes is not working properly with FreqF != 0
    #   * Plots and results have not been fully implemented

