#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from copy import copy
import math

# import Levels_rework2 as Levels
import Levels
import FileReader
import Results
from Encore import Encore
from RMatrix import PTBayes
import Resonances
from SpinGroups import SpinGroups

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

    A  = 181
    EB = (1e-5,2550)

    pM   = [0.12547, 0.14404]
    FreqF = 1/15
    Freq = [1/8.9868, 1/8.3065]
    Gnm  = [44.11355, 33.38697]
    Ggm  = [55.00000, 55.00000]
    dfn  = [1, 1]
    dfg  = [250, 250]
    l    = [0, 0]
    j    = [3.0, 4.0]

    SGs = SpinGroups.make(l, j)
    MP = Resonances.MeanParameters(Freq=Freq, Gnm=Gnm, nDOF=dfn, Ggm=Ggm, gDOF=dfg, A=A, sg=SGs, EB=EB, FreqF=FreqF)

    # Res, Types = FileReader.readSammyPar('/Users/colefritsch/ENCORE/Python_ENCORE/SAMNDF (3).PAR')
    Res, Types, Missed_Res, Missed_Types = MP.sample()

    Prior, TPPrior = PTBayes(Res, MP)

    runMaster = Levels.RunMaster(Res.E, MP.EB, Prior, TPPrior, MP.FreqAll, 'Missing', MissingFrac=np.array(pM).reshape(1,-1))
    # runMaster = Levels.RunMaster(Res.E, MP.EB, Prior, TPPrior, MP.FreqAll)
    Posterior, LTP = runMaster.WigBayes(True, verbose=True)

    print()
    print(Posterior)
    print()
    print(f'L = {Res.len}')

    Results.PrintScore(Prior, Types, 'PT-only')
    Results.PrintScore(Posterior, Types, 'Wigner+PT')
    # Results.PrintPredScore(Prior, 'Prior')
    # Results.PrintPredScore(Posterior, 'ENCORE')

    # Results.PrintScore(Prior, Types, 'Prior', 'probability score')
    # Results.PrintScore(Posterior, Types, 'ENCORE', 'probability score')
    # Results.PrintPredScore(Prior, 'Prior', 'probability score')
    # Results.PrintPredScore(Posterior, 'ENCORE', 'probability score')

    Results.ConfusionMatrix(Posterior, Types)

    Results.ProbCorrPlot(Posterior, Types, ['A', 'B', 'F'])



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

