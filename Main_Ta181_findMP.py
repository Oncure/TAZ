#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from copy import copy
import math
from scipy.optimize import minimize
from scipy.stats import chisquare

import matplotlib.pyplot as plt

import Levels
import FileReader
import Results
from Encore import Encore
from RMatrix import PTBayes, ReduceFactor, NuclearRadius
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

    Ggm  = [55.00000, 55.00000]
    dfn  = [1, 1]
    dfg  = [250, 250]
    l    = [0, 0]
    j    = [3.0, 4.0]
    pF = 0.0025

    mult = 3 # Missing resonance candidate density

    SGs = SpinGroups.make(l, j)
    Res, Types = FileReader.readSammyPar('/Users/colefritsch/ENCORE/Python_ENCORE/SAMNDF (3).PAR')

    Freq = [1/8.9868, 1/8.3065]
    Gnm  = [44.11355, 33.38697]
    FreqF = pF * np.sum(Freq) / (1-pF)
    MP_True = Resonances.MeanParameters(Freq=Freq, Gnm=Gnm, nDOF=dfn, Ggm=Ggm, gDOF=dfg, A=A, sg=SGs, EB=EB, FreqF=FreqF)
    # Res, Types, Missed_Res, Missed_Types = MP_True.sample()
    def run_ltp_loop(params):
        Gnm = params[2:]
        Freq = params[:2]
        FreqF = pF * np.sum(Freq) / (1-pF)

        print('=================================')
        print(f'Freq1, Freq2, Gnm1, Gnm2, FreqF')
        print(f'{Freq[0]:.8f},{Freq[1]:.8f},{Gnm[0]:.8f},{Gnm[1]:.8f},{FreqF:.8f}')
        print('=================================')

        MP = Resonances.MeanParameters(Freq=Freq, Gnm=Gnm, nDOF=dfn, Ggm=Ggm, gDOF=dfg, A=A, sg=SGs, EB=EB, FreqF=FreqF)
        
        Prior, TPPrior = PTBayes(Res, MP)
        def frac(e, g):
            return ((0.0 + 0.0000022*e**2)/29) * Freq[g]/np.sum(Freq)
        E, Prior = Res.AddMissing(Prior, frac, mult, EB)
        # E = Res.E

        runMaster = Levels.RunMaster(E, MP.EB, Prior, TPPrior, MP.FreqAll)
        Posterior, LTP = runMaster.WigBayesPartitionMaster(True, verbose=True)

        print()
        print(Posterior)
        print()
        print(f'LTP = {LTP:.8E}')
        print()
        Results.PrintScore(Prior, Types, 'PT-only')
        Results.PrintScore(Posterior, Types, 'Wigner+PT')

        with open('/Users/colefritsch/ENCORE/Python_ENCORE/LTP_values_FM.csv', 'a') as file:
            file.write(f'{Freq[0]:.8f},{Freq[1]:.8f},{Gnm[0]:.8f},{Gnm[1]:.8f},{FreqF:.8f},{LTP:.8E}\n')
        
        return -LTP

    IP = [1/8.9868, 1/8.3065, 44.11355, 33.38697]
    # IP = [0.04, 0.231662 - 0.041-0.04, 44.11355+0.9, 33.38697+10.5]
    result = minimize(run_ltp_loop, x0=IP, bounds=[(0,0.5), (0,0.5), (0,75), (0,75)], method='Nelder-Mead', tol=1e-4)
    print(result.x)
    
    # # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Freq1 = 0.04
    # FreqSum = 0.231662 - .041
    # Freq = [Freq1, FreqSum-Freq1]
    # Gnm  = [44.11355+0.9, 33.38697+10.5]
    # # pM   = [0.12547, 0.14404]
    # pM    = [0, 0]
    # FreqF = FreqSum * pF

    # SGs = SpinGroups.make(l, j)
    # MP = Resonances.MeanParameters(Freq=Freq, Gnm=Gnm, nDOF=dfn, Ggm=Ggm, gDOF=dfg, A=A, sg=SGs, EB=EB, FreqF=FreqF)

    # print('====================================')
    # print(f'Mean Level Spacings = {MP.MLS}')
    # print(f'Mean Neutron Widths = {MP.Gnm}')
    # print('====================================')

    # Res, Types = FileReader.readSammyPar('/Users/colefritsch/ENCORE/Python_ENCORE/SAMNDF (3).PAR')
    # # Res, Types, Missed_Res, Missed_Types = MP.sample()

    # Prior, TPPrior = PTBayes(Res, MP)
    # def frac(e, g):
    #     return ((0.0 + 0.0000022*e**2)/29) * Freq[g]/np.sum(Freq)
    # E, Prior = Res.AddMissing(Prior, frac, 3, EB)

    # print(E)
    # print(Prior)

    # Prior = np.zeros((Res.len,3))
    # for i, t in enumerate(Types):
    #     Prior[i,t] = 1


    # runMaster = Levels.RunMaster(Res.E, MP.EB, Prior, TPPrior, MP.FreqAll, 'Missing', MissingFrac=np.array(pM).reshape(1,-1))
    
    
    # # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # # Plotting and GOF:
    # # merge = Merger(s.Freq[:,group], s.LevelSpacingDist, BrodyParam=BP, err=s.err, pM=PM)
    # # LS[:,:,g], iMax[:,:,g] = merge.FindLevelSpacings(s.E, s.EB, s.Prior[:,group])
    # LS   = np.zeros((Res.len+2, Res.len+2, 2), 'f8')
    # iMax = np.zeros((Res.len+2,     2, 2), 'i4')
    # for g in range(2):
    #     merger = Levels.Merger(MP.Freq[:,g], pM=None)
    #     LS[:,:,g], iMax[:,:,g] = merger.FindLevelSpacings(Res.E, MP.EB, Prior[:,g], verbose=True)
    # ENCORE = Encore(Prior, LS, iMax)
    # Assignments = ENCORE.WigSample(1)
    # LSA = np.diff(Res.E[Assignments[:,0] == 0]) / MP.MLS[0,0]
    # LSB = np.diff(Res.E[Assignments[:,0] == 1]) / MP.MLS[0,1]
    # ac = NuclearRadius(A)
    # GnA = Res.Gn[Assignments[:,0] == 0] * ReduceFactor(Res.E[Assignments[:,0] == 0], l=l[0], A=A, ac=ac) / MP.Gnm[0,0]
    # GnB = Res.Gn[Assignments[:,0] == 1] * ReduceFactor(Res.E[Assignments[:,0] == 1], l=l[1], A=A, ac=ac) / MP.Gnm[0,1]

    # def Wig(x):
    #     return np.pi/2 * x * np.exp(-np.pi/4 * x**2)
    # def PT(x):
    #     return np.exp(-x/2) / np.sqrt(2*np.pi*x)


    # X = np.linspace(0, 5, 1000)
    # Y = Wig(X)

    # HA,edges = np.histogram(LSA)
    # HA = HA / np.sum(HA)
    # Xbin = (edges[:-1] + edges[1:])/2
    # chitestA, _ = chisquare(HA, Wig(Xbin))
    # print(f'Wigner Chi-squared test A = {chitestA}')

    # plt.figure()
    # plt.hist(LSA, density=True)
    # plt.plot(X, Y, '-k')
    # plt.xlabel('Normalized Level Spacing', fontsize=16)
    # plt.ylabel('Probability Density', fontsize=16)
    # plt.title('Level spacings for J=3', fontsize=18)
    # plt.tight_layout()
    # plt.show()





    # HB,edges = np.histogram(LSB)
    # HB = HB / np.sum(HB)
    # Xbin = (edges[:-1] + edges[1:])/2
    # chitestB, _ = chisquare(HB, Wig(Xbin))
    # print(f'Wigner Chi-squared test B = {chitestB}')

    # plt.figure()
    # plt.hist(LSB, density=True)
    # plt.plot(X, Y, '-k')
    # plt.xlabel('Normalized Level Spacing', fontsize=16)
    # plt.ylabel('Probability Density', fontsize=16)
    # plt.title('Level spacings for J=4', fontsize=18)
    # plt.tight_layout()
    # plt.show()





    # X = np.linspace(0, 15, 1000)[1:]
    # Y = PT(X)

    # HA,edges = np.histogram(GnA)
    # HA = HA / np.sum(HA)
    # Xbin = (edges[:-1] + edges[1:])/2
    # chitestA, _ = chisquare(HA, PT(Xbin))
    # print(f'Neutron width Chi squared = {chitestA}')

    # plt.figure()
    # plt.hist(GnA, density=True)
    # plt.plot(X, Y, '-k')
    # plt.xlabel('Normalized Neutron Width', fontsize=16)
    # plt.ylabel('Probability Density', fontsize=16)
    # plt.title('Level spacings for J=4', fontsize=18)
    # plt.tight_layout()
    # plt.show()

    # HB,edges = np.histogram(GnB)
    # HB = HB / np.sum(HB)
    # Xbin = (edges[:-1] + edges[1:])/2
    # chitestB, _ = chisquare(HB, PT(Xbin))
    # print(f'Neutron width Chi squared = {chitestB}')

    # plt.figure()
    # plt.hist(GnA, density=True)
    # plt.plot(X, Y, '-k')
    # plt.xlabel('Normalized Neutron Width', fontsize=16)
    # plt.ylabel('Probability Density', fontsize=16)
    # plt.title('Level spacings for J=4', fontsize=18)
    # plt.tight_layout()
    # plt.show()

    # # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # # Getting LTP:
    # runMaster = Levels.RunMaster(E, MP.EB, Prior, TPPrior, MP.FreqAll)
    # Posterior, LTP = runMaster.WigBayesPartitionMaster(True, verbose=True)

    # print()
    # print(Posterior)
    # print()
    # print()
    # print(f'LTP = {LTP:.8E}')
    # print()
    # Results.PrintScore(Prior, Types, 'PT-only')
    # Results.PrintScore(Posterior, Types, 'Wigner+PT')

    # # Results.ConfusionMatrix(Posterior, Types)

    # with open('/Users/colefritsch/ENCORE/Python_ENCORE/LTP_values_FM.csv', 'a') as file:
    #     file.write(f'{Freq[0]:.8f},{Freq[1]:.8f},{Gnm[0]:.8f},{Gnm[1]:.8f},{pM[0]:.8f},{pM[1]:.8f},{FreqF:.8f},{LTP:.8E}\n')


    # ============================================================================================

    # Current issues:
    #   * WigBayes is not working properly with FreqF != 0
    #   * Plots and results have not been fully implemented

