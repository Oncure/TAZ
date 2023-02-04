import numpy as np
import pandas as pd

import SpinGroups
import RMatrix
import Distributions

__doc__ = """
This file keeps the "MeanParameters" class and "Resonances" class. The "MeanParameters" class
contains all relevent information that is independent of individual resonances, such as isotopic
mass, isotope spin, mean level-spacing, etc. The "Resonances" class contains resonance-specific
information such as Energy, widths, and spin-group assignments.
"""

# =================================================================================================
#    Mean Parameters:
# =================================================================================================
class MeanParameters:
    """
    MeanParameters is a class that contains information about a particular reaction, such as the
    target nuclei mass, nuclear radius, spin, mean level spacing, and more.

    I        :: int   :: target particle spin
    Z        :: int   :: target atomic number
    A        :: int   :: target atomic mass
    mass     :: float :: target nuclei mass
    ac       :: float :: target nuclear radius
    EB       :: tuple :: Energy range for evaluation       NOTE: I should move this elsewhere
    FreqF    :: float :: False resonance level density
    Gn_trunc :: float :: Lowest recordable neutron width
    MissFrac :: float :: Fraction of Resonances that have been missed
    sg       :: list  :: Spingroups for the reaction
    Freq     :: list  :: Resonance level densities for each spingroup
    FreqAll  :: list  :: Resonance and false level densities
    MLS      :: list  :: Resonance mean level spacings for each spingroup
    Gnm      :: list  :: Resonance mean neutron widths for each spingroup
    nDOF     :: list  :: Resonance neutron width degrees of freedom for each spingroup
    Ggm      :: list  :: Resonance mean gamma width for each spingroup
    gDOF     :: list  :: Resonance gamma width degrees of freedom for each spingroup
    w        :: list  :: Brody resonance parameter
    """

    DEFAULT_GDOF = 500

    def __init__(self, **kwargs):
        """
        Initializes reaction parameters with keyword arguments.
        """

        ParamNames = set(kwargs.keys())
        def ParamsIn(*Names: str, **options: str):
            if 'type' in options.keys():    ParamType = options['type']
            else:                           ParamType = 'array1'
            ParamSet = (ParamNames & set(Names))
            ParamList = list(ParamSet)
            if len(ParamList) >= 2:
                ParamStr = '\"' + '\", \"'.join(ParamList[:-1]) + f', and \"{ParamList[-1]}\"'
                raise ValueError(f'Cannot accept multiple parameters, {ParamStr}.')
            elif len(ParamList) == 1:
                param = ParamList[0]
                if ParamType == 'array1':
                    paramValue = np.array(kwargs[param]).reshape(1,-1)
                elif ParamType == 'float':
                    paramValue = float(kwargs[param])
                elif ParamType == 'int':
                    paramValue = int(kwargs[param])
                elif ParamType == 'halfint':
                    paramValue = SpinGroups.halfint(kwargs[param])
                elif ParamType == 'tuple':
                    paramValue = tuple(kwargs[param])
                elif ParamType == 'pass':
                    paramValue = kwargs[param]
                return paramValue, True
            else:
                return None, False

        # Isotope Spin:
        self.I, IExists = ParamsIn('I', 'S', 'spin', 'isotope_spin', type='halfint')

        # Atomic Number:
        self.Z, ZExists = ParamsIn('Z', 'atomic_number', type='int')

        # Atomic Mass:
        self.A, AExists = ParamsIn('A', 'atomic_mass', type='int')

        # Mass:
        value, massExists = ParamsIn('mass', 'Mass', 'm', 'M', type='float')
        if massExists:  self.mass = value
        elif AExists:   self.mass = float(self.A)

        # Atomic Radius:
        value, acExists = ParamsIn('Ac', 'ac', 'atomic_radius', 'scatter_radius', type='float')
        if acExists:    self.ac = value
        elif AExists:   self.ac = RMatrix.NuclearRadius(self.A)

        # Energy Range:
        value, EBExists = ParamsIn('EB', 'energy_bounds', 'energy_range', type='tuple')
        if EBExists:
            self.EB = value
            if len(self.EB) != 2:
                raise ValueError('"EB" can only have two values for an interval')
            elif self.EB[0] > self.EB[1]:
                raise ValueError('"EB" must be a valid increasing interval')

        # False Frequency:
        value, FalseFreqExists = ParamsIn('freqF', 'FreqF', 'false_frequency', type='float')
        self.FreqF = value if FalseFreqExists else 0.0

        # Truncation Width:
        value, truncExists = ParamsIn('trunc', 'Gn_trunc', 'truncation_width', type='float')
        self.Gn_trunc = value if truncExists else 0.0

        # Missing Fraction:
        self.MissFrac = Distributions.FractionMissing(self.Gn_trunc)

        # Spin-Groups:
        self.sg, sgExists = ParamsIn('sg', 'SG', 'spin-group', type='pass')

        # Frequencies:
        self.Freq, FreqExists = ParamsIn('freq', 'Freq', 'frequency', 'Frequency')

        # Mean Level Spacings:
        value, MLSExists = ParamsIn('mean_spacing', 'mean_level_spacing', 'mls', 'MLS')
        if FreqExists & MLSExists:
            raise ValueError('Cannot have both mean level spacing and frequencies')
        elif MLSExists:
            self.Freq = 1.0 / value

        # Mean Neutron Widths:
        self.Gnm, GnmExists = ParamsIn('mean_neutron_width', 'Gnm')

        # Neutron Channel Degrees of Freedom:
        value, nDOFExists = ParamsIn('nDOF', 'nDF')
        if nDOFExists:
            self.nDOF = value
        elif GnmExists:
            self.nDOF = [1] * self.Gnm.shape[1] # Assume that there is one DOF

        # Mean Gamma Widths:
        self.Ggm, GgmExists = ParamsIn('mean_gamma_width', 'Ggm')

        # Gamma Channel Degrees of Freedom:
        value, gDOFExists = ParamsIn('gDOF', 'gDF')
        if gDOFExists:
            self.gDOF = value
        elif GgmExists:
            self.gDOF = [self.DEFAULT_GDOF] * self.Ggm.shape[1] # Arbitrarily high DOF
        
        # Brody Parameter:
        self.w, wExists = ParamsIn('w', 'brody', 'Brody', 'brody_parameter')

    @property
    def n(self):        return self.Freq.shape[1]
    @property
    def L(self):        return np.array(self.sg.L).reshape(1,-1)
    @property
    def J(self):        return np.array(self.sg.J).reshape(1,-1)
    @property
    def S(self):        return np.array(self.sg.S).reshape(1,-1)
    @property
    def MLS(self):      return 1.0 / self.Freq
    @property
    def FreqAll(self):  return np.append(self.Freq, np.array(self.FreqF, ndmin=2), axis=1)

    @classmethod
    def readJSON(cls, file:str):
        """
        Creates a MeanParameters object by importing 
        """

        import json
        param_dict = json.loads(file)
        return cls(**param_dict)
            
    def sample(self, ensemble:str='NNE'):
        """
        Samples resonance parameters based on the given information.
        ...
        """

        # Energy Sampling:
        n = self.Freq.shape[1]
        if self.w is None:
            Et = [RMatrix.SampleEnergies(self.EB, self.Freq[0,g], w=None, ensemble=ensemble) for g in range(n)]
        else:
            Et = [RMatrix.SampleEnergies(self.EB, self.Freq[0,g], w=self.w[0,g], ensemble=ensemble) for g in range(n)]
        if self.FreqF != 0.0:
            Et += [RMatrix.SampleEnergies(self.EB, self.FreqF, w=None, ensemble='Poisson')]
        E = np.array([e for Etg in Et for e in Etg])

        # Width Sampling:
        Gn = np.array([ele for g in range(n) for ele in RMatrix.SampleNeutronWidth(Et[g], self.Gnm[0,g], self.nDOF[0,g], self.L[0,g], self.A, self.ac)])
        Gg = np.array([ele for g in range(n) for ele in RMatrix.SampleGammaWidth(len(Et[g]), self.Ggm[0,g], self.gDOF[0,g])])
        if self.FreqF != 0.0:
            # raise NotImplementedError('The width sampling for false resonances has not been worked out yet.')
            N_F = len(Et[-1])

            cumprobs = np.cumsum(self.Freq[0,:]) / np.sum(self.Freq[0,:])
            R = np.random.rand(N_F,1)
            F_idx = np.sum(cumprobs <= R, axis=1)

            GnF = np.zeros((N_F,1))
            GgF = np.zeros((N_F,1))
            for i in range(N_F):
                GnF[i,0] = RMatrix.SampleNeutronWidth(Et[-1][i:i+1], self.Gnm[0,F_idx[i]], self.nDOF[0,F_idx[i]], self.L[0,F_idx[i]], self.A, self.ac)
                GgF[i,0] = RMatrix.SampleGammaWidth(len(Et[-1][i:i+1]), self.Ggm[0,F_idx[i]], self.gDOF[0,F_idx[i]])

            Gn = np.concatenate((Gn,GnF), axis=0)
            Gg = np.concatenate((Gg,GgF), axis=0)
            # Gn = np.concatenate((Gn,np.ones((len(Et[-1]),1))), axis=0)
            # Gg = np.concatenate((Gg,np.ones((len(Et[-1]),1))), axis=0)

        # # Sorting Indices:
        # idx = np.argsort(E)
        # E  = E[idx]
        # Gn = Gn[idx]
        # Gg = Gg[idx]

        # Spin-group indices:
        if self.FreqF != 0.0:
            spingroups = np.array([g for g in range(n+1) for e in Et[g]])[idx]
        else:
            spingroups = np.array([g for g in range(n) for e in Et[g]])[idx]

        # Missing Resonances:
        # missed_idx = np.random.rand(*E.shape) < self.MissFrac

        # =================================================
        def frac(e, g):
            f = np.concatenate((self.Freq[0,:], np.array([0.0])), axis=0)
            return ((0.0 + 0.0000022*e**2)/29) * f[g]/np.sum(f)
        missed_idx = np.random.rand(*E.shape) < frac(E, spingroups)
        # missed_idx = np.random.rand(*E.shape) < 0.0
        # =================================================

        # Returning Resonance Data:
        resonances = Resonances(E=E, Gn=Gn, Gg=Gg)
        return resonances[~missed_idx], spingroups[~missed_idx], resonances[missed_idx], spingroups[missed_idx]

# =================================================================================================
# Resonances:
# =================================================================================================
class Resonances:
    """
    ...
    """

    def __init__(self, E, *args, **kwargs):
        self.properties = ['E']

        Idx = np.argsort(E)
        self.E = np.array(E).reshape(-1)[Idx]

        def ParamsIn(name:str, num:int):
            if name in kwargs.keys():   var = kwargs['Gn']
            elif len(args) >= num+1:    var = args[num]
            else:                       return None
            self.properties.append(name)
            return np.array(var).reshape(-1)[Idx]

        self.Gn  = ParamsIn('Gn' , 1)
        self.Gg  = ParamsIn('Gg' , 2)
        self.GfA = ParamsIn('GfA', 3)
        self.GfB = ParamsIn('GfB', 4)
        self.SG  = ParamsIn('SG' , 5)

    # Get resonances by indexing the "Resonances" object:
    def __getitem__(self, Idx):
        kwargs = {}
        # print('-=============================')
        # print(Idx)
        for property in self.properties:
            if   property == 'E'   :    kwargs['E']   = self.E[Idx]
            elif property == 'Gn'  :    kwargs['Gn']  = self.Gn[Idx]
            elif property == 'Gg'  :    kwargs['Gg']  = self.Gg[Idx]
            elif property == 'GfA' :    kwargs['GfA'] = self.GfA[Idx]
            elif property == 'GfB' :    kwargs['GfB'] = self.GfB[Idx]
            elif property == 'SG'  :    kwargs['SG']  = self.SG[Idx]
        return Resonances(**kwargs)

    # Print the resonance data as a table:
    def __str__(self):
        Data = []
        if 'E'   in self.properties :   Data.append(self.E)
        if 'Gn'  in self.properties :   Data.append(self.Gn)
        if 'Gg'  in self.properties :   Data.append(self.Gg)
        if 'GfA' in self.properties :   Data.append(self.GfA)
        if 'GfB' in self.properties :   Data.append(self.GfB)
        if 'SG'  in self.properties :   Data.append(self.SG)
        Data = np.array(Data).reshape(len(self.properties), self.len).T
        table_str = '\n'.join(str(pd.DataFrame(data=Data, columns=self.properties)).split('\n')[:-2])
        return table_str

    @property
    def len(self):
        return self.E.size

    def AddMissing(self, Prior, frac, mult, EB=None):
        """
        ...
        """
        
        if EB is None:
            EB = (min(self.E), max(self.E))
        num_miss = round(self.len * mult)
        E_miss = np.linspace(*EB, num_miss)
        
        Prior_miss = np.zeros((len(E_miss), Prior.shape[1]), dtype='f8')
        for g in range(Prior.shape[1]-1):
            Prior_miss[:,g] = frac(E_miss, g) / mult
        Prior_miss[:,-1] = 1.0 - np.sum(Prior_miss, axis=1)
        
        E_comb = np.concatenate((self.E, E_miss))
        idx = np.argsort(E_comb)
        E_comb = E_comb[idx]
        Prior_comb = np.concatenate((Prior, Prior_miss))[idx,:]
        
        return E_comb, Prior_comb
        
