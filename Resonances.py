import numpy as np
import pandas as pd

import SpinGroups
import RMatrix

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

    Attributes:
    ----------
    I        :: int
        target particle spin.
    Z        :: int
        target atomic number.
    A        :: int
        target atomic mass.
    mass     :: float
        target nuclei mass.
    ac       :: float
        target nuclear radius.
    EB       :: float [2]
        Energy range for evaluation.
    FreqF    :: float
        False resonance level density.
    Gn_trunc :: float
        Lowest recordable neutron width.
    MissFrac :: float
        Fraction of Resonances that have been missed.
    sg       :: SpinGroups
        Spingroups for the reaction.
    Freq     :: float [G]
        Resonance level densities for each spingroup.
    FreqAll  :: float [G+1]
        Resonance and false level densities.
    MLS      :: float [G]
        Resonance mean level spacings for each spingroup.
    Gnm      :: float [G]
        Resonance mean neutron widths for each spingroup.
    nDOF     :: float [G]
        Resonance neutron width degrees of freedom for each spingroup.
    Ggm      :: float [G]
        Resonance mean gamma width for each spingroup.
    gDOF     :: float [G]
        Resonance gamma width degrees of freedom for each spingroup.
    w        :: float [G]
        Brody resonance parameter.
    """

    DEFAULT_GDOF = 500

    def __init__(self, **kwargs):
        """
        Initializes reaction parameters with keyword arguments.
        """

        param_names = set(kwargs.keys())
        def paramsIn(*names:str, **options:str):
            if 'type' in options.keys():    param_type = options['type']
            else:                           param_type = 'array1'
            param_set = (param_names & set(names))
            param_list = list(param_set)
            if len(param_list) >= 2:
                param_str = '\"' + '\", \"'.join(param_list[:-1]) + f', and \"{param_list[-1]}\"'
                raise ValueError(f'Cannot accept multiple parameters, {param_str}.')
            elif len(param_list) == 1:
                param = param_list[0]
                if param_type == 'array1':
                    param_value = np.array(kwargs[param]).reshape(1,-1)
                elif param_type == 'float':
                    param_value = float(kwargs[param])
                elif param_type == 'int':
                    param_value = int(kwargs[param])
                elif param_type == 'halfint':
                    param_value = SpinGroups.halfint(kwargs[param])
                elif param_type == 'tuple':
                    param_value = tuple(kwargs[param])
                elif param_type == 'pass':
                    param_value = kwargs[param]
                return param_value, True
            else:
                return None, False

        # Isotope Spin:
        self.I, IExists = paramsIn('I', 'S', 'spin', 'isotope_spin', type='halfint')

        # Atomic Number:
        self.Z, ZExists = paramsIn('Z', 'atomic_number', type='int')

        # Atomic Mass:
        self.A, AExists = paramsIn('A', 'atomic_mass', type='int')

        # Mass:
        mass, massExists = paramsIn('mass', 'Mass', 'm', 'M', type='float')
        if massExists:  self.mass = mass
        elif AExists:   self.mass = float(self.A)

        # Atomic Radius:
        ac, acExists = paramsIn('Ac', 'ac', 'atomic_radius', 'scatter_radius', type='float')
        if acExists:    self.ac = ac
        elif AExists:   self.ac = RMatrix.NuclearRadius(self.A)

        # Energy Range:
        EB, EBExists = paramsIn('EB', 'energy_bounds', 'energy_range', type='tuple')
        if EBExists:
            self.EB = EB
            if len(self.EB) != 2:
                raise ValueError('"EB" can only have two values for an interval')
            elif self.EB[0] > self.EB[1]:
                raise ValueError('"EB" must be a valid increasing interval')

        # False Frequency:
        FreqF, FalseFreqExists = paramsIn('freqF', 'FreqF', 'false_frequency', type='float')
        self.FreqF = FreqF if FalseFreqExists else 0.0

        # Spin-Groups:
        self.sg, sgExists = paramsIn('sg', 'SG', 'spin-group', type='pass')

        # Frequencies:
        self.Freq, FreqExists = paramsIn('freq', 'Freq', 'frequency', 'Frequency')

        # Mean Level Spacings:
        MLS, MLSExists = paramsIn('mean_spacing', 'mean_level_spacing', 'mls', 'MLS')
        if FreqExists & MLSExists:
            raise ValueError('Cannot have both mean level spacing and frequencies')
        elif MLSExists:
            self.Freq = 1.0 / MLS

        # Truncation Width:
        num_sgs = self.Freq.size
        Gn_trunc, truncExists = paramsIn('trunc', 'Gn_trunc', 'truncation_width')
        if truncExists:
            if Gn_trunc.size == 1:
                self.Gn_trunc = Gn_trunc * np.ones((1,num_sgs))
            else:
                self.Gn_trunc = Gn_trunc # FIXME: MAKE THIS AS A FUNCTION OF ENERGY!
        else:
            self.Gn_trunc = np.zeros((1,num_sgs))

        # Missing Fraction:
        MissFrac, MissFracExists = paramsIn('MissFrac', 'miss_frac', 'pM')
        if MissFracExists:
            self.MissFrac = MissFrac
        else:
            self.MissFrac = RMatrix.FractionMissing(self.Gn_trunc)

        # Mean Neutron Widths:
        self.Gnm, GnmExists = paramsIn('mean_neutron_width', 'Gnm')

        # Neutron Channel Degrees of Freedom:
        nDOF, nDOFExists = paramsIn('nDOF', 'nDF')
        if nDOFExists:
            self.nDOF = nDOF
        elif GnmExists:
            self.nDOF = np.ones((self.Gnm.shape[1],)) # Assume that there is one DOF

        # Mean Gamma Widths:
        self.Ggm, GgmExists = paramsIn('mean_gamma_width', 'Ggm')

        # Gamma Channel Degrees of Freedom:
        gDOF, gDOFExists = paramsIn('gDOF', 'gDF')
        if gDOFExists:
            self.gDOF = gDOF
        elif GgmExists:
            self.gDOF = self.DEFAULT_GDOF * np.ones((self.Ggm.shape[1],)) # Arbitrarily high DOF
        
        # Brody Parameter:
        self.w, wExists = paramsIn('w', 'brody', 'Brody', 'brody_parameter')

    @property
    def n(self):
        'Number of Spingroups'
        return self.Freq.shape[1]
    @property
    def L(self):
        'Angular Quantum Number'
        return np.array(self.sg.L).reshape(1,-1)
    @property
    def J(self):
        'Total Spin Quantum Number'
        return np.array(self.sg.J).reshape(1,-1)
    @property
    def S(self):
        '...'
        return np.array(self.sg.S).reshape(1,-1)
    @property
    def MLS(self):
        'Mean Level Spacing'
        return 1.0 / self.Freq
    @property
    def FreqAll(self):
        'Frequencies, including False Frequency'
        return np.append(self.Freq, np.array(self.FreqF, ndmin=2), axis=1)
    
    def __str__(self):
        raise NotImplementedError('String representation of "MeanParameters" has not been implemented yet.')

    @classmethod
    def readJSON(cls, file:str):
        """
        Creates a MeanParameters object by importing a JSON file.
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
        Et = []
        for g in range(n):
            w = self.w[0,g] if self.w is not None else None
            et = RMatrix.SampleEnergies(self.EB, self.Freq[0,g], w=w, ensemble=ensemble)
            # Append to group:
            Et.append(et)
        if self.FreqF != 0.0:
            Et += [RMatrix.SampleEnergies(self.EB, self.FreqF, w=None, ensemble='Poisson')]
        
        E = np.array([e for Etg in Et for e in Etg])

        # Width Sampling:
        Gn = np.array([ele for g in range(n) \
                       for ele in RMatrix.SampleNeutronWidth(Et[g], self.Gnm[0,g], self.nDOF[0,g], self.L[0,g], self.A, self.ac)])
        Gg = np.array([ele for g in range(n) \
                       for ele in RMatrix.SampleGammaWidth(len(Et[g]), self.Ggm[0,g], self.gDOF[0,g])])
        # False Width sampling:
        # False widths are sampled by taking the frequency-weighted average of each spingroup's width distributions.
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
        idx = np.argsort(E)
        E  = E[idx]
        Gn = Gn[idx]
        Gg = Gg[idx]

        # Spin-group indices:
        if self.FreqF != 0.0:
            spingroups = np.array([g for g in range(n+1) for e in Et[g]], dtype=int)[idx]
        else:
            spingroups = np.array([g for g in range(n) for e in Et[g]], dtype=int)[idx]

        # Missing Resonances:
        if self.MissFrac is not None:
            miss_frac = np.concatenate((self.MissFrac, np.zeros((1,1))), axis=1)
            missed_idx = (np.random.rand(*E.shape) < miss_frac[0,spingroups])

        # =================================================
        # def frac(e, g):
        #     f = np.concatenate((self.Freq[0,:], np.array([0.0])), axis=0)
        #     return ((0.0 + 0.0000022*e**2)/29) * f[g]/np.sum(f)
        # missed_idx = np.random.rand(*E.shape) < frac(E, spingroups)
        # missed_idx = np.random.rand(*E.shape) < 0.0
        # =================================================

        # Returning Resonance Data:
        resonances = Resonances(E=E, Gn=Gn, Gg=Gg)
        return resonances[~missed_idx], spingroups[~missed_idx], resonances[missed_idx], spingroups[missed_idx]
    
    def distributions(self, dist_type:str='Wigner', err:float=5e-3):
        """
        ...
        """

        if   dist_type == 'Wigner':
            return RMatrix.Distributions.wigner(self.Freq)
        elif dist_type == 'Brody':
            return RMatrix.Distributions.brody(self.Freq, self.w)
        elif dist_type == 'Missing':
            return RMatrix.Distributions.missing(self.Freq, self.MissFrac, err)
        else:
            raise NotImplementedError(f'The distribution type, "{dist_type}", has not been implemented yet.')

# =================================================================================================
# Resonances:
# =================================================================================================
class Resonances:
    """
    ...
    """

    def __init__(self, E, Gn=None, Gg=None, GfA=None, GfB=None, SG=None):
        self.properties = ['E']

        indices = np.argsort(E)
        self.E = np.array(E).reshape(-1)[indices]
        if Gn  is not None:
            self.properties.append('Gn')
            self.Gn  = np.array(Gn ).reshape(-1)[indices]
        if Gg  is not None:
            self.properties.append('Gg')
            self.Gg  = np.array(Gg ).reshape(-1)[indices]
        if GfA is not None:
            self.properties.append('GfA')
            self.GfA = np.array(GfA).reshape(-1)[indices]
        if GfB is not None:
            self.properties.append('GfB')
            self.GfB = np.array(GfB).reshape(-1)[indices]
        if SG  is not None:
            self.properties.append('SG')
            self.SG  = np.array(SG ).reshape(-1)[indices]

    # Get resonances by indexing the "Resonances" object:
    def __getitem__(self, indices):
        kwargs = {}
        if 'E'   in self.properties :    kwargs['E']   = self.E[indices]
        if 'Gn'  in self.properties :    kwargs['Gn']  = self.Gn[indices]
        if 'Gg'  in self.properties :    kwargs['Gg']  = self.Gg[indices]
        if 'GfA' in self.properties :    kwargs['GfA'] = self.GfA[indices]
        if 'GfB' in self.properties :    kwargs['GfB'] = self.GfB[indices]
        if 'SG'  in self.properties :    kwargs['SG']  = self.SG[indices]
        return Resonances(**kwargs)

    # Print the resonance data as a table:
    def __str__(self):
        data = []
        if 'E'   in self.properties :   data.append(self.E)
        if 'Gn'  in self.properties :   data.append(self.Gn)
        if 'Gg'  in self.properties :   data.append(self.Gg)
        if 'GfA' in self.properties :   data.append(self.GfA)
        if 'GfB' in self.properties :   data.append(self.GfB)
        if 'SG'  in self.properties :   data.append(self.SG)
        data = np.array(data).reshape(len(self.properties), self.len).T
        table_str = '\n'.join(str(pd.DataFrame(data=data, columns=self.properties)).split('\n')[:-2])
        return table_str

    @property
    def len(self):
        return self.E.size

    def AddMissing(self, prior, frac, mult, EB=None):
        """
        ...
        """
        
        if EB is None:
            EB = (min(self.E), max(self.E))
        num_miss = round(self.len * mult)
        E_miss = np.linspace(*EB, num_miss)
        
        prior_miss = np.zeros((len(E_miss), prior.shape[1]), dtype='f8')
        for g in range(prior.shape[1]-1):
            prior_miss[:,g] = frac(E_miss, g) / mult
        prior_miss[:,-1] = 1.0 - np.sum(prior_miss, axis=1)
        
        E_combined = np.concatenate((self.E, E_miss))
        idx = np.argsort(E_combined)
        E_combined = E_combined[idx]
        prior_combined = np.concatenate((prior, prior_miss))[idx,:]
        
        return E_combined, prior_combined