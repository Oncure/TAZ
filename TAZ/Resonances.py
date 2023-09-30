import numpy as np
import pandas as pd

from . import halfint, RMatrix

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
                    param_value = np.array(kwargs[param]).reshape(1,self.num_sgs)
                elif param_type == 'float':
                    param_value = float(kwargs[param])
                elif param_type == 'int':
                    param_value = int(kwargs[param])
                elif param_type == 'halfint':
                    param_value = halfint(kwargs[param])
                elif param_type == 'tuple':
                    param_value = tuple(kwargs[param])
                elif param_type == 'pass':
                    param_value = kwargs[param]
                return param_value, True
            else:
                return None, False
            
        # Spin-Groups:
        self.sg, sgExists = paramsIn('sg', 'SG', 'spingroups', type='pass')
        if not sgExists:
            raise ValueError('The spingroups are a required argument for MeanParameters.')
        self.num_sgs = self.sg.num_sgs


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
        else:           self.mass = None
        # Atomic Radius:
        ac, acExists = paramsIn('Ac', 'ac', 'atomic_radius', 'scatter_radius', type='float')
        if acExists:    self.ac = ac
        elif AExists:   self.ac = RMatrix.NuclearRadius(self.A)
        else:           self.ac = None
        # Energy Range:
        EB, EBExists = paramsIn('EB', 'energy_bounds', 'energy_range', type='tuple')
        if EBExists:
            self.EB = EB
            if len(self.EB) != 2:           raise ValueError('"EB" can only have two values for an interval')
            elif self.EB[0] > self.EB[1]:   raise ValueError('"EB" must be a valid increasing interval')
        else:
            self.EB = None
        # False Frequency:
        FreqF, FalseFreqExists = paramsIn('freqF', 'FreqF', 'false_frequency', type='float')
        self.FreqF = FreqF if FalseFreqExists else 0.0
        
        # Frequencies:
        self.Freq, FreqExists = paramsIn('freq', 'Freq', 'frequency', 'Frequency')
        
        # Mean Level Spacings:
        MLS, MLSExists = paramsIn('mean_spacing', 'mean_level_spacing', 'mls', 'MLS')
        if FreqExists & MLSExists:
            raise ValueError('Cannot have both mean level spacing and frequencies')
        elif MLSExists:
            self.Freq = 1.0 / MLS

        # Brody Parameter:
        w, wExists = paramsIn('w', 'brody', 'Brody', 'brody_parameter')
        if wExists:
            if w.size == 1:     self.w = w * np.ones((1,self.num_sgs))
            else:               self.w = w
        else:
            self.w = np.ones((1,self.num_sgs))

        # Mean Neutron Widths:
        self.Gnm, GnmExists = paramsIn('mean_neutron_width', 'Gnm')

        # Neutron Channel Degrees of Freedom:
        nDOF, nDOFExists = paramsIn('nDOF', 'nDF')
        if nDOFExists:      self.nDOF = nDOF
        else:               self.nDOF = np.ones((1,self.num_sgs)) # Assume that there is one DOF

        # Mean Gamma Widths:
        self.Ggm, GgmExists = paramsIn('mean_gamma_width', 'Ggm')

        # Gamma Channel Degrees of Freedom:
        gDOF, gDOFExists = paramsIn('gDOF', 'gDF')
        if gDOFExists:      self.gDOF = gDOF
        else:               self.gDOF = self.DEFAULT_GDOF * np.ones((1,self.num_sgs)) # Arbitrarily high DOF

        # Truncation Width:
        Gn_trunc, truncExists = paramsIn('trunc', 'Gn_trunc', 'truncation_width')
        if truncExists:
            if Gn_trunc.size == 1:      self.Gn_trunc = Gn_trunc * np.ones((1,self.num_sgs))
            else:                       self.Gn_trunc = Gn_trunc # FIXME: MAKE THIS AS A FUNCTION OF ENERGY!
        else:
            self.Gn_trunc = np.zeros((1,self.num_sgs))

        # Missing Fraction:
        MissFrac, MissFracExists = paramsIn('MissFrac', 'miss_frac', 'pM')
        if MissFracExists:
            if truncExists:
                raise ValueError('One cannot specify "MissFrac" and "Gn_trunc" at the same time.')
            self.MissFrac = MissFrac
            self.given_miss_frac = True
        else:
            self.MissFrac = RMatrix.FractionMissing(self.Gn_trunc)
            self.given_miss_frac = False

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
    
    def __repr__(self):
        # raise NotImplementedError('String representation of "MeanParameters" has not been implemented yet.')
        txt = ''
        txt += f'Nuclear Spin        = {self.I}\n'
        txt += f'Atomic Number       = {self.Z}\n'
        txt += f'Atomic Mass Number  = {self.A}\n'
        txt += f'Atomic Mass         = {self.mass:.5f} (amu)\n'
        txt += f'Channel Radius      = {self.ac:.7f} (fm)\n'
        txt += f'Energy Bounds       = {self.EB[0]:.3e} < E < {self.EB[1]:.3e} (eV)\n'
        txt += f'False Level Density = {self.FreqF:.7f} (1/eV)\n'
        txt += '\n'
        properties = ['Level Densities', \
                      'Brody Parameters', \
                      'Mean Neutron Width', \
                      'Neutron Width DOF', \
                      'Mean Gamma Width', \
                      'Gamma Width DOF', \
                      'Truncation N Width', \
                      'Missing Fraction']
        data = np.concatenate((self.Freq, self.w, self.Gnm, self.nDOF, self.Ggm, self.gDOF, self.Gn_trunc, self.MissFrac), axis=0)
        txt += str(pd.DataFrame(data=data, index=properties, columns=self.sg.SGs))
        return txt
    def __str__(self):
        return self.__repr__()

    @classmethod
    def readJSON(cls, file:str):
        """
        Creates a MeanParameters object by importing a JSON file.
        """

        import json
        param_dict = json.loads(file)
        return cls(**param_dict)
            
    def sample(self, ensemble:str='NNE', rng=None, seed:int=None):
        """
        Samples resonance parameters based on the given information.
        
        ...
        """

        if rng is None:
            rng = np.random.default_rng(seed)

        # Energy Sampling:
        n = self.Freq.shape[1]
        E          = np.zeros((0,))
        Gn         = np.zeros((0,))
        Gg         = np.zeros((0,))
        spingroups = np.zeros((0,), dtype=int)
        for g in range(n):
            # Energy sampling:
            w = self.w[0,g] if self.w is not None else None
            E_group  = RMatrix.SampleEnergies(self.EB, self.Freq[0,g], w=w, ensemble=ensemble, rng=rng)
            
            # Width sampling:
            num_group = len(E_group)
            Gn_group = RMatrix.SampleNeutronWidth(E_group, self.Gnm[0,g], self.nDOF[0,g], self.L[0,g], self.A, self.ac, rng=rng)
            Gg_group = RMatrix.SampleGammaWidth(num_group, self.Ggm[0,g], self.gDOF[0,g], rng=rng)
            
            # Append to group:
            E          = np.concatenate((E         , E_group ))
            Gn         = np.concatenate((Gn        , Gn_group))
            Gg         = np.concatenate((Gg        , Gg_group))
            spingroups = np.concatenate((spingroups, g*np.ones((num_group,), dtype=int)))

        # False Resonances:
        if self.FreqF != 0.0:
            # Energy sampling:
            E_false = RMatrix.SampleEnergies(self.EB, self.FreqF, w=None, ensemble='Poisson')
            
            # False width sampling:
            # False widths are sampled by taking the frequency-weighted average of each spingroup's width distributions.
            num_false = len(E_false)
            Gn_false_group = np.zeros((num_false,n))
            Gg_false_group = np.zeros((num_false,n))
            for g in range(n):
                Gn_false_group[:,g] = RMatrix.SampleNeutronWidth(E_false, self.Gnm[0,g], self.nDOF[0,g], self.L[0,g], self.A, self.ac, rng=rng)
                Gg_false_group[:,g] = RMatrix.SampleGammaWidth(num_false, self.Ggm[0,g], self.gDOF[0,g], rng=rng)
            cumprobs = np.cumsum(self.Freq[0,:]) / np.sum(self.Freq[0,:])
            R = rng.uniform(size=(num_false,1))
            idx = np.arange(num_false)
            group_idx = np.sum(cumprobs <= R, axis=1)
            Gn_false = Gn_false_group[idx,group_idx]
            Gg_false = Gg_false_group[idx,group_idx]

            # Append to group:
            E          = np.concatenate((E         , E_false ))
            Gn         = np.concatenate((Gn        , Gn_false))
            Gg         = np.concatenate((Gg        , Gg_false))
            spingroups = np.concatenate((spingroups, n*np.ones((num_false,), dtype=int)))

        # Sorting Indices:
        idx = np.argsort(E)
        E  = E[idx]
        Gn = Gn[idx]
        Gg = Gg[idx]
        spingroups = spingroups[idx]

        # Missing Resonances:
        if self.MissFrac is not None:
            if self.given_miss_frac:
                miss_frac = np.concatenate((self.MissFrac, np.zeros((1,1))), axis=1)
                missed_idx = (rng.uniform(size=E.shape) < miss_frac[0,spingroups])

            else: # given Gn_trunc
                Gn_trunc = np.concatenate((self.Gn_trunc, np.zeros((1,1))), axis=1)
                rGn = np.zeros(Gn.shape)
                for g in range(n):
                    spingroup_g = (spingroups == g)
                    rGn[spingroup_g] = Gn[spingroup_g] * RMatrix.ReduceFactor(E[spingroup_g], self.L[0,g], self.A, self.ac)
                missed_idx = (rGn <= Gn_trunc[0,spingroups])

        # Caught resonances:
        E_caught  =  E[~missed_idx]
        Gn_caught = Gn[~missed_idx]
        Gg_caught = Gg[~missed_idx]
        resonances_caught = Resonances(E=E_caught, Gn=Gn_caught, Gg=Gg_caught)
        spingroups_caught = spingroups[~missed_idx]
        
        # Missing resonances:
        E_missed  =  E[missed_idx]
        Gn_missed = Gn[missed_idx]
        Gg_missed = Gg[missed_idx]
        resonances_missed = Resonances(E=E_missed, Gn=Gn_missed, Gg=Gg_missed)
        spingroups_missed = spingroups[missed_idx]

        # =================================================
        # def frac(e, g):
        #     f = np.concatenate((self.Freq[0,:], np.array([0.0])), axis=0)
        #     return ((0.0 + 0.0000022*e**2)/29) * f[g]/np.sum(f)
        # missed_idx = np.random.rand(*E.shape) < frac(E, spingroups)
        # missed_idx = np.random.rand(*E.shape) < 0.0
        # =================================================

        # Returning resonance data:
        return resonances_caught, spingroups_caught, resonances_missed, spingroups_missed
    
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
        
    def fit(self, quantity:str, spingroup:int, cdf:bool=False):
        """
        ...
        """

        if   quantity == 'energies':
            if not cdf: # PDF
                f = lambda e: 1.0 / (self.EB[1] - self.EB[0])
            else: # CDF
                f = lambda e: (e - self.EB[0]) / (self.EB[1] - self.EB[0])
        elif quantity == 'level spacing':
            if np.all(self.w == 1.0):
                if self.MissFrac == 0.0:
                    dist_type = 'Wigner'
                else:
                    dist_type = 'Missing'
            else:
                if self.MissFrac == 0.0:
                    dist_type = 'Brody'
                else:
                    raise NotImplementedError('The level-spacing distribution for Brody distribution with missing levels has not been implemented yet.')
            if not cdf: # PDF
                f = self.distributions(dist_type)[spingroup].f0
            else: # CDF
                f = lambda x: 1.0 - self.distributions(dist_type)[spingroup].f1(x)
        elif quantity == 'neutron width':
            if not cdf: # PDF
                f = lambda rGn: RMatrix.PorterThomasPDF(rGn, self.Gnm[spingroup], trunc=self.Gn_trunc[spingroup], dof=self.nDOF[spingroup])
            else: # CDF
                f = lambda rGn: RMatrix.PorterThomasCDF(rGn,  self.Gnm[spingroup], trunc=self.Gn_trunc[spingroup], dof=self.nDOF[spingroup])
        elif quantity in ('gamma width', 'capture width'):
            if not cdf: # PDF
                f = lambda rGg: RMatrix.PorterThomasPDF(rGg, self.Ggm[spingroup], trunc=0.0, dof=self.gDOF[spingroup])
            else: # CDF
                f = lambda rGg: RMatrix.PorterThomasCDF(rGg,  self.Ggm[spingroup], trunc=0.0, dof=self.gDOF[spingroup])

        return f

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