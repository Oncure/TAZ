import numpy as np
import pandas as pd

from . import halfint, RMatrix, SpinGroup, SpinGroups

__doc__ = """
This file keeps the "MeanParameters" class and "Resonances" class. The "MeanParameters" class
contains all relevent information that is independent of individual resonances, such as isotopic
mass, isotope spin, mean level-spacing, etc. The "Resonances" class contains resonance-specific
information such as Energy, widths, and spin-group assignments.
"""

# =================================================================================================
#    Particle:
# =================================================================================================

class Particle:
    """
    Attributes:
    ----------
    Z        :: int
        Atomic number
    A        :: int
        Atomic mass
    I        :: halfint
        Particle spin
    mass     :: float
        Nuclei mass
    ac       :: float
        Nuclear radius
    """

    mass_neutron = 1.008665 # amu

    def __init__(self, Z:int=None, A:int=None, I:halfint=None,
                 mass:float=None, AWRI:float=None,
                 radius:float=None, name:str=None):
        """
        Initialize a Particle object.
        """
        # Atomic Number:
        if Z is not None:   self.Z = int(Z)
        else:               self.Z = None
        # Atomic Mass:
        if A is not None:   self.A = int(A)
        else:               self.A = None
        # Isotope Spin:
        if I is not None:   self.I = halfint(I)
        else:               self.I = None
        # Mass: (amu)
        if mass is not None:    self.mass = float(mass)
        elif AWRI is not None:  self.mass = float(AWRI*self.mass_neutron)
        elif A is not None:     self.mass = float(A)
        else:                   self.mass = None
        # Nuclear Radius: (fm)
        if radius is not None:  self.radius = float(radius)
        elif A is not None:     self.radius = 1.23 * self.A**(1/3)
        # Particle Name:
        if name is not None:
            self.name = str(name)
        elif (A is not None) and (Z is not None):
            self.name = str(Z*1000+A)
        else:
            self.name = '???'

        def __repr__(self):
            txt  = f'Particle:       {self.name}\n'
            txt += f'Atomic Number:  {self.Z}\n'
            txt += f'Atomic Mass:    {self.A}\n'
            txt += f'Nuclear Spin:   {self.I}\n'
            txt += f'Mass:           {self.mass} amu\n'
            txt += f'Nuclear Radius: {self.radius} fm\n'
            return txt
        def __str__(self):
            return self.name
    
Neutron = Particle(I=0.5, Z=0, A=1, mass=1.008665, radius=0.8, name='neutron')

# =================================================================================================
#    Mean Parameters:
# =================================================================================================

class MeanParameters:
    """
    MeanParameters is a class that contains information about a particular reaction, such as the
    target nuclei mass, nuclear radius, spin, mean level spacing, and more.

    Let "G" be the number of spingroups.

    Target Isotope Attributes:
    -------------------------
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
    
    Resonance Attributes:
    --------------------
    sg       :: SpinGroups
        Spingroups for the reaction.
    EB       :: float [2]
        Energy range for evaluation.
    FreqF    :: float
        False resonance level density.
    Freq     :: float [G]
        Resonance level densities for each spingroup.
    MLS      :: float [G]
        Resonance mean level spacings for each spingroup.
    w        :: float [G]
        Brody resonance parameter.
    Gnm      :: float [G]
        Resonance mean neutron widths for each spingroup.
    nDOF     :: float [G]
        Resonance neutron width degrees of freedom for each spingroup.
    Ggm      :: float [G]
        Resonance mean gamma width for each spingroup.
    gDOF     :: float [G]
        Resonance gamma width degrees of freedom for each spingroup.
    Gn_trunc :: float [G]
        Lowest recordable neutron width.
    MissFrac :: float [G]
        Fraction of Resonances that have been missed.
    """

    DEFAULT_GDOF = 500

    def __init__(self,
                 I:halfint=None, Z:int=None, A:int=None,
                 mass:float=None, ac:float=None,
                 EB:tuple=None, FreqF:float=None,
                 sg:SpinGroups=None,
                 Freq:list=None, MLS:list=None,
                 w:list=None,
                 Gnm:list=None, nDOF:list=None,
                 Ggm:list=None, gDOF:list=None,
                 Gn_trunc:list=None, MissFrac:list=None):
        """
        Initializes reaction parameters with keyword arguments.
        """
            
        # Spin-Groups:
        if sg is None:
            raise ValueError('The spingroups are a required argument for MeanParameters.')
        elif type(sg) != SpinGroups:
            raise TypeError('"sg" must have type "SpinGroups".')
        self.sg = sg
        self.num_sgs = self.sg.num_sgs

        # Isotope Spin:
        if I is not None:   self.I = halfint(I)
        else:               self.I = None
        
        # Atomic Number:
        if Z is not None:   self.Z = int(Z)
        else:               self.Z = None
        
        # Atomic Mass:
        if A is not None:   self.A = int(A)
        else:               self.A = None
        
        # Mass:
        if mass is not None:    self.mass = float(mass)
        elif A is not None:     self.mass = float(A)
        else:                   self.mass = None
        
        # Atomic Radius:
        if ac is not None:      self.ac = float(ac)
        elif A is not None:     self.ac = RMatrix.NuclearRadius(self.A)
        else:                   self.ac = None
        
        # Energy Range:
        if EB is not None:
            if len(EB) != 2:        raise ValueError('"EB" can only have two values for an interval.')
            elif EB[0] > EB[1]:     raise ValueError('"EB" must be a valid increasing interval.')
            self.EB = (float(EB[0]), float(EB[1]))
        else:
            self.EB = None
        
        # False Frequency:
        if FreqF is not None:   self.FreqF = float(FreqF)
        else:                   self.FreqF = 0.0
        
        # Frequencies:
        if Freq is not None and MLS is not None:
            raise ValueError('Cannot have both mean level spacing and frequencies.')
        elif Freq is not None:
            if (not hasattr(Freq, '__iter__')) or (len(Freq) != self.num_sgs):
                raise TypeError(f'"Freq" must be an array with length equal to the number of spingroups, {self.num_sgs}.')
            self.Freq = np.array(Freq, dtype=float).reshape(-1,)
        elif MLS is not None:
            if (not hasattr(MLS, '__iter__')) or (len(MLS) != self.num_sgs):
                raise TypeError(f'"MLS" must be an array with length equal to the number of spingroups, {self.num_sgs}.')
            self.Freq = 1.0 / np.array(MLS, dtype=float).reshape(-1,)
        else:
            self.Freq = None

        # Brody Parameter:
        if w is not None:
            if hasattr(w, '__iter__'):
                if (len(w) != self.num_sgs):
                    raise TypeError(f'"w" must have a length equal to the number of spingroups, {self.num_sgs}.')
                self.w = np.array(w, dtype=float).reshape(-1,)
            else:
                self.w = float(w) * np.ones((self.num_sgs,))
        else:
            self.w = np.ones((self.num_sgs,))

        # Mean Neutron Widths:
        if Gnm is not None:
            if (not hasattr(Gnm, '__iter__')) or (len(Gnm) != self.num_sgs):
                raise TypeError(f'"Gnm" must be an array with length equal to the number of spingroups, {self.num_sgs}.')
            self.Gnm = np.array(Gnm, dtype=float).reshape(-1,)
        else:
            self.Gnm = None

        # Neutron Channel Degrees of Freedom:
        if nDOF is not None:
            if (not hasattr(nDOF, '__iter__')) or (len(nDOF) != self.num_sgs):
                raise TypeError(f'"nDOF" must be an array with length equal to the number of spingroups, {self.num_sgs}.')
            self.nDOF = np.array(nDOF, dtype=int).reshape(-1,)
        else:
            self.nDOF = np.ones((self.num_sgs,), dtype=int)

        # Mean Gamma Widths:
        if Ggm is not None:
            if (not hasattr(Ggm, '__iter__')) or (len(Ggm) != self.num_sgs):
                raise TypeError(f'"Ggm" must be an array with length equal to the number of spingroups, {self.num_sgs}.')
            self.Ggm = np.array(Ggm, dtype=float).reshape(-1,)
        else:
            self.Ggm = None

        # Gamma Channel Degrees of Freedom:
        if gDOF is not None:
            if (not hasattr(gDOF, '__iter__')) or (len(gDOF) != self.num_sgs):
                raise TypeError(f'"gDOF" must be an array with length equal to the number of spingroups, {self.num_sgs}.')
            self.gDOF = np.array(gDOF, dtype=int).reshape(-1,)
        else:
            self.gDOF = self.DEFAULT_GDOF * np.ones((self.num_sgs,), dtype=int)

        # Truncation Width:
        if Gn_trunc is not None:
            if (not hasattr(Gn_trunc, '__iter__')) or (len(Gn_trunc) != self.num_sgs):
                raise TypeError(f'"Gn_trunc" must be an array with length equal to the number of spingroups, {self.num_sgs}.')
            self.Gn_trunc = np.array(Gn_trunc, dtype=float).reshape(-1,) # FIXME: MAKE THIS AS A FUNCTION OF ENERGY!
        else:
            self.Gn_trunc = np.zeros((self.num_sgs,), dtype=float)

        # Missing Fraction:
        if MissFrac is not None:
            if (not hasattr(MissFrac, '__iter__')) or (len(MissFrac) != self.num_sgs):
                raise TypeError(f'"MissFrac" must be an array with length equal to the number of spingroups, {self.num_sgs}.')
            self.MissFrac = np.array(MissFrac, dtype=float).reshape(-1,) # FIXME: MAKE THIS AS A FUNCTION OF ENERGY!
            self.given_miss_frac = True
        elif Gn_trunc is not None:
            self.MissFrac = RMatrix.FractionMissing(self.Gn_trunc, self.Gnm, self.nDOF) # FIXME: MAKE THIS AS A FUNCTION OF ENERGY!
            self.given_miss_frac = False
        else:
            self.MissFrac = np.zeros((self.num_sgs,), dtype=float)
            self.given_miss_frac = False

    @property
    def L(self):
        'Orbital Angular Momentum'
        return np.array(self.sg.L)
    @property
    def J(self):
        'Total Angular Momentum'
        return np.array(self.sg.J)
    @property
    def S(self):
        'Channel Spin'
        return np.array(self.sg.S)
    @property
    def MLS(self):
        'Mean Level Spacing'
        return 1.0 / self.Freq
    @property
    def FreqAll(self):
        'Frequencies, including False Frequency'
        return np.concatenate((self.Freq, [self.FreqF]))
    
    def __repr__(self):
        txt = ''
        txt += f'Nuclear Spin        = {self.I}\n'
        txt += f'Atomic Number       = {self.Z}\n'
        txt += f'Atomic Mass Number  = {self.A}\n'
        txt += f'Atomic Mass         = {self.mass:.5f} (amu)\n'
        txt += f'Channel Radius      = {self.ac:.7f} (fm)\n'
        txt += f'Energy Bounds       = {self.EB[0]:.3e} < E < {self.EB[1]:.3e} (eV)\n'
        txt += f'False Level Density = {self.FreqF:.7f} (1/eV)\n'
        txt += '\n'
        data = np.vstack((self.Freq, self.w, self.Gnm, self.nDOF, self.Ggm, self.gDOF, self.Gn_trunc, self.MissFrac))
        properties = ['Level Densities', \
                      'Brody Parameters', \
                      'Mean Neutron Width', \
                      'Neutron Width DOF', \
                      'Mean Gamma Width', \
                      'Gamma Width DOF', \
                      'Truncation N Width', \
                      'Missing Fraction']
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
            
    def sample(self, ensemble:str='NNE',
               rng=None, seed:int=None):
        """
        Samples resonance parameters based on the given information.
        
        Parameters:
        ----------
        ensemble :: 'NNE', 'GOE', 'GUE', 'GSE', or 'Poisson'
            The ensemble to use for resonance energy sampling. Default is 'NNE'.
        
        rng      :: default_rng
            A provided `default_rng`. Default is `None`.
    
        seed     :: int
            If no `rng` is provided, then a random number seed can be specified.

        Returns:
        -------
        resonances_caught :: Resonances
            The recorded resonances.
        
        spingroups_caught :: int, array-like
            An ID for the recorded resonances' spingroups.

        resonances_missing :: Resonances
            The missed resonances.
        
        spingroups_missed :: int, array-like
            An ID for the missed resonances' spingroups.
        """

        if rng is None:
            rng = np.random.default_rng(seed)

        # Energy Sampling:
        E          = np.zeros((0,))
        Gn         = np.zeros((0,))
        Gg         = np.zeros((0,))
        spingroups = np.zeros((0,), dtype=int)
        for g in range(self.num_sgs):
            # Energy sampling:
            w = self.w[g] if self.w is not None else None
            E_group  = RMatrix.SampleEnergies(self.EB, self.Freq[g], w=w, ensemble=ensemble, rng=rng)
            
            # Width sampling:
            len_group = len(E_group)
            Gn_group = RMatrix.SampleNeutronWidth(E_group, self.Gnm[g], self.nDOF[g], self.L[g], self.A, self.ac, rng=rng)
            Gg_group = RMatrix.SampleGammaWidth(len_group, self.Ggm[g], self.gDOF[g], rng=rng)
            
            # Append to group:
            E          = np.concatenate((E         , E_group ))
            Gn         = np.concatenate((Gn        , Gn_group))
            Gg         = np.concatenate((Gg        , Gg_group))
            spingroups = np.concatenate((spingroups, g*np.ones((len_group,), dtype=int)))

        # False Resonances:
        if self.FreqF != 0.0:
            # Energy sampling:
            E_false = RMatrix.SampleEnergies(self.EB, self.FreqF, w=None, ensemble='Poisson')
            
            # False width sampling:
            # False widths are sampled by taking the frequency-weighted average of each spingroup's width distributions.
            num_false = len(E_false)
            Gn_false_group = np.zeros((num_false,self.num_sgs))
            Gg_false_group = np.zeros((num_false,self.num_sgs))
            for g in range(self.num_sgs):
                Gn_false_group[:,g] = RMatrix.SampleNeutronWidth(E_false, self.Gnm[g], self.nDOF[g], self.L[g], self.A, self.ac, rng=rng)
                Gg_false_group[:,g] = RMatrix.SampleGammaWidth(num_false, self.Ggm[g], self.gDOF[g], rng=rng)
            cumprobs = np.cumsum(self.Freq) / np.sum(self.Freq)
            R = rng.uniform(size=(num_false,1))
            idx = np.arange(num_false)
            group_idx = np.sum(cumprobs <= R, axis=1)
            Gn_false = Gn_false_group[idx,group_idx]
            Gg_false = Gg_false_group[idx,group_idx]

            # Append to group:
            E          = np.concatenate((E         , E_false ))
            Gn         = np.concatenate((Gn        , Gn_false))
            Gg         = np.concatenate((Gg        , Gg_false))
            spingroups = np.concatenate((spingroups, self.num_sgs*np.ones((num_false,), dtype=int)))

        # Sorting Indices:
        idx = np.argsort(E)
        E  = E[idx]
        Gn = Gn[idx]
        Gg = Gg[idx]
        spingroups = spingroups[idx]

        # Missing Resonances:
        if self.MissFrac is not None:
            if self.given_miss_frac: # given "MissFrac" directly
                miss_frac = np.concatenate((self.MissFrac, [0]))
                missed_idx = (rng.uniform(size=E.shape) < miss_frac[spingroups])
            else: # given Gn_trunc
                Gn_trunc = np.concatenate((self.Gn_trunc, [0]))
                rGn = np.zeros(Gn.shape)
                for g in range(self.num_sgs):
                    spingroup_g = (spingroups == g)
                    rGn[spingroup_g] = Gn[spingroup_g] * RMatrix.ReduceFactor(E[spingroup_g], self.L[g], self.A, self.ac)
                missed_idx = (rGn <= Gn_trunc[spingroups])

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

        # Returning resonance data:
        return resonances_caught, spingroups_caught, resonances_missed, spingroups_missed
    
    def distributions(self, dist_type:str='Wigner', err:float=5e-3):
        """
        Returns the `Distributions` object for the level-spacing, based on the mean parameters
        and provided distribution type, `dist_type`.

        Parameters:
        ----------
        dist_type :: 'Wigner', 'Brody', or 'Missing'
            the level-spacings distribution type. Default is 'Wigner'.
        
        err       :: float
            A probability threshold in which any more missing resonances would be unlikely.

        Returns:
        -------
        distributions :: Distributions
            The `Distributions` object for level-spacings, based on the mean parameters.
        """

        if   dist_type == 'Wigner':
            distributions = RMatrix.Distributions.wigner(self.Freq)
        elif dist_type == 'Brody':
            distributions = RMatrix.Distributions.brody(self.Freq, self.w)
        elif dist_type == 'Missing':
            distributions = RMatrix.Distributions.missing(self.Freq, self.MissFrac, err)
        else:
            raise NotImplementedError(f'The distribution type, "{dist_type}", has not been implemented yet.')
        return distributions
        
    def fit(self, quantity:str, spingroup, cdf:bool=False):
        """
        The expected distribution fit for the specified quantity, `quantity`, for the specified
        spingroup, `spingroup`.

        Parameters:
        ----------
        quantity  :: 'energies', 'level spacing', 'neutron width', 'gamma width', or 'capture width'
            The quantity for which the expected distribution is given.

        spingroup :: int or SpinGroup
            The spingroup for the expected distribution.

        cdf       :: bool
            If true, the expected cumulative density function is provided; else, the probability
            density function is provided. Default = False.
        """

        # Matching spingroup:
        g = self.__match_spingroup(spingroup)

        if   quantity == 'energies':
            if not cdf: # PDF
                f = lambda e: 1.0 / (self.EB[1] - self.EB[0])
            else: # CDF
                f = lambda e: (e - self.EB[0]) / (self.EB[1] - self.EB[0])
        elif quantity == 'level spacing':
            if g == self.num_sgs:
                if not cdf: # PDF
                    f = lambda x: self.FreqF * np.exp(-self.FreqF * x)
                else: # CDF
                    f = lambda x: 1 - np.exp(-self.FreqF * x)
            else:
                if self.w[g] == 1.0:
                    if self.MissFrac[g] == 0.0:
                        dist_type = 'Wigner'
                    else:
                        dist_type = 'Missing'
                else:
                    if self.MissFrac[g] == 0.0:
                        dist_type = 'Brody'
                    else:
                        raise NotImplementedError('The level-spacing distribution for Brody distribution with missing levels has not been implemented yet.')
                if not cdf: # PDF
                    f = self.distributions(dist_type)[g].f0
                else: # CDF
                    f = lambda x: 1.0 - self.distributions(dist_type)[g].f1(x)
        elif quantity == 'neutron width':
            if not cdf: # PDF
                f = lambda rGn: RMatrix.PorterThomasPDF(rGn, self.Gnm[g], trunc=self.Gn_trunc[g], dof=self.nDOF[g])
            else: # CDF
                f = lambda rGn: RMatrix.PorterThomasCDF(rGn, self.Gnm[g], trunc=self.Gn_trunc[g], dof=self.nDOF[g])
        elif quantity in ('gamma width', 'capture width'):
            if not cdf: # PDF
                f = lambda rGg: RMatrix.PorterThomasPDF(rGg, self.Ggm[g], trunc=0.0, dof=self.gDOF[g])
            else: # CDF
                f = lambda rGg: RMatrix.PorterThomasCDF(rGg, self.Ggm[g], trunc=0.0, dof=self.gDOF[g])

        return f
    
    def __match_spingroup(self, spingroup):
        """
        ...
        """

        if spingroup in ('false', 'False'):
            spingroup = self.num_sgs
        elif type(spingroup) == SpinGroup:
            for g in range(self.num_sgs):
                if spingroup == self.sg[g]:
                    spingroup = g
                    break
            else:
                raise ValueError(f'The provided spingroup, {spingroup}, does not match any of the recorded spingroups.')
        elif type(spingroup) != int:
            raise TypeError('The provided spingroup is not an integer ID nor is it a "SpinGroup" object.')
        return spingroup

# =================================================================================================
# Resonances:
# =================================================================================================

class Resonances:
    """
    A data storage object for the resonance parameters, such as resonance energies, and partial
    widths.

    Attributes:
    ----------
    E   :: float, array-like
        Resonance energies.
    Gn  :: float, array-like
        Resonance neutron widths. Default is None.
    Gg  :: float, array-like
        Resonance gamma (capture) widths. Default is None.
    GfA :: float, array-like
        Resonance fission A widths. Default is None.
    GfB :: float, array-like
        Resonance fission B widths. Default is None.
    SG  :: int or SpinGroup, array-like
        Resonance spingroup assignments. Default is None.
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