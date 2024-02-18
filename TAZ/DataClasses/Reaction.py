import numpy as np
import pandas as pd
from typing import List

from TAZ import Theory
from TAZ.DataClasses import Spingroup
from TAZ.DataClasses import Particle, Neutron
from TAZ.DataClasses import Resonances

__doc__ = """
This file keeps the "Reaction" class. The "Reaction" class contains all relevent
information that is independent of individual resonances, such as the target and projectile
particles, spingroup combinations, mean level densities, mean partial widths, etc.
"""

def spingroupParameter(mean_parameters, num_groups:int, dtype:type=float):
    """
    Correctly formats the spingroup-based parameters.

    Parameters:
    ----------
    mean_parameters :: array-like
        The parameters that are being formatted.
    num_spingroups  :: int
        The number of recorded spingroups.
    dtype           :: type
        The type for formatting the spingroup-based parameters. Default = float.

    Returns:
    -------
    mean_parameters :: array-like
        The reformatted spingroup-based parameters.
    """

    if (not hasattr(mean_parameters, '__iter__')) \
        or (np.array(mean_parameters).size != num_groups):
        from inspect import stack
        name = stack()[1][0].f_locals[mean_parameters]
        raise TypeError(f'"{name}" must be an array with length equal to the number of spingroups, {num_groups}.')
    return np.array(mean_parameters, dtype=dtype).reshape(num_groups,)

# =================================================================================================
#    Reaction:
# =================================================================================================

class Reaction:
    """
    Reaction is a class that contains information about a particular reaction, such as the
    target particle, projectile particle, spingroups, and mean resonance parameters such as mean
    level-spacing, and mean partial widths.
    """

    DEFAULT_GDOF = 500 # the default number of degrees of freedom on the gamma (capture) width.

    def __init__(self,
                 targ:Particle=None, proj:Particle=Neutron,
                 ac:float=None,
                 EB:tuple=None, false_dens:float=0.0,
                 spingroups:List[Spingroup]=None,
                 lvl_dens:list=None, MLS:list=None,
                 brody_param:list=None,
                 gn2m:list=None, nDOF:list=None,
                 gg2m:list=None, gDOF:list=None,
                 Gn_trunc:list=None, MissFrac:list=None):
        """
        Initializes reaction parameters with keyword arguments.

        Attributes:
        ----------
        targ        :: Particle
            Target particle object.
        proj        :: Particle
            Projectile particle object. Default = Neutron.
        ac          :: float
            Reaction channel radius in femtometers.    
        EB          :: float [2]
            Energy range for evaluation.
        false_dens  :: float
            False resonance level density.
        spingroups  :: List [Spingroup]
            Spingroups for the reaction.
        lvl_dens    :: float [G]
            Resonance level densities for each spingroup.
        MLS         :: float [G]
            Resonance mean level spacings for each spingroup.
        brody_param :: float [G]
            Brody resonance parameter.
        gn2m        :: float [G]
            Resonance mean reduced neutron widths for each spingroup.
        nDOF        :: float [G]
            Resonance neutron width degrees of freedom for each spingroup.
        Gn_trunc    :: float [G]
            Lowest recordable neutron width.
        gg2m        :: float [G]
            Resonance mean reduced gamma (capture) width for each spingroup.
        gDOF        :: float [G]
            Resonance gamma (capture) width degrees of freedom for each spingroup.
        MissFrac    :: float [G]
            Fraction of Resonances that have been missed.
        """

        # Target Particle:
        if targ is not None:
            if type(targ) != Particle:
                raise TypeError('"targ" must by a "Particle" object.')
            self.targ = targ
        else:
            self.targ = None

        # Projectile Particle:
        if proj is not None:
            if type(proj) != Particle:
                raise TypeError('"proj" must by a "Particle" object.')
            self.proj = proj
        else:
            self.proj = None

        # Channel Radius:
        if ac is not None:
            if   ac > 1.00: print(Warning(f'The channel radius, {ac} 1e-12 cm, is quite high. Make sure it is in units of square-root barns or 1e-12 cm.'))
            elif ac < 0.08: print(Warning(f'The channel radius, {ac} 1e-12 cm, is quite low. Make sure it is in units of square-root barns or 1e-12 cm.'))
            self.ac = float(ac)
        elif (self.proj is not None and self.proj.radius is not None) \
         and (self.targ is not None and self.targ.radius is not None):
            self.ac = self.proj.radius + self.targ.radius
        else:
            self.ac = None
        
        # Energy Range:
        if EB is not None:
            if len(EB) != 2:        raise ValueError('"EB" can only have two values for an interval.')
            elif EB[0] > EB[1]:     raise ValueError('"EB" must be a valid increasing interval.')
            self.EB = (float(EB[0]), float(EB[1]))
        else:
            self.EB = None
        
        # False Level Density:
        if false_dens is not None:   self.false_dens = float(false_dens)
        else:                        self.false_dens = 0.0

        # Spingroups:
        if spingroups is None:
            raise ValueError('The spingroups are a required argument for initialization of a "Reaction" object.')
        elif (type(spingroups) != list) or (type(spingroups[0]) != Spingroup):
            raise TypeError('"spingroups" must be a list of "Spingroup" objects.')
        self.spingroups = spingroups
        self.num_groups = len(spingroups)
        
        # Level Densities:
        if lvl_dens is not None and MLS is not None:
            raise ValueError('Cannot have both mean level spacing and level densities.')
        elif lvl_dens is not None:
            self.lvl_dens = spingroupParameter(lvl_dens, self.num_groups, dtype=float)
        elif MLS is not None:
            self.lvl_dens = 1.0 / spingroupParameter(MLS, self.num_groups, dtype=float)
        else:
            self.lvl_dens = None

        # Brody Parameter:
        if brody_param is not None:
            if hasattr(brody_param, '__iter__'):
                self.brody_param = spingroupParameter(brody_param, self.num_groups, dtype=float)
            else:
                self.brody_param = float(brody_param) * np.ones((self.num_groups,))
        else:
            self.brody_param = np.ones((self.num_groups,))

        # Mean Neutron Widths:
        if gn2m is not None:    self.gn2m = spingroupParameter(gn2m, self.num_groups, dtype=float)
        else:                   self.gn2m = None

        # Neutron Channel Degrees of Freedom:
        if nDOF is not None:    self.nDOF = spingroupParameter(nDOF, self.num_groups, dtype=int)
        else:                   self.nDOF = np.ones((self.num_groups,), dtype=int)

        # Mean Gamma Widths:
        if gg2m is not None:    self.gg2m = spingroupParameter(gg2m, self.num_groups, dtype=float)
        else:                   self.gg2m = None

        # Gamma Channel Degrees of Freedom:
        if gDOF is not None:    self.gDOF = spingroupParameter(gDOF, self.num_groups, dtype=int)
        else:                   self.gDOF = self.DEFAULT_GDOF * np.ones((self.num_groups,), dtype=int)

        # Truncation Width:
        if Gn_trunc is not None:
            self.Gn_trunc = float(Gn_trunc) # FIXME: MAKE THIS AS A FUNCTION OF ENERGY!
            self.Gn_trunc_provided = True
        else:
            self.Gn_trunc = 0.0 # meV
            self.Gn_trunc_provided = False

        # Missing Fraction:
        if MissFrac is not None:
            self.MissFrac = spingroupParameter(MissFrac, self.num_groups, dtype=float) # FIXME: MAKE THIS AS A FUNCTION OF ENERGY!
        elif self.Gn_trunc_provided:
            self.MissFrac = Theory.fraction_missing_Gn(self.Gn_trunc, self.gn2m, self.nDOF, self.ac, self.gn2m, self.nDOF)
        else:
            self.MissFrac = np.zeros((self.num_groups,), dtype=float)

    @property
    def L(self):
        'Orbital Angular Momentum'
        return np.array([spingroup.L for spingroup in self.spingroups])
    @property
    def J(self):
        'Total Angular Momentum'
        return np.array([spingroup.J for spingroup in self.spingroups])
    @property
    def S(self):
        'Channel Spin'
        return np.array([spingroup.S for spingroup in self.spingroups])
    @property
    def MLS(self):
        'Mean Level Spacing'
        return 1.0 / self.lvl_dens
    @property
    def lvl_dens_all(self):
        'Level densities, including False level density'
        return np.concatenate((self.lvl_dens, [self.false_dens]))
    
    def __repr__(self):
        txt = ''
        txt += f'Target Particle      = {self.targ.name}\n'
        txt += f'Projectile Particle  = {self.proj.name}\n'
        txt += f'Channel Radius       = {self.ac:.7f} (fm)\n'
        txt += f'Energy Bounds        = {self.EB[0]:.3e} < E < {self.EB[1]:.3e} (eV)\n'
        txt += f'False Level Density  = {self.false_dens:.7f} (1/eV)\n'

        if self.Gn_trunc_provided:
            txt += f'Trunc. Neutron Width = {self.Gn_trunc:.7f} (1/meV)\n'

        txt += '\n'

        param_vals  = [self.lvl_dens, self.brody_param, self.gn2m, self.nDOF, self.gg2m, self.gDOF]
        param_names = ['Level Densities', \
                      'Brody Parameters', \
                      'Mean Neutron Width', \
                      'Neutron Width DOF', \
                      'Mean Gamma Width', \
                      'Gamma Width DOF']
        
        if isinstance(self.MissFrac, np.ndarray):
            param_names.append('Missing Fraction')
            param_vals.append(self.MissFrac)

        data = np.vstack(param_vals)
        txt += str(pd.DataFrame(data=data, index=param_names, columns=self.spingroups))
        return txt
    def __str__(self):
        return self.__repr__()

    @classmethod
    def readJSON(cls, file:str):
        """
        Creates a Reaction object by importing a JSON file.
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
        resonances_caught  :: Resonances
            The recorded resonances.
        spingroups_caught  :: int, array-like
            An ID for the recorded resonances' spingroups.
        resonances_missing :: Resonances
            The missed resonances.
        spingroups_missed  :: int, array-like
            An ID for the missed resonances' spingroups.
        """

        if rng is None:
            rng = np.random.default_rng(seed)

        # Energy Sampling:
        E          = np.zeros((0,))
        Gn         = np.zeros((0,))
        Gg         = np.zeros((0,))
        spingroups = np.zeros((0,), dtype=int)
        for g in range(self.num_groups):
            # Energy sampling:
            brody_param = self.brody_param[g] if self.brody_param is not None else None
            E_group  = Theory.SampleEnergies(self.EB, self.lvl_dens[g], w=brody_param, ensemble=ensemble, rng=rng)
            
            # Width sampling:
            len_group = len(E_group)
            Gn_group = Theory.SampleNeutronWidth(E_group, self.gn2m[g], self.nDOF[g], self.L[g], ac=self.ac,
                                                  mass_targ=self.targ.mass, mass_proj=self.proj.mass,
                                                  rng=rng)
            Gg_group = Theory.SampleGammaWidth(len_group, self.gg2m[g], self.gDOF[g], rng=rng)
            
            # Append to group:
            E          = np.concatenate((E         , E_group ))
            Gn         = np.concatenate((Gn        , Gn_group))
            Gg         = np.concatenate((Gg        , Gg_group))
            spingroups = np.concatenate((spingroups, g*np.ones((len_group,), dtype=int)))

        # False Resonances:
        if self.false_dens != 0.0:
            # Energy sampling:
            E_false = Theory.SampleEnergies(self.EB, self.false_dens, w=None, ensemble='Poisson')
            
            # False width sampling:
            # False widths are sampled by taking the level-density-weighted average of each spingroup's width distributions.
            num_false = len(E_false)
            Gn_false_group = np.zeros((num_false,self.num_groups))
            Gg_false_group = np.zeros((num_false,self.num_groups))
            for g in range(self.num_groups):
                Gn_false_group[:,g] = Theory.SampleNeutronWidth(E_false, self.gn2m[g], self.nDOF[g], self.L[g], ac=self.ac,
                                                                 mass_targ=self.targ.mass, mass_proj=self.proj.mass,
                                                                 rng=rng)
                Gg_false_group[:,g] = Theory.SampleGammaWidth(num_false, self.gg2m[g], self.gDOF[g], rng=rng)
            idx = np.arange(num_false)
            group_idx = rng.choice(self.num_groups, size=(num_false,), p=self.lvl_dens/np.sum(self.lvl_dens))
            Gn_false = Gn_false_group[idx,group_idx]
            Gg_false = Gg_false_group[idx,group_idx]

            # Append to group:
            E          = np.concatenate((E         , E_false ))
            Gn         = np.concatenate((Gn        , Gn_false))
            Gg         = np.concatenate((Gg        , Gg_false))
            spingroups = np.concatenate((spingroups, self.num_groups*np.ones((num_false,), dtype=int)))

        # Sorting Indices:
        idx = np.argsort(E)
        E  = E[idx]
        Gn = Gn[idx]
        Gg = Gg[idx]
        spingroups = spingroups[idx]

        # Missing Resonances:
        if self.Gn_trunc_provided: # given Gn_trunc
            missed_idx = (Gn <= self.Gn_trunc)
        else:
            miss_frac = np.concatenate((self.MissFrac, [0]))
            missed_idx = (rng.uniform(size=E.shape) < miss_frac[spingroups])

        # Caught resonances:
        E_caught  =  E[~missed_idx]
        Gn_caught = Gn[~missed_idx]
        Gg_caught = Gg[~missed_idx]
        resonances_caught = Resonances(E=E_caught, Gn=Gn_caught, Gg=Gg_caught, ladder_bounds=self.EB)
        spingroups_caught = spingroups[~missed_idx]
        
        # Missing resonances:
        E_missed  =  E[missed_idx]
        Gn_missed = Gn[missed_idx]
        Gg_missed = Gg[missed_idx]
        resonances_missed = Resonances(E=E_missed, Gn=Gn_missed, Gg=Gg_missed, ladder_bounds=self.EB)
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

        distributions = []
        if   dist_type == 'Wigner':
            for g in range(self.num_groups):
                distribution = Theory.LevelSpacingDists.WignerGen(lvl_dens=self.lvl_dens[g])
                distributions.append(distribution)
        elif dist_type == 'Brody':
            for g in range(self.num_groups):
                distribution = Theory.LevelSpacingDists.BrodyGen(lvl_dens=self.lvl_dens[g], w=self.brody_param[g])
                distributions.append(distribution)
        elif dist_type == 'Missing':
            for g in range(self.num_groups):
                # if type(self.MissFrac[g]) != float:
                #     raise NotImplementedError('Missing Level-spacing generator only works with constant float MissFrac at this time.')
                distribution = Theory.LevelSpacingDists.MissingGen(lvl_dens=self.lvl_dens[g], pM=self.MissFrac[g], err=err)
                distributions.append(distribution)
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
        spingroup :: int or Spingroup
            The spingroup for the expected distribution.
        cdf       :: bool
            If true, the expected cumulative density function is provided; else, the probability
            density function is provided. Default = False.

        Returns:
        -------
        fit       :: function
            The probability distribution of the specified quantity.
        """

        # Matching spingroup to id:
        g = Spingroup.id(spingroup, self.spingroups)

        # Determining and returning distribution:
        if   quantity == 'energies':
            if not cdf: # PDF
                fit = lambda e: 1.0 / (self.EB[1] - self.EB[0])
            else: # CDF
                fit = lambda e: (e - self.EB[0]) / (self.EB[1] - self.EB[0])
        elif quantity == 'level spacing':
            if g == self.num_groups:
                if not cdf: # PDF
                    fit = lambda x: self.false_dens * np.exp(-self.false_dens * x)
                else: # CDF
                    fit = lambda x: 1 - np.exp(-self.false_dens * x)
            else:
                if self.brody_param[g] == 1.0:
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
                    fit = self.distributions(dist_type)[g].pdf
                else: # CDF
                    fit = self.distributions(dist_type)[g].cdf
        elif quantity == 'neutron width':
            # FIXME: THESE TRUNCTATIONS ARE NOT THE SAME AS GN_TRUNC!!!
            # if not cdf: # PDF
            #     fit = Theory.porter_thomas_dist(mean=self.gn2m[g], df=self.nDOF[g], trunc=self.Gn_trunc[g]).pdf
            # else: # CDF
            #     fit = Theory.porter_thomas_dist(mean=self.gn2m[g], df=self.nDOF[g], trunc=self.Gn_trunc[g]).cdf
            if not cdf: # PDF
                fit = Theory.porter_thomas_dist(mean=self.gn2m[g], df=self.nDOF[g], trunc=0.0).pdf
            else: # CDF
                fit = Theory.porter_thomas_dist(mean=self.gn2m[g], df=self.nDOF[g], trunc=0.0).cdf
        elif quantity in ('gamma width', 'capture width'):
            if not cdf: # PDF
                fit = Theory.porter_thomas_dist(mean=self.gg2m[g], df=self.gDOF[g], trunc=0.0).pdf
            else: # CDF
                fit = Theory.porter_thomas_dist(mean=self.gg2m[g], df=self.gDOF[g], trunc=0.0).cdf
        return fit