import numpy as np
import pandas as pd

from TAZ.DataClasses import Spingroups

__doc__ = """
This file keeps the "Resonances" class. The "Resonances" class contains resonance-specific
information such as energy, partial widths, and spingroup assignments.
"""

# =================================================================================================
# Resonances:
# =================================================================================================

class Resonances:
    """
    A data storage object for the resonance parameters, such as resonance energies, and partial
    widths. Spingroup assignments can optionally be stored as well.

    Attributes:
    ----------
    E             :: float, array-like
        Resonance energies.
    Gn            :: float, array-like
        Resonance neutron widths. Default is None.
    Gg            :: float, array-like
        Resonance gamma (capture) widths. Default is None.
    GfA           :: float, array-like
        Resonance fission A widths. Default is None.
    GfB           :: float, array-like
        Resonance fission B widths. Default is None.
    SG            :: int or Spingroup, array-like
        Resonance spingroup assignments. Default is None.
    ladder_bounds ::  float [2]
        Resonance ladder bounds. Default is None.
    """

    def __init__(self, E, Gn=None, Gg=None, GfA=None, GfB=None, SG=None, ladder_bounds:tuple=None):
        """
        Creates a Resonances object.

        Parameters:
        ----------
        E             :: float, array-like
            Resonance energies.
        Gn            :: float, array-like
            Resonance neutron widths. Default is None.
        Gg            :: float, array-like
            Resonance gamma (capture) widths. Default is None.
        GfA           :: float, array-like
            Resonance fission A widths. Default is None.
        GfB           :: float, array-like
            Resonance fission B widths. Default is None.
        SG            :: int or Spingroup, array-like
            Resonance spingroup assignments. Default is None.
        ladder_bounds :: tuple[float]
            Resonance ladder bounds. Default is None.
        """

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
            if type(SG) == Spingroups:
                SG = SG.SGs
            self.SG  = np.array(SG ).reshape(-1)[indices]
        if ladder_bounds is not None:
            if len(ladder_bounds) != 2:
                raise ValueError('"ladder_bounds" can only have two values for an interval.')
            elif ladder_bounds[0] > ladder_bounds[1]:
                raise ValueError('"ladder_bounds" must be a valid increasing interval.')
            self.ladder_bounds = (float(ladder_bounds[0]), float(ladder_bounds[1]))
        else:
            self.ladder_bounds = None

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
        data = np.column_stack(data)
        table_str = '\n'.join(str(pd.DataFrame(data=data, columns=self.properties)).split('\n')[:-2])
        return table_str

    @property
    def len(self):
        return self.E.size
    def __len__(self):
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