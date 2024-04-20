import numpy as np
import pandas as pd

from TAZ import Spingroup
from TAZ import Reaction

__doc__ = """
This file is used to interface between ATARI and TAZ Dataclasses.
"""

def ATARI_to_TAZ(particle_pair, **kwargs):
    """
    Converts an ATARI Particle_Pair object to a TAZ Reaction object.

    Parameters
    ----------
    particle_pair: Particle_Pair
        A Particle_Pair instance from ATARI.
    **kwargs: dict
        Additional Reaction keyword arguments for reaction that have not yet been specified.

    Returns
    -------
    reaction: Reaction
        A TAZ Reaction dataclass instance.
    resonances: DataFrame
        A dataframe of resonance parameters.
    spingroup_IDs: ndarray of int
        TAZ spingroup IDs (which may be different than J_IDs).
    """

    energy_bounds = (min(particle_pair.energy_range), max(particle_pair.energy_range))
    
    spingroups = []
    Dm         = []
    gn2m       = []
    nDOF       = []
    gg2m       = []
    gDOF       = []
    J_ID       = []
    for Jpi, mean_param_dict in particle_pair.spin_groups.items():
        J = abs(Jpi)
        for L in np.unique(mean_param_dict['Ls']):
            # S = ???
            spingroups.append(Spingroup(L, J))
            Dm  .append(mean_param_dict['<D>'])
            gn2m.append(mean_param_dict['<gn2>'])
            nDOF.append(mean_param_dict['n_dof'])
            gg2m.append(mean_param_dict['<gg2>'])
            gDOF.append(mean_param_dict['g_dof'])
            J_ID.append(mean_param_dict['J_ID'])

    # Resonances:
    resonances = particle_pair.resonance_ladder.sort_values(by=['E'], ignore_index=True)
    res_J_IDs = resonances['J_ID'].to_numpy()
    
    # Spingroup IDs:
    spingroup_IDs = np.empty_like(res_J_IDs, dtype=int)
    for TAZ_ID, jid in enumerate(J_ID):
        spingroup_IDs[res_J_IDs == jid] = TAZ_ID

    reaction_params = {
        'targ'       : particle_pair.target,
        'proj'       : particle_pair.projectile,
        'ac'         : particle_pair.ac,
        'EB'         : energy_bounds,
        'spingroups' : spingroups,
        'MLS'        : Dm,
        'gn2m'       : gn2m,
        'nDOF'       : nDOF,
        'gg2m'       : gg2m,
        'gDOF'       : gDOF,
        'J_ID'       : J_ID,
        'resonances' : resonances
    }

    for key, value in kwargs.items():
        reaction_params[key] = value
    reaction = Reaction(**reaction_params)
    return reaction, resonances, spingroup_IDs

def TAZ_to_ATARI(reaction:Reaction, **kwargs):
    """
    ...
    """

    raise NotImplementedError()
    return particle_pair