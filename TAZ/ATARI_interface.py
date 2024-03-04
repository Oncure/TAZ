import pandas as pd

from TAZ import Spingroup
from TAZ import Reaction

__doc__ = """
This file is used to interface between ATARI and TAZ Dataclasses.
"""

def ATARI_to_TAZ_reaction(particle_pair, **kwargs):
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
    """

    # raise NotImplementedError('...')

    energy_bounds = (min(particle_pair.energy_range), max(particle_pair.energy_range))
    
    spingroups = []
    Dm   = []
    gn2m = []
    nDOF = []
    gg2m = []
    gDOF = []
    for Jpi, mean_param_dict in particle_pair.spin_groups.items():
        L = mean_param_dict['Ls']
        J = abs(Jpi)
        # S = mean_param_dict['chs']
        spingroups.append(Spingroup(L, J))
        Dm  .append(mean_param_dict['<D>'])
        gn2m.append(mean_param_dict['<gn2>'])
        nDOF.append(mean_param_dict['<n_dof>'])
        gg2m.append(mean_param_dict['<gg2>'])
        gDOF.append(mean_param_dict['<g_dof>'])

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
        'gDOF'       : gDOF
    }

    # Resonance ladder:
    reaction.resonances = particle_pair.resonance_ladder

    for key, value in kwargs.items():
        reaction_params[key] = value
    reaction = Reaction(**reaction_params)
    return reaction

def TAZ_to_ATARI_reaction(reaction:Reaction):
    """
    ...
    """

    raise NotImplementedError('...')
    return particle_pair