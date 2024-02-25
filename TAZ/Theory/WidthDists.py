import numpy as np
from scipy.special import gammainc, iv

from TAZ.DataClasses import MASS_NEUTRON
from TAZ.Theory.RMatrix import PenetrationFactor, Rho

__doc__ = """
This module contains partial width probability distributions.
"""

# =================================================================================================
#    Width Probability Distributions
# =================================================================================================

def fraction_missing_gn2(trunc:float, gn2m:float=1.0, dof:int=1):
    """
    Gives the fraction of missing resonances due to the truncation in reduced neutron width.

    Parameters:
    ----------
    trunc :: float
        The lower limit on the reduced neutron width.
    gn2m  :: float
        The mean reduced neutron width. Default = 1.0.
    dof   :: int
        The number of degrees of freedom for the chi-squared distribution.

    Returns:
    -------
    fraction_missing :: float
        The fraction of missing resonances within the spingroup.
    """
    fraction_missing = gammainc(dof/2, dof*trunc/(2*gn2m))
    return fraction_missing

def fraction_missing_Gn(trunc:float,
                        l:int, mass_targ:float, ac:float,
                        gn2m:float=1.0, dof:int=1):
    """
    Gives the fraction of missing resonances due to the truncation in partial neutron width.

    Parameters:
    ----------
    trunc     :: float
        The lower limit on the reduced neutron width.
    l         :: int
        The orbital-angular momentum for the channel.
    mass_targ :: float
        Mass of the target nucleus.
    ac        :: float
        Channel radius.
    gn2m      :: float
        The mean reduced neutron width. Default = 1.0.
    dof       :: int
        The number of degrees of freedom for the chi-squared distribution.

    Returns:
    -------
    fraction_missing :: function: float -> float
        The fraction of missing resonances within the spingroup as a function of energy.
    """

    def func(E):
        gn2_trunc = trunc * ReduceFactor(E, l, mass_targ, ac)
        fraction_missing = fraction_missing_gn2(gn2_trunc, gn2m, dof)
        return fraction_missing
    return func
    
def ReduceFactor(E, l:int, mass_targ:float, ac:float,
                 mass_proj:float=MASS_NEUTRON,
                 mass_targ_after:float=None,
                 mass_proj_after:float=None,
                 E_thres:float=None):
    """
    Multiplication factor to convert from neutron width to reduced neutron width.

    Parameters:
    ----------
    E               :: float, array-like
        Resonance energies.
    l               :: int, array-like
        Orbital angular momentum number.
    mass_proj       :: float
        Mass of the projectile. Default = 1.008665 amu (neutron mass).
    mass_targ_after :: float
        Mass of the target after the reaction. Default = mass_targ.
    mass_proj_after :: float
        Mass of the target before the reaction. Default = mass_proj.
    E_thres         :: float
        Threshold energy for the reaction. Default is calculated from Q-value.

    Returns:
    -------
    reduce_factor :: float, array-like
        A multiplication factor that converts neutron widths into reduced neutron widths.
    """

    rho = Rho(mass_targ, ac, E,
              mass_proj, mass_targ_after, mass_proj_after, E_thres)
    reduce_factor = 1.0 / (2.0*PenetrationFactor(rho,l))
    return reduce_factor

# def FissionDistPDF(Gf,
#                    gf2mA:float     , dofA:int=1,
#                    gf2mB:float=None, dofB:int=1,
#                    trunc:float=0.0):
#     """
#     ...
#     """

#     if gf2mB == None:
#         return PorterThomasPDF(Gf, gf2mA, trunc, dofA)
#     else:
#         if trunc != 0.0:
#             raise NotImplementedError('Truncation has not been implemented yet.')
#         elif not (dofA == dofB == 1):
#             raise NotImplementedError('The fission distribution with more than 1 degree of freedom has not been implemented yet.')
        
#         # Source:   https://www.publish.csiro.au/ph/pdf/PH750489
#         a = Gf * (1/gf2mB - 1/gf2mA)/4
#         b = Gf * (1/gf2mB + 1/gf2mA)/4
#         prob = np.exp(-b) * iv(0, a) / np.sqrt(4*gf2mA*gf2mB)
#         return prob