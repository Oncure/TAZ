import numpy as np

__doc__ = """
This module is the collection of relevant R-Matrix Theory quantities. Many of these equations are
found in the ENDF and SAMMY manuals.
"""

def NuclearRadius(A:int) -> float:
    """
    Finds the nuclear radius from the atomic mass.

    Based on equation D.14 in ENDF manual.

    Parameters:
    ----------
    A  :: int
        Atomic mass of the isotope.

    Returns:
    --------
    ac :: float
        Nuclear radius of the isotope.
    """
    return 1.23 * A**(1/3) + 0.8 # fm = 10^-15 m

def Rho(mass_targ:float, ac:float, E,
        mass_proj:float=None,
        mass_targ_after:float=None, mass_proj_after:float=None,
        E_thres:float=None):
    """
    Finds the momentum factor, `rho`.

    Based on equation II A.9 in the SAMMY manual.

    Parameters:
    ----------
    mass_targ       :: float
        Mass of the target isotope.
    ac              :: float
        Channel radius .
    E               :: float, array-like
        Energy points for which Rho is evaluated.
    mass_proj       :: float
        Mass of the projectile. Default = 1.008665 amu (neutron mass).
    mass_targ_after :: float
        Mass of the target after the reaction. Default = mass_targ.
    mass_proj_after :: float
        Mass of the target before the reaction. Default = mass_proj.
    E_thres         :: float
        Threshold energy for the reaction. Default is calculated from Q-value.
        
    Returns:
    --------
    rho :: float, array-like
        Momentum factor, ρ.
    """

    if mass_proj is None:
        mass_proj = 1.008665 # amu (neutron mass)
    if mass_targ_after is None:
        mass_targ_after = mass_targ # assume elastic scattering
    if mass_proj_after is None:
        mass_proj_after = mass_proj # assume elastic scattering
    if E_thres is None:
        Q_value = mass_targ + mass_proj - mass_targ_after - mass_proj_after
        E_thres = - ((mass_targ + mass_proj)/mass_targ) * Q_value # Eq. II C2.1 in the SAMMY manual

    # Error Checking:
    if any(E < E_thres):
        raise ValueError(f'The given energies are below the threshold energy of {E_thres} eV.')

    CONSTANT = 0.0001546691274; # ((amu)^{1/2} * fm) / h_bar --> eV^{-1/2}
    mass_ratio_before = mass_targ / (mass_proj + mass_targ)
    mass_ratio_after  = 2 * mass_proj_after * mass_targ_after / (mass_proj_after + mass_targ_after)
    Delta_E = E-E_thres
    rho = CONSTANT * np.sqrt((mass_ratio_before * mass_ratio_after) * Delta_E) * ac
    return rho

def PenetrationFactor(rho, l:int):
    """
    Finds the Penetration factor.

    Based on table II A.1 in the SAMMY manual.

    Parameters:
    ----------
    rho :: float, array-like
        Momentum factor.
    l   :: int, array-like
        Orbital angular momentum quantum number.

    Returns:
    --------
    pen_factor :: float, array-like
        Penetration factor.
    """

    def _penetrationFactor(rho, l:int):
        rho2 = rho**2
        if   l == 0:
            return rho
        elif l == 1:
            return rho*rho2    / (  1 +    rho2)
        elif l == 2:
            return rho*rho2**2 / (  9 +  3*rho2 +   rho2**2)
        elif l == 3:
            return rho*rho2**3 / (225 + 45*rho2 + 6*rho2**2 + rho2**3)
        else: # l >= 4
            
            # l = 3:
            denom = (225 + 45*rho2 + 6*rho2**2 + rho2**3)
            P = rho*rho2**3 / denom
            S = -(675 + 90*rho2 + 6*rho2**2) / denom

            # Iteration equation:
            for l_iter in range(4,l+1):
                mult = rho2 / ((l_iter-S)**2 + P**2)
                P = mult*P
                S = mult*S - l_iter
            return P

    if hasattr(l, '__iter__'): # is iterable
        pen_factor = np.zeros((rho.shape[0],l.shape[1]))
        for g, lg in enumerate(l[0,:]):
            pen_factor[:,g] = _penetrationFactor(rho,lg)
    else: # is not iterable
        pen_factor = np.array(_penetrationFactor(rho,l))
    return pen_factor