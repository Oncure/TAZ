import numpy as np

__doc__ = """
This module is the collection of relevant R-Matrix Theory quantities. Many of these equations are
found in the ENDF and SAMMY manuals.
"""

def NuclearRadius(A:float) -> float:
    """
    Finds the nuclear radius from the atomic mass.

    Based on equation D.14 in ENDF manual.

    Inputs:
    ------
    A  :: float
        Atomic mass of the isotope.

    Returns:
    --------
    ac :: float
        Nuclear radius of the isotope.
    """
    return 1.23 * A**(1/3) + 0.8 # fm = 10^-15 m

def Rho(A:float, ac:float, E, E_thres:float=0.0):
    """
    Finds the momentum factor, `rho`.

    Based on equation II A.9 in the SAMMY manual.

    Inputs:
    ------
    A       :: float
        Atomic mass of the isotope.
    ac      :: float
        Channel radius.
    E       :: float, array-like
        Energy points for which Rho is evaluated.
    E_thres :: float
        Threshold energy for the reaction. Default is 0.0

    Returns:
    --------
    float, array-like
        Momentum factor, `rho`.
    """
    if any(E < E_thres):
        raise ValueError(f'The given energies are below the threshold energy of {E_thres} eV.')
    CONSTANT = 0.002197; # sqrt(2Mn)/hbar; units of (10^(-12) cm sqrt(eV)^-1)
    return CONSTANT*ac*(A/(A+1))*np.sqrt(E-E_thres)

def PenetrationFactor(rho, l:int):
    """
    Finds the Penetration factor.

    Based on table II A.1 in the SAMMY manual.

    Inputs:
    ------
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

    if hasattr(l, '__iter__'): # Is iterable
        pen_factor = np.zeros((rho.shape[0],l.shape[1]))
        for t, lt in enumerate(l[0,:]):
            pen_factor[:,t] = _penetrationFactor(rho,lt)
    else: # Is not iterable
        pen_factor = np.array(_penetrationFactor(rho,l))
    return pen_factor