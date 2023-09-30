import numpy as np

from .RMatrix import NuclearRadius
from .WidthDists import ReduceFactor

__doc__ = """
This module compiles Mean Parameter Estimation Methods.
"""

# =================================================================================================
#    Mean Parameter Estimation:
# =================================================================================================

def MeanSpacingEst(E, SGs, method='mean'):
    """
    ...
    """

    n = np.max(SGs) + 1
    if method == 'mean':
        return np.array([np.mean(np.diff(E[SGs == g])) for g in range(n)]).reshape(1,-1)
    else:
        raise NotImplementedError('Unknown method.')

def MeanNWidthEst(Gn, SGs, E, L, DoF, A, ac=None, method='mean'):
    """
    ...
    """

    if ac == None:
        ac = NuclearRadius(A)

    #FIXME: FIXME FIXME FIXME FIXME ...
    # Gn_red = Gn * ReduceFactor(E, L[SGs], A, ac)
    Gn_red = Gn * ReduceFactor(E, 0, A, ac)
    n = np.max(SGs) + 1
    if method == 'mean':
        return np.array([np.mean(Gn_red[SGs == g]) for g in range(n)]).reshape(1,-1)
    else:
        raise NotImplementedError('Unknown method.')

def MeanGWidthEst(Gg, SGs, DoF, method='mean'):
    """
    ...
    """

    n = np.max(SGs) + 1
    if method == 'mean':
        return np.array([np.mean(Gg[SGs == g]) for g in range(n)]).reshape(1,-1)
    else:
        raise NotImplementedError('Unknown method.')
    
# ...

# =================================================================================================
#    Missing and False Resonance PDFs:
# =================================================================================================
    
# ...