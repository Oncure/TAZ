
__doc__ = """
This file defines all classes related to spingroups. "halfint" is a class for half-integers.
"SpinGroup" is a class to define one spingroup (a pair of orbital angular momentum, total angular
momentum, and channel spin). "SpinGroups" is a class for storing multiple "SpinGroup" objects.
"""

class halfint:
    """
    Data type for half-integers to represent spins.
    """

    def __init__(self, value):
        if type(value) == halfint:
            self = value
        else:
            if value % 0.5 != 0.0:
                raise ValueError(f'The number, {value}, is not a half-integer.')
            self.__2x_value = int(2*value)
    
    @property
    def parity(self):   return '+' if self.__2x_value >= 0 else '-'
    @property
    def value(self):    return 0.5 * float(self.__2x_value)

    def __repr__(self):
        if self.__2x_value % 2 == 0:
            return f'{self.__2x_value//2}'
        else:
            return f'{self.__2x_value}/2'

    # Arithmetic:
    def __float__(self):
        return self.value
    def __eq__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value == other.value
        else:
            return self.value == other
    def __ne__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value != other.value
        else:
            return self.value != other
    def __lt__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value < other.value
        else:
            return self.value < other
    def __le__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value <= other.value
        else:
            return self.value <= other
    def __gt__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value > other.value
        else:
            return self.value > other
    def __ge__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value >= other.value
        else:
            return self.value >= other
    def __add__(self, other):
        if type(other) == self.__class__:
            return self.__class__(self.value + other.value)
        elif type(other) == int:
            return self.__class__(self.value + other)
        else:
            return self.value + other
    def __radd__(self, other):
        if type(other) == int:
            return self.__class__(other + self.value)
        else:
            return other + self.value
    def __sub__(self, other):
        if type(other) == self.__class__:
            return self.__class__(self.value - other.value)
        elif type(other) == int:
            return self.__class__(self.value - other)
        else:
            return self.value - other
    def __rsub__(self, other):
        if type(other) == int:
            return self.__class__(other - self.value)
        else:
            return other - self.value
    def __mul__(self, other):
        if type(other) == self.__class__:
            return self.value * other.value
        elif (type(other) == int) and (other % 2 == 0):
            return self.__2x_value * (other // 2)
        else:
            return self.value * other
    def __rmul__(self, other):
        if (type(other) == int) and (other % 2 == 0):
            return self.__2x_value * (other // 2)
        else:
            return self.value * other

class SpinGroup:
    """
    A class containing the orbital angular momentum, "L", and the total spin, "J", for the
    reaction. The quantum number, "S", can also be given optionally.

    Attributes:
    ----------
    L :: int
        Orbital angular momentum.
    J :: halfint
        Total angular momentum.
    S :: halfint
        Channel spin. Default = None.
    """

    def __init__(self, l:int, j:halfint, s:halfint=None):
        """
        Creates a SpinGroup object based on the quantum numbers for the reaction.

        Parameters:
        ----------
        l :: int
            Orbital angular momentum.
        j :: halfint
            Total angular momentum.
        s :: halfint
            Channel spin. Default = None.
        """

        self.L = int(l)
        
        if type(j) == halfint:      self.J = j
        else:                       self.J = halfint(j)

        if s is not None:   self.S = halfint(s)
        else:               self.S = None

    def __format__(self, spec:str):
        if (spec is None) or (spec == 'jpi'):
            if self.L % 2:  return f'{self.J}-'
            else:           return f'{self.J}+'
        elif spec == 'lj':
            return f'({self.L},{self.J})'
        elif spec == 'ljs':
            return f'({self.L},{self.J},{self.S})'
        else:
            raise ValueError('Unknown format specifier.')

    def __repr__(self):
        return f'{self:ljs}'
    
    def __str__(self):
        return f'{self:jpi}'
    
    def g(self, spin_target:halfint, spin_proj:halfint):
        'Statistical spin factor'
        return (2*self.J+1) / ((2*spin_target+1) * (2*spin_proj+1))

class SpinGroups:
    """
    A class that contains a list of spingroups, or generates spingroups from the target and
    projectile spins.

    Attributes:
    ----------
    SGs         :: list [Spingroup]
        List of spingroups.
    l_max       :: int
        Maximum orbital angular momentum. Default = None.
    spin_target :: halfint
        Intrinsic spin of the target particle. Default = None.
    spin_proj   :: halfint
        Intrinsic spin of the projectile particle. Default = None.
    """

    def __init__(self, SGs:list, l_max:int=None,
                 spin_target:halfint=None, spin_proj:halfint=None):
        """
        Makes a SpinGroups object that holds information on the possible spingroup states.

        Parameters:
        ----------
        SGs :: list [Spingroup]
            List of possible spingroups.
        l_max       :: int
            Maximum orbital angular momentum. Default = None.
        spin_target :: halfint
            Intrinsic spin of the target particle. Default = None.
        spin_proj   :: halfint
            Intrinsic spin of the projectile particle. Default = None.
        """
        self.SGs   = SGs
        self.l_max = l_max
        # Target spin:
        if spin_target is None:     self.spin_target = None
        else:                       self.spin_target = halfint(spin_target)
        # Projectile spin:
        if spin_proj is None:       self.spin_proj = None
        else:                       self.spin_proj = halfint(spin_proj)

    @classmethod
    def make(cls, Ls:list, Js:list, Ss:list=None):
        """
        Generates spingroups from the provided "Ls", "Js" and "Ss" quantities.

        Parameters:
        ----------
        Ls :: list [int]
            The ordered list of orbital angular momentums numbers.
        Js :: list [halfint]
            The ordered list of total angular momentum numbers.
        Ss :: list [halfint]
            The ordered list of channel spin numbers.

        Returns:
        -------
        spingroups :: SpinGroups
            The generated spingroups.
        """
        
        if (Ss != None) and (len(Ls) == len(Js) == len(Ss)):
            sgs = [SpinGroup(Ls[g], Js[g], Ss[g]) for g in range(len(Ls))]
        elif (len(Ls) == len(Js)) and (Ss == None):
            sgs = [SpinGroup(Ls[g], Js[g], None) for g in range(len(Ls))]
        else:
            raise ValueError('The number of "L", "J", and "S" values for spin-groups are not equal.')
        l_max = max(*Ls, 0) # I need to add another l-value in case len(Ls)=1; otherwise, max(*Ls) will break
        return cls(sgs, l_max=l_max)

    @classmethod
    def find(cls, spin_target:halfint, spin_proj:halfint=1/2, l_max:int=1):
        """
        Finds all of the valid spingroups with "l" less than or equal to "l_max".

        Parameters:
        ----------
        spin_target :: halfint
            The quantum spin number for the target nuclei.
        spin_proj   :: halfint
            The quantum spin number for the projectile nuclei.
        l_max       :: int
            The maximum orbital angular momentum number generated.

        Returns:
        -------
        spingroups  :: SpinGroups
            The generated spingroups.
        """

        l_max = int(l_max)
        sgs   = []
        for l in range(l_max+1):
            for s2 in range(abs(2*(spin_target - spin_proj)), 2*(spin_target + spin_proj+1), 2):
                for j2 in range(abs(s2 - 2*l),s2 + 2*l+2, 2):
                    sgs.append(SpinGroup(l, j2/2, s2/2))
        return cls(sgs, l_max, spin_target=spin_target, spin_proj=spin_proj)

    @property
    def L(self):
        'Orbital angular momentum number'
        return [sg.L for sg in self.SGs]
    @property
    def J(self):
        'Total angular momentum number'
        return [sg.J for sg in self.SGs]
    @property
    def S(self):
        'Channel spin number'
        return [sg.S for sg in self.SGs]
    @property
    def g(self):
        'Statistical spin factor'
        if (self.spin_target is None) or (self.spin_proj is None):
            raise ValueError('Target spin and/or projectile spin were not provided. The statistical spin factors, "g" cannot be calculated.')
        return [sg.g(self.spin_target, self.spin_proj) for sg in self.SGs]
    
    def __len__(self):
        return len(self.SGs)
    @property
    def num_sgs(self):
        'The number of spingroups'
        return len(self.SGs)
    
    def __getitem__(self, indices):
        if hasattr(indices, '__iter__'):
            return self.__class__(self.SGs[indices], self.l_max, self.spin_target, self.spin_proj)
        else:
            return self.SGs[indices]
        
    def id(self, spingroup:SpinGroup):
        """
        Returns an integer index ID if provided a spingroup. If an integer id is provided, the id
        is passed.
        """

        if spingroup in ('false', 'False'):
            return self.num_sgs
        elif type(spingroup) == SpinGroup:
            for g in range(self.num_sgs):
                if spingroup == self.SGs[g]:
                    return g
            raise ValueError(f'The provided spingroup, {spingroup}, does not match any of the recorded spingroups.')
        elif type(spingroup) == int:
            if spingroup > self.num_sgs:
                raise ValueError(f'The provided spingroup id, {spingroup}, is above the number of spingroups, {self.num_sgs}.')
            return spingroup
        else:
            raise TypeError(f'The provided spingroup, {spingroup}, is not an integer ID nor is it a "SpinGroup" object.')