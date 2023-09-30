from numpy import uint8

class halfint:
    """
    Data type for half-integers to represent spins.
    """

    def __init__(self, value):
        if (2*value) % 1:
            raise ValueError(f'The number, {value}, is not a half-integer.')
        self.__2x_value = uint8(2*value)
    
    @property
    def parity(self):   return '+' if self.__2x_value >= 0 else '-'
    @property
    def value(self):    return self.__2x_value / 2

    def __str__(self):
        if self.__2x_value % 2 == 0:
            return '{:2}'.format(self.__2x_value//2)
        else:
            return '{:2}/2'.format(self.__2x_value)

    # Arithmetic:
    def __eq__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value == other.value
        else:
            return self.value == other
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
    def __ne__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value != other.value
        else:
            return self.value != other
    def __add__(self, other):
        if type(other) == self.__class__:
            return self.__class__(self.value + other.value)
        else:
            return self.value + other
    def __radd__(self, other):
        return self.value + other
    def __sub__(self, other):
        if type(other) == self.__class__:
            return self.__class__(self.value - other.value)
        else:
            return self.value - other
    def __rsub__(self, other):
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
    reaction. The quantum number, "s", can also be given optionally.
    """

    def __init__(self, l:int, j:halfint, s:halfint=None):
        self.L = uint8(l)
        
        if type(j) == halfint:      self.J = j
        else:                       self.J = halfint(j)

        if s is not None:
            if type(s) == halfint:      self.S = s
            else:                       self.S = halfint(s)
            self.S = halfint(s)
        else:
            self.S = None

    def name(self, option:str='j^pi') -> str:
        """
        Returns the string written form of the spingroup.

        ...
        """

        if   option.lower() == 'j^pi':
            if self.L % 2:  return f'{self.J}-'
            else:           return f'{self.J}+'
        elif option.lower() == '(l,j)':
            return f'({self.L},{self.J})'
        elif option.lower() == '(l,j,s)':
            return f'({self.L},{self.J},{self.S})'
        else:
            raise ValueError('Unknown option.')

    def __str__(self):
        return self.name('j^pi')
    
    def g(self, spin_target, spin_proj):
        'Statistical spin factor'
        return (2*self.J+1) / ((2*spin_target+1) * (2*spin_proj+1))

class SpinGroups:
    """
    A class that contains a list of spingroups, or generates spingroups from the target and
    projectile spins.

    ...
    """

    def __init__(self, SGs:list, l_max:int=None, spin_target=None, spin_proj=None):
        self.SGs   = SGs
        self.l_max = l_max

        # Target spin:
        if spin_target is None:
            self.spin_target = None
        elif type(spin_target) == halfint:
            self.spin_target = spin_target
        else:
            self.spin_target = halfint(spin_target)

        # Projectile spin:
        if spin_proj is None:
            self.spin_proj = None
        elif type(spin_proj) == halfint:
            self.spin_proj = spin_proj
        else:
            self.spin_proj = halfint(spin_proj)

    @classmethod
    def make(cls, Ls, Js, Ss=None):
        """
        Generates spingroups from the provided "Ls", "Js" and "Ss" quantities.
        """
        
        if (Ss != None) and (len(Ls) == len(Js) == len(Ss)):
            sgs = [SpinGroup(Ls[g], Js[g], Ss[g]) for g in range(len(Ls))]
        elif (len(Ls) == len(Js)) and (Ss == None):
            sgs = [SpinGroup(Ls[g], Js[g], None) for g in range(len(Ls))]
        else:
            raise ValueError('The number of "L", "J", and "S" values for spin-groups are not equal.')
        l_max = max(*Ls, 0) # I need to add another l-value in case len(Ls)=1; otherwise, max(*Ls) will break
        return cls(sgs, l_max)

    @classmethod
    def find(cls, spin_target, spin_proj=1/2, l_max:int=1):
        """
        Finds all of the valid spingroups with "l" less than or equal to "l_max".
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
        return [sg.L for sg in self.SGs]
    @property
    def J(self):
        return [sg.J for sg in self.SGs]
    @property
    def S(self):
        return [sg.S for sg in self.SGs]
    @property
    def num_sgs(self):
        return len(self.SGs)
    @property
    def g(self):
        'Statistical spin factor'
        if (self.spin_target is None) or (self.spin_proj is None):
            raise ValueError('Target spin and/or projectile spin were not provided. The statistical spin factors, "g" cannot be calculated.')
        return [sg.g(self.spin_target, self.spin_proj) for sg in self.SGs]