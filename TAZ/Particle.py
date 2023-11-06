from . import halfint

__doc__ = """
This file stores the "Particle" class. The "Particle" class contains all relevent information to a
specific particle. Objects of this class are used when defining a reaction's properties in
"MeanParameters". The "Neutron" object has already been defined for convenience.
"""

# =================================================================================================
#    Particle:
# =================================================================================================

mass_neutron = 1.00866491588 # amu

class Particle:
    """
    Attributes:
    ----------
    Z      :: int
        Atomic number
    A      :: int
        Atomic mass
    I      :: halfint
        Particle spin
    mass   :: float
        Nuclei mass in amu
    AWRI   :: float
        Nuclei mass divided by neutron mass
    radius :: float
        Nuclear mean square radius in femtometers
    name   :: str
        Name of the particle
    """

    def __init__(self, Z:int=None, A:int=None, I:halfint=None,
                 mass:float=None, AWRI:float=None,
                 radius:float=None, name:str=None):
        """
        Initialize a Particle object.
        """
        # Atomic Number:
        if Z is not None:   self.Z = int(Z)
        else:               self.Z = None
        # Atomic Mass:
        if A is not None:   self.A = int(A)
        else:               self.A = None
        # Isotope Spin:
        if I is not None:   self.I = halfint(I)
        else:               self.I = None
        # Mass: (amu)
        if mass is not None:    self.mass = float(mass)
        elif AWRI is not None:  self.mass = float(AWRI)*mass_neutron
        elif A is not None:     self.mass = float(A)
        else:                   self.mass = None
        # AWRI:
        if self.mass is not None:   self.AWRI = mass / mass_neutron
        else:                       self.AWRI = None
        # Nuclear Radius: (fm)
        if radius is not None:  self.radius = float(radius)
        elif A is not None:     self.radius = 1.23 * self.A**(1/3)
        # Particle Name:
        if name is not None:
            self.name = str(name)
        elif (A is not None) and (Z is not None):
            self.name = str(Z*1000+A)
        else:
            self.name = '???'

    def __repr__(self):
        txt  = f'Particle       = {self.name}\n'
        txt += f'Atomic Number  = {self.Z}\n'
        txt += f'Atomic Mass    = {self.A}\n'
        txt += f'Nuclear Spin   = {self.I}\n'
        txt += f'Mass           = {self.mass} (amu)\n'
        txt += f'Nuclear Radius = {self.radius} (fm)\n'
        return txt
    def __str__(self):
        return self.name
    
Neutron = Particle(I=0.5, Z=0, A=1, mass=mass_neutron, radius=0.8, name='neutron')