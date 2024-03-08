import warnings

from TAZ.DataClasses import HalfInt

__doc__ = """
This file stores the "Particle" class. The "Particle" class contains all relevent information to a
specific particle. Objects of this class are used when defining a reaction's properties in
"Reaction" objects. The "Neutron" and "Proton" objects has already been defined for convenience.
"""

# =================================================================================================
#    Particle:
# =================================================================================================

MASS_NEUTRON = 1.00866491588  # amu  (source: ENDF-6 manual Table H.2)
MASS_PROTON  = 1.007276466621 # amu  (source: ENDF-6 manual Table H.2)

class Particle:
    """
    Particle is a class that contains information about a particular particle, including atomic
    number, atomic mass number, nuclei mass, nuclear radius, and the name of the particle.
    """

    def __init__(self, Z:int, A:int, I:HalfInt, mass:float,
                 radius:float=None, name:str=None):
        """
        Initialize a Particle object.

        Attributes
        ----------
        Z      : int
            Atomic number
        A      : int
            Atomic mass number
        I      : HalfInt
            Particle spin
        mass   : float
            Nuclei mass in atomic mass units (amu)
        AWRI   : float
            Nuclei mass divided by neutron mass
        radius : float
            Nuclear mean square radius in 1e-12 centimeters
        name   : str
            Name of the particle
        """
        # Atomic Number:
        self._Z = int(Z)
        # Atomic Mass:
        if A < Z:
            raise ValueError(f'The atomic mass number, {A}, cannot be less than the atomic number, {Z}.')
        self._A = int(A)
        # Isotope Spin:
        self._I = HalfInt(I)
        # Mass: (amu)
        if   mass > 300.0:     warnings.warn(f'The particle mass, {mass} amu, is quite high. Make sure it is in units of amu.', UserWarning)
        elif mass <   1.0:     warnings.warn(f'The particle mass, {mass} amu, is quite low. Make sure it is in units of amu.', UserWarning)
        self._mass = float(mass)
        # Nuclear Radius: (fm)
        if radius is not None:
            if   radius > 1.00:     warnings.warn(f'The nuclear radius, {radius} 1e-12 cm, is quite high. Make sure it is in units of square-root barns or 1e-12 cm.', UserWarning)
            elif radius < 0.05:     warnings.warn(f'The nuclear radius, {radius} 1e-12 cm, is quite low. Make sure it is in units of square-root barns or 1e-12 cm.', UserWarning)
            self._radius = float(radius)
        else:
            self._radius = 0.123 * self.A**(1/3) # 1e-12 cm
        # Particle Name:
        if name is not None:    self._name = str(name)
        else:                   self._name = str(self.Z*1000+self.A) # MCNP ID for the isotope.

    @property
    def Z(self):
        'Atomic number'
        return self._Z
    @property
    def A(self):
        'Atomic mass number'
        return self._A
    @property
    def I(self):
        'Particle spin'
        return self._I
    @property
    def mass(self):
        'Nuclei mass in atomic mass units (amu)'
        return self._mass
    @property
    def radius(self):
        'Nuclear mean square radius in 1e-12 cm'
        return self._radius
    @property
    def name(self):
        'Name of the particle'
        return self._name
    @property
    def N(self):
        'Number of Neutrons'
        return self._A - self._Z
    @property
    def AWRI(self):
        'Nuclei mass divided by neutron mass'
        return self._mass / MASS_NEUTRON

    def __repr__(self):
        txt  = f'Particle       = {self._name}\n'
        txt += f'Atomic Number  = {self._Z}\n'
        txt += f'Atomic Mass    = {self._A}\n'
        txt += f'Nuclear Spin   = {self._I}\n'
        txt += f'Mass           = {self._mass:.7f} (amu)\n'
        txt += f'Nuclear Radius = {self._radius:.7f} (√b)\n'
        return txt
    
    def __str__(self):
        return self.name
    
Neutron = Particle(Z=0 , A=1  , I=0.5, mass=MASS_NEUTRON, radius=0.8  , name='neutron')
Proton  = Particle(Z=1 , A=1  , I=0.5, mass=MASS_PROTON , radius=0.833, name='proton' )
Ta181   = Particle(Z=73, A=181, I=3.5, mass=180.94803   , radius=None , name='Ta181'  )