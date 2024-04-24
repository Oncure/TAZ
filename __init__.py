import os
this_path = os.path.realpath(__file__)
TAZ_directory = os.path.dirname(this_path)

import sys
sys.path.append(TAZ_directory)

from .TAZ import *