import numpy as np
from disk_density_profiles import *

   
def spherical_potential_keplerian(r,soft=0.01):
    return SplineProfile(r,soft)


