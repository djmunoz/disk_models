from __future__ import print_function
from distutils.core import setup
import numpy as np
import os
import sys


setup(
    name='disk_models',
    version="0.0.1",
    author='Diego J. Munoz',
    author_email = 'diego.munoz.anguita@gmail.com',
    url='https://github.com/',
    packages=['disk_models','disk_models.disk_hdf5'],
    description='Setup three-dimensional models of accretion disks in hydrostatic and centrifugal equilibrium',
    install_requires = ['numpy','scipy','tables'],
)
