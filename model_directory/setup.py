# -*- coding: utf-8 -*-
"""

@author: Lewis Dean
"""
import numpy

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("*.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False
)