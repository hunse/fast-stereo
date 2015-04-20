"""
To build:
    python setup.py build_ext --inplace


The idea of scanning for Cython extension modules was adapted from
https://github.com/cython/cython/wiki/PackageHierarchy
"""
import os
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np


ext_modules = []
ext_modules.append(Extension(
    "bp.bp",
    ["bp/bp.pyx", "bp/stereo.cpp"],
    include_dirs=["bp", "/opt/opencv/include", np.get_include()],
    library_dirs=["/opt/opencv/lib"],
    libraries=["opencv_core", "opencv_imgproc"],
    language="c++",
    extra_compile_args=["-w", "-O3"]))

setup(
    name="bp",
    ext_modules=ext_modules,
    cmdclass = {'build_ext': build_ext},
)
