from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

name = "solver1"
sources = ["solver.pyx"]
dirs = ['./']

# mod = cythonize([Extension(name, sources=sources, include_dirs=dirs, extra_compile_args=['-std=c99'])])
# setup(ext_modules=mod, annotate=True)  # annotate=True?

setup(
    ext_modules=cythonize(Extension(name, 
                                    sources,  
                                    extra_compile_args=['-std=c99']), annotate=True))
