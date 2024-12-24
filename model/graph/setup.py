from setuptools import setup
from Cython.Build import cythonize

setup(
      ext_modules=cythonize("walks.pyx"),
  )


from setuptools import setup, Extension
import numpy as np

setup(
    # ... other setup parameters ...
    ext_modules=[
        Extension(
            'walks',
            ['walks.c'],
            include_dirs=[np.get_include()]  # Add this line
        )
    ]
)

# python setup.py build_ext --inplace