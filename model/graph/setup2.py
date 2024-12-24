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