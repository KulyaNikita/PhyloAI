##########
# setup.py
##########

from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'QuartetPatternCounterModule',
        ['QP_python_bindings.cpp', 'QuartetPattern-counter-core.cpp'],  # Both files!
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3', '-march=native', '-flto'],
    ),
]

setup(
    name='QuartetPatternCounterModule',
    version='1.5',
    ext_modules=ext_modules,
    zip_safe=False,
)

