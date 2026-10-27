"""Build QPC for the Python interpreter running this installer."""
import sys
from setuptools import Extension, setup
import pybind11

flags = ['-std=c++17', '-O3']
thread_flags = ['-pthread'] if sys.platform != 'win32' else []
if sys.platform == 'win32':
    flags = ['/std:c++17', '/O2']

setup(
    name='QuartetPatternCounterModule',
    version='1.6.1',
    description='Batched quartet site-pattern frequencies and effective lengths',
    license='BSD-3-Clause',
    install_requires=['numpy>=1.21'],
    ext_modules=[Extension(
        'QuartetPatternCounterModule',
        ['QP_python_bindings.cpp', 'QuartetPattern-counter-core.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=flags + thread_flags,
        extra_link_args=thread_flags,
    )],
    zip_safe=False,
)
