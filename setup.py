# -*- coding: utf-8 -*-
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='igorwriter',
    version='0.4.1',
    description='Write IGOR binary (.ibw) or text (.itx) files from numpy array',
    long_description=long_description,
    url='https://github.com/t-onoz/igorwriter',
    author='t-onoz',
    license='MIT',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],

    packages=find_packages(),
    package_data={
        'igorwriter': ['builtins/*.txt']
    },

    install_requires=['numpy'],  # Optional
)
