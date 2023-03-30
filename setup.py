# Author: Taisei Saida <saida.taisei.tj@alumni.tsukuba.ac.jp>
# Copyright (c) 2023 Taisei Saida
# License: MIT

from setuptools import setup
# import TL_GPRSM

DESCRIPTION = "TL-GPRSM: Tlansfer Learning Gaussian Process Regression Surrogate Model"
NAME = 'TL_GPRSM'
AUTHOR = 'Taisei Saida'
AUTHOR_EMAIL = 'saida.taisei.tj@alumni.tsukuba.ac.jp'
URL = 'https://github.com/SaidaTaisei/TL-GPRSM'
LICENSE = 'MIT License'
DOWNLOAD_URL = 'https://github.com/SaidaTaisei/TL-GPRSM'
VERSION = 1.0.3#TL_GPRSM.__version__
PYTHON_REQUIRES = ">=3.8"

INSTALL_REQUIRES = [
    'numpy>=1.7,<=1.23.0',
    'GPy>=1.10.0',
    'h5py>=3.1.0',
]

PACKAGES = [
    'TL_GPRSM',
    'TL_GPRSM.models',
    'TL_GPRSM.utils',
]

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Topic :: Scientific/Engineering',
]

with open('README.md', 'r') as fp:
    readme = fp.read()

description = readme

setup(name=NAME,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=AUTHOR,
      maintainer_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      long_description=description,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      python_requires=PYTHON_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      packages=PACKAGES,
      classifiers=CLASSIFIERS
    )
