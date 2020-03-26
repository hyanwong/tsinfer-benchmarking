#!/usr/bin/env python3

import setuptools

with open('README.md') as f:
    long_description = ''.join(f.readlines())


setuptools.setup(
    name='tsinfer-benchmarking',
    version='0.1',
    packages=setuptools.find_packages(exclude=['tests']),
    include_package_data=True,

    description='Benchmarking for tsinger',
    long_description=long_description,
    author='Yan Wong',
    author_email='yan.wong@bdi.ox.ac.uk',
    url='https://github.com/hyanwong/tsinfer-benchmarking',

    # All versions are fixed just for case. Once in while try to check for new versions.
    install_requires=[
        'pip3-multiple-versions',
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    zip_safe=False,
)