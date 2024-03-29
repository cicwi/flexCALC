#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.md') as history_file:
    history = history_file.read()

requirements = [
    "numpy",
    "tqdm",
    "simpleitk",
    "scipy",	
    "scikit-image",
    "flexdata",
    "flextomo"]

setup_requirements = [ ]

test_requirements = [ ]

draw_requirements = [
    "networkx",
    "pygraphviz",
    ]

mesh_requirements = [
    "numpy-stl",
    ]

dev_requirements = [
    'autopep8',
    'rope',
    'jedi',
    'flake8',
    'importmagic',
    'autopep8',
    'black',
    'yapf',
    'snakeviz',
    # Documentation
    'sphinx',
    'sphinx_rtd_theme',
    'recommonmark',
    # Other
    'bumpversion',
    'watchdog',
    'coverage',
    
    ]

setup(
    author="Alex Kostenko",
    author_email='a.kostenko@cwi.nl',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="ASTRA-based cone beam reconstruction routines",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='flexcalc',
    name='flexcalc',
    packages=find_packages(include=['flexcalc']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require={ 'dev': dev_requirements, "draw": draw_requirements, "mesh": mesh_requirements },
    url='https://github.com/cicwi/flexcalc',
    version='0.1.0',
    zip_safe=False,
)
