from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

draw_requirements = [
    'networkx',
    'pygraphviz'
]

mesh_requirements = [
    'numpy-stl'
]

dev_requirements = [
    'sphinx',
    'sphinx_rtd_theme',
    'myst-parser'
]

setup(
    name='flexcalc',
    version='1.0.0',
    description='CT data pre- and post-processing tools, simulation of spectral data, and batch-processing of large number of datasets',
    url='https://github.com/cicwi/flexcalc',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Alex Kostenko',
    license='GNU General Public License v3',
    packages=find_packages(include=['flexcalc']),
    install_requires=[
        'numpy',
        'tqdm',
        'simpleitk',
        'scipy',
        'scikit-image',
        'flexdata',
        'flextomo'
    ],
    extras_require={
        'dev': dev_requirements,
        'draw': draw_requirements,
        'mesh': mesh_requirements
    }
)
