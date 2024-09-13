from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

draw_requirements = [
    "networkx",
    "pygraphviz"
]

mesh_requirements = [
    "numpy-stl"
]

dev_requirements = [
    "sphinx",
    "sphinx_rtd_theme",
    "myst-parser"
]

setup(
    author="Alex Kostenko",
    description="CT data pre- and post-processing tools, simulation of spectral data, and batch-processing of large number of datasets",
    install_requires=[
        "numpy",
        "tqdm",
        "simpleitk",
        "scipy",
        "scikit-image",
        "flexdata",
        "flextomo"
    ],
    license="GNU General Public License v3",
    long_description=readme,
    long_description_content_type="text/markdown",
    name="flexcalc",
    packages=find_packages(include=["flexcalc"]),
    extras_require={
        "dev": dev_requirements,
        "draw": draw_requirements,
        "mesh": mesh_requirements
    },
    url="https://github.com/cicwi/flexcalc",
    version="0.1.0",
)
