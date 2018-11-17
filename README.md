# flexCALC

This project is a part of the larger X-ray tomographic reconstruction toolbox comprised of flexDATA, flexTOMO and flexCALC.
flexCALC contains various routines useful with tomographic reconstructions but not directly reconstruction algorithms. These routines include data pre- and post-processing tools, simulation of spectral data, and batch-processing of large number of datasets. 

## Getting Started

Before installing flexCALC, please download and install [flexDATA](https://github.com/cicwi/flexdata) and [flexTOMO](https://github.com/cicwi/flextomo). Once installation of flexTOMO is complete, one can install flexCALC from the source code or using [Anaconda](https://www.anaconda.com/download/). 

### Installing with conda

Simply install with:
```
TODO
```

### Installing from source

To install flexCALC, clone this GitHub project. Go to the cloned directory and run PIP installer:
```
git clone https://github.com/cicwi/flexcalc.git
cd flexcalc
pip install -e .
```

## Running the examples

To learn about the functionality of the package check out our examples folder. Examples are separated into blocks that are best to run in Spyder environment step-by-step.

## Modules

flexCALC is comprised of several modules:

* process:     pre- and post-processing routines. For instance: volume registration, rings removal, stl model generator etc. 
* resolution:  apply Contrast Transfer Functions (CTF) to the data to mimic resolution limits or phase contrast  
* spectrum: use NIST data to simulate spectral characteristics of materials and to simulate polychromatic projections 
* pipe: define a batch processing pipeline and push multiple datasets through it

Typical code:
```
# Import:
from flextomo import project
from flexcalc import process

# Read data and apply beam-hardening:
proj, meta = process.process_flex(path, sample = 2, skip = 2)
proj = process.equivalent_density(proj, meta, energy, spectrum, compound = 'Al', density = 2.7) 

# Reconstruct:
vol = project.init_volume(proj)
project.FDK(proj, vol, meta['geometry'])
```

## Authors and contributors

* **Alexander Kostenko** - *Initial work*

See also the list of [contributors](https://github.com/cicwi/flexdata/contributors) who participated in this project.

## How to contribute

Contributions are always welcome. Please submit pull requests against the `develop` branch.

If you have any issues, questions, or remarks, then please open an issue on GitHub.

## License

This project is licensed under the GNU GENERAL PUBLIC License - see the [LICENSE.md](LICENSE.md) file for details
