# flexCALC

This project is a part of the larger X-ray tomographic reconstruction toolbox comprised of [flexDATA](https://github.com/cicwi/flexDATA), [flexTOMO](https://github.com/cicwi/flexTOMO) and [flexCALC](https://github.com/cicwi/flexCALC).
flexCALC contains various routines useful with tomographic reconstructions but not directly reconstruction algorithms. These routines include data pre- and post-processing tools, simulation of spectral data, and batch-processing of large number of datasets.

## Getting Started

We recommend that the user installs [conda package manager](https://docs.anaconda.com/miniconda/) for Python 3.

### Installing with conda

`conda install flexcalc -c cicwi -c astra-toolbox -c nvidia`

### Installing with pip

`pip install flexcalc`

### Installing from source

```bash
git clone https://github.com/cicwi/flexcalc.git
cd flexcalc
pip install -e .
```

## Running the examples

To learn about the functionality of the package check out our `examples/` folder. Examples are separated into blocks that are best to run in VS Code / Spyder environment step-by-step.

## Modules

flexCALC is comprised of several modules:

* `flexcalc.process`: Pre- and post-processing routines. For instance: volume registration, rings removal etc.
* `flexcalc.analyze`: Utilities for data analysis.
* `flexcalc.batch`: Define a batch processing pipeline and push multiple datasets through it.

Typical usage:

```python
# Import:
from flextomo import project
from flexcalc import process

# Read data and apply beam-hardening:
proj, geom = process.process_flex(path)
proj = process.equivalent_density(proj, geom, energy, spec, compound = 'Al', density = 2.7)

# Align the rotation centre:
process.optimize_rotation_center(proj, geom, subscale = 2)

# Reconstruct:
vol = project.init_volume(proj)
project.FDK(proj, vol, meta['geometry'])
```

## Authors and contributors

* **Alexander Kostenko** - *Initial work*
* **Willem Jan Palenstijn** - *Packaging, installation and maintenance*
* **Alexander Skorikov** - *Packaging, installation and maintenance*

## How to contribute

Contributions are always welcome. If you have any issues, questions, or remarks, then please open an issue on GitHub.

## License

This project is licensed under the GNU GENERAL PUBLIC License - see the [LICENSE.md](LICENSE.md) file for details
