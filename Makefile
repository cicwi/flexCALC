.PHONY: clean clean-test clean-pyc clean-build docs help install_dev
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

docs/.nojekyll:
	touch docs/.nojekyll

docs: docs/.nojekyll install_dev ## generate Sphinx HTML documentation, including API docs
	make --directory doc_sources clean
	make --directory doc_sources html
	$(BROWSER) docs/index.html

install: clean ## install the package to the active Python's site-packages
	pip install .

install_dev: ## install in editable mode
	pip install -e .[dev]

conda_package: ## build Conda package
	conda build conda/ -c cicwi -c astra-toolbox -c nvidia -c conda-forge

pypi_wheels: ## build wheel package for PyPI
	python -m build --wheel
