# Installation

## Pip

You can install the `PyMRItools` package directly from GitHub using Python 3.10 and `pip`.
We advise
[working in a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments)
using `venv` to keep your system's Python installation
untouched.

```shell
# Create a new virtual environment in the .venv directory
python3.10 -m venv .venv

# Install PyMRItools directory from GitHub
pip install git+https://github.com/schmidt-jo/PyMRItools
```

## Conda

For the development of the package, we prefer `conda` with a dedicated environment
where PyMRItools are installed as an _editable_ package.
First, clone the PyMRItools repository or your clone of it:

```shell
git clone https://github.com/schmidt-jo/PyMRItools.git
cd PyMRItools
```

Now create a conda environment from the `environment.yml` provided in the project directory.
We prefer
[conda-forge](https://conda-forge.org/download/)
and from within the PyMRItools directory,
the `mri_tools_env` environment can be created with (we use `mamba`, but `conda` is also fine):

```shell
mamba env create -f environment.yml
```

The next step is to activate the environment and install PyMRItools in edit mode.
Since all package dependencies are already available in the conda environment,
we include the `--no-deps` option:

```shell
mamba activate mri_tools_env
pip install --no-deps -e .
```

Now you can use PyMRItools in this virtual environment, and if you work on the code,
changes are directly reflected.

## Building an Apptainer for HPC