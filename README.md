# PyMRItools

Toolbox for MRI sequence programming, reconstruction, simulation, processing, and analysis using Python.

## Installation

### Pip

You can install the `PyMRItools` package directly from GitHub using Python 3.10 and `pip`.
We advise
[working in a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments)
using `venv` to keep your system's Python installation
untouched.

```shell
# Create a new virtual environment in the .venv directory
python3.10 -m venv .venv

# Install PyMRItools directory from GitHub
pip install git+https://https://github.com/schmidt-jo/PyMRItools
```

### Conda

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

## Usage

For common tools, we provide commands that are available once the virtual environment is activated.
The command line tools are based on some of the .py scripts in the repository.
Both, scripts and commands, can parse a set of arguments available via the `--help` option.

### EMC Simulation

**Description:** Bloch equation simulation derived from the EMC method by [Ben-Eliezer et al. 2015](https://doi.org/10.1002/mrm.25156).
The function creates a dictionary. Value ranges to simulate for can be given in the settings.
The simulation needs exact sequence RF and slice selective gradient events (specified in a emc_params file) that need to be obtained from sequence simulations.

**Commandline Tool:** `emc_simulation`

**Python Example:**

```shell
python3.10 pymritools/simulation/emc/core/simulate.py \
    -c ./examples/simulation/emc_settings.json
```

### Dictionary Grid Search

**Description:** Dictionary matching method for e.g. EMC dictionary (or any database that can be converted to the DB object in `pymritools.config.database`).
The algorithm takes input data and a dictionary and uses brute force to determine the best-matching pattern to the data in the dictionary.
Specifically for the EMC / R2 estimation usecase, one can provide input B1 information to regularize the matching method by first selecting the sub-dictionary corresponding to the voxel B1 provided.

**Commandline Tool:** `modeling_dictionary_grid_search`

**Python Example:**

```shell
python3.10 pymritools/modeling/dictionary/grid_search.py \
    -c ./examples/modeling/emc/config.json
```

### Mono-Exponential Fitting

**Description:** Simple mono-exponential least squares fitting module.
According to $S(t) = S_0 \  e^{(-R t)}$ the function saves the optimal solutions S0 and R as maps.

**Commandline Tool:** `modeling_mono_exponential_fit`

**Python Example:**

```shell
python3.10 pymritools/modeling/decay/mexp/mexp.py \
    -c ./examples/modeling/mexp/config.json
```

### Denoising MPPCA

**Description:** Denoising algorithm based on the method by [Does et al. 2019](https://doi.org/10.1002/mrm.27658) employing a MP-PCA using spatial redundancy in relaxation data.
Additionally, a magnitude bias correction is implemented ([Manjon et al. 2015](http://dx.doi.org/10.1016/j.media.2015.01.004)),
which is derived from a noise property estimation from background voxels.
The estimation is following [StJean et al. 2020](https://doi.org/10.1016/j.media.2020.101758) and using the autodmri package provided by StJean ([github - autodmri](https://github.com/samuelstjean/autodmri?tab=readme-ov-file)) .
This noise estimation assumes stationary noise properties per imaging slice.

**Commandline Tool:** `processing_denoise_mppca`

**Python Example:**

```shell
python3.10 pymritools/processing/denoising/mppca/denoise.py \
    -c ./examples/processing/denoising/config.json
```

### Gibbs Unringing

**Description:** Gibbs unringing algorithm based on the method by [Kellner et al. 2016](https://doi.org/10.1002/mrm.26054).

**Commandline Tool:** `processing_unring`

**Python Example:**

```shell
python3.10 pymritools/processing/unringing/gibbs_unr.py \
    -c ./examples/processing/unringing/config.json
```
