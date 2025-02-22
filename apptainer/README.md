# Building PyMRItools Apptainer Container

This directory contains the necessary files to build an Apptainer container for PyMRItools with ROCm support.
The container provides a consistent environment for running PyMRItools on the MPCDF Viper-GPU cluster with all required dependencies.

## Files Description

- `viper_apptainer.def`: The Apptainer definition file that specifies how to build the container
- `viper_environment.yml`: The Conda environment specification file that defines the Python dependencies

## Prerequisites

1. Apptainer (formerly Singularity) installed on your system
2. Sufficient disk space (approximately 10GB)
3. Internet connection to download packages

## Building the Container

1. Navigate to the container directory:
   ```bash
   cd container
   ```

2. Build the Apptainer container:
   ```bash
   sudo apptainer build pymritools.sif viper_apptainer.def
   ```

The build process will:
- Use ROCm Ubuntu 24.04 as the base image
- Install Miniforge3 (minimal Conda distribution)
- Create a conda environment named "mri_tools_env"
- Install all dependencies specified in `viper_environment.yml`
- Install additional pip packages including:
  - triton
  - autodmri
  - twixtools
  - pypulseq
  - PyTorch with ROCm 6.2.4 support

## Using the Container

Once built, you can run commands inside the container using:

```bash
apptainer exec pymritools.sif <command>
```

or 

```bash
apptainer shell pymritools.sif
```

The container automatically activates the conda environment on execution.
