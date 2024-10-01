# PyMRItools
Toolbox for all things MRI with python. Sequence Programming, Reconstruction, Simulation, Processing, Analysis.


## Install

## Usage

## Contribution & best practise
1) IO
   - *.nii* in *.nii* out for all tools
   - *ismrmrd* file format for raw data if nii not feasible
   - Need additional rule for recon steps? $\rightarrow$ *.nii* might not be feasible for uncombined data, use *.pt* and store tensors?
2) Config
    - we have a central configuration module. This way we can maintain and access configuration dataclasses throughout different modules.
    - i.e. store values upon sequence programming and sequence creation in a class and access class at recon or raw data collection


## Test / Examples

- Theoretically those should be the CLI scripts installed:
- set working directory to top level (PyMRItools)
1) `emc_simulation`:
   - PyMRItools/pymritools/simulation/emc/core/simulate.py -c ./examples/simulation/emc_settings.json
2) `processing_denoise_mppca`:
   - PyMRItools/pymritools/processing/denoising/mppca/denoise.py -c ./examples/processing/denoising/config.json
3) `processing_unring`:
   - PyMRItools/pymritools/processing/unringing/gibbs_unr.py -c ./examples/processing/unringing/config.json
4) `modeling_grid_search`:
   - PyMRItools/pymritools/modeling/dictionary/grid_search.py -c ./examples/modeling/emc/config.json