# PyMRItools
Toolbox for all things MRI with python. Sequence Programming, Reconstruction, Simulation, Analysis.


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
