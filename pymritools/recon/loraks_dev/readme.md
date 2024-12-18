# DEV considerations

- [x] Fix a requirement to the input shape: `shape = (m, ne, nc, nz, ny, nx)`
- [ ] How to handle all of our torch functions that don't need a device? Fix either this or the speed test function
- [ ] Clean up: 
  - Remove code that is no longer used and that we potentially won't use in the future
  - Move main functions from package files to tests folder
- [ ] SVD
  - [ ] Jochen check all `m.T` if a `m.H` is required 
  - check unified interface of functions
  - remove rank cropping to get either full dimension or oversampled dimensions
  - implement performance and quality tests on real phantom C matrix (SVD ground truth, SOR-SVD, RSVD, w/wo power iter)
  - What happens if we don't crop the rank but use the exact oversampling. How little oversampling can we get away with?
- [ ] Revise patch indices creation
- [ ] Create utility/test functions to estimate the required GPU memory for a specific case
  - Calculate analytically the expected memory consumption
- [ ] Is a simple learning rate and steps along the gradient really the fastest/bestest idea?

# Experiments and Results

- Detailed memory requirements
- Phantom, PyTorch vs Matlab
  - Speed and reconstruction quality
  - How many iterations?
  - Testcase at the memory limit
- Invivo testcases: MPM and MESE
  - Unclear how to patch and batch

# Questions and Answers

- What are we going to do about all the parameters?
  - Loraks rank
  - Learning rate
  - Loss lambda
  - Loss criterion for stopping
  - SOR-SVD, RSVD oversampling size
  - SOR-SVD, RSVD number of power iterations
