# DEV considerations

- [x] Fix a requirement to the input shape: `shape = (m, ne, nc, nz, ny, nx)`
- [ ] How to handle all of our torch functions that don't need a device? Fix either this or the speed test function
- [ ] Clean up: 
  - Remove code that is no longer used and that we potentially won't use in the future
  - Move main functions from package files to tests folder
- [ ] SVD
  - [x] Jochen check all `m.T` if a `m.H` is required 
  - [x] check unified interface of functions
  - [x] remove rank cropping to get either full dimension or oversampled dimensions
  - [x] implement performance and quality tests on real phantom C matrix (SVD ground truth, SOR-SVD, RSVD, w/wo power iter)
  - [x] What happens if we don't crop the rank but use the exact oversampling. How little oversampling can we get away with?
  - [ ] How big is the memory impact of the SVD variants for certain matrix sizes?
- [ ] Revise patch indices creation
- [ ] Create utility/test functions to estimate the required GPU memory for a specific case
  - Calculate analytically the expected memory consumption
- [ ] Is a simple learning rate and steps along the gradient really the fastest/bestest idea?
- [ ] Implement minimal and clean recon that takes prepared input and just does the iterations
- [x] Can we estimate the target LORAKS rank from one SVD run upfront?
  - [x] Implement different subsampling schemes for each echo on the SL phantom
- [ ] Why can't we use `lambda * l1 + (1-lambda)*l2` for the loss? Test case is to set lambda to zero now and the iteration should converge immediately. Is that true?
- [ ] Do we want a tensor norm that is indifferent to the actual size/dimensionality? What do we expect when calculating norms in C-Space and in K-Space?
- [ ] If we use a relative data consistency error, we could get rid of lambda: We define maximum percentage that we allow to be the data consistency error and scale the low-rank error to its maximum value

- [ ] Split input into real and imag - real valued tensors? Otherwise, no C implementation with compile possible (no complex implementation of `torch.compile()`)
- [ ] Use linear indexing and matrix operators in AC - LORAKS
- [ ] Fix input floating point deopth to `tf32` for memory efficiency?

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
