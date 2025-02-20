# DEV considerations

- [x] Todo Jochen: Find paper that estimates the cropping rank from median or something
  - [Manjon et al. 2015](http://dx.doi.org/10.1016/j.media.2015.01.004) uses the median of eigenvalues to estimate the noise contributions in singular values.
  - we could use something similar. If the data is sparse i.e. low rank, the median of the eigenvalues might provide a robust estimator for the singular value cutoff region.
- [ ] Can we automatically adjust and calculate the rank? E.g. 90% SV integral
  - [ ] Quality difference between SV cropping and ReLU shift (polynomial decay, exponential decay)
- [ ] Can we estimate the target LORAKS rank from one SVD run upfront?
- [ ] How to handle all of our torch functions that don't need a device? Fix either this or the speed test function
- [ ] Clean up: 
  - Remove code that is no longer used and that we potentially won't use in the future
  - Move main functions from package files to tests folder
- [ ] Is a simple learning rate and steps along the gradient really the fastest/bestest idea?
  - [ ] Test known optimizers like Adam, LBFGS
- [ ] Why can't we use `lambda * l1 + (1-lambda)*l2` for the loss? Test case is to set lambda to zero now and the iteration should converge immediately. Is that true?
  - [ ] If we use a relative data consistency error, we could get rid of lambda: We define maximum percentage that we allow to be the data consistency error and scale the low-rank error to its maximum value
- [ ] Can we express a complex C matrix with a real valued one and get usable results from the SVD
- [ ] Use linear indexing and matrix operators in AC - LORAKS
- [ ] Make input floating point depth configurable to `tf32` for memory efficiency?
- [ ] Chose c-matrix method to be complex or real valued
 

- [x] Fix a requirement to the input shape: `shape = (m, ne, nc, nz, ny, nx)`
- [x] SVD
  - [x] Jochen check all `m.T` if a `m.H` is required 
  - [x] check unified interface of functions
  - [x] remove rank cropping to get either full dimension or oversampled dimensions
  - [x] implement performance and quality tests on real phantom C matrix (SVD ground truth, SOR-SVD, RSVD, w/wo power iter)
  - [x] What happens if we don't crop the rank but use the exact oversampling. How little oversampling can we get away with?
  - [x] How big is the memory impact of the SVD variants for certain matrix sizes?
- [x] Revise patch indices creation
- [x] Create utility/test functions to estimate the required GPU memory for a specific case
  - Calculate analytically the expected memory consumption
- [x] Implement minimal and clean recon that takes prepared input and just does the iterations
- [x] Implement different subsampling schemes for each echo on the SL phantom
- [x] Do we want a tensor norm that is indifferent to the actual size/dimensionality? What do we expect when calculating norms in C-Space and in K-Space?


- [ ] Check Licensing

# Experiments and Results

Datasets:
  - Case 1: simple 2D image (Jupiter)
  - Case 2: high-dimensional phantom image (coils, echoes)
  - Case 3: in vivo real data

- Detailed memory requirements
  - Calculation and trace (on C1 and C2)
  - SVD variants
- Performance Benchmarking:
  - PyTorch CPU vs. GPU (C2)
  - SVD variants?
  - PyTorch (fastest) vs. Matlab
- Performance Quality:
  - Reconstruction quality (C1 and C2)
  - SVD oversampling and power iterations (C1 oder C2)
  - How many iterations? (C2 with cool subsampling)
  - Testcase realistic dataset (C3)
  - Invivo testcases: MPM and MESE
    - Unclear how to patch and batch

# Questions and Answers

- What are we going to do about all the parameters?
  - Loraks rank
  - Learning rate
  - Loss lambda
  - Loss criterion for stopping
  - [x] SOR-SVD, RSVD oversampling size (in dataset tests) 
  - [x] SOR-SVD, RSVD number of power iterations (in dataset tests)


# Questions Justin
- S FFT convolution:
  The FFT convolution argument is clear for the C-Matrix type, however, the S-matrix
  convolution structure is not as obvious. How would one need to adopt the $L_i$ operator considering in our algorithm
  only the forward pass to the loss function is necessary. It appears as its a sum of the C convolution structure
  and a conjugated filtering method.