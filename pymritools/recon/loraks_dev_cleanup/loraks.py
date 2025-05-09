"""
LORAKS algorithm for reconstruction of MRI data.

The end user interface is set up here,
We then have 2 LORAKS "flavors: AC-LORAKS and P-LORAKS.
AC is an option for data where an autocalibration region is available and
can be used to speed up computations and reduce memory consumption
Both are based on the Low rank modelling of k-space described originally by Haldar et al.
https://pubmed.ncbi.nlm.nih.gov/24595341/;

AC LORAKS: https://ieeexplore.ieee.org/document/7164018/
P-LORAKS: https://pubmed.ncbi.nlm.nih.gov/25952136/

This repository was driven by implementing GPU acceleration and feasible Joint-LORAKS reconstruction:
https://onlinelibrary.wiley.com/doi/10.1002/mrm.27076

For each "flavor" we have different computation options.
1) The Loraks matrix Type used (C or S)
2) Data consistency handling (true data consistency or regularizing / balancing loss terms)
3) Rank parameter
4) Neighborhood size parameter

# ToDo: Do we actually just hide some options for using the interface to remove complexity?
    e.g. the leastsqrs option only needs to be available if it really gives a performance difference
     (less memory consumption or quicker) otherwise it might just be for us testing.
     Also "fast" vs "regular" compute might not need to be an option.
     If fast is working, why would one want to use slow? It might make a difference as Barbara has shown
     for some reconstructions but it might not be our task to cover all of those cases.

The algorithm is using torch autograd to perform direct optimization of the Minimization equations.
"""
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch


class LoraksImplementation(Enum):
    P_LORAKS = auto()
    AC_LORAKS = auto()


class LowRankAlgorithmType(Enum):
    TORCH_LOWRANK_SVD = auto()
    RANDOM_SVD = auto()
    SOR_SVD = auto()


class MeasurementType(Enum):
    AC_DATA = auto()
    CALIBRATIONLESS = auto()


class RegularizationType(Enum):
    DATACONSISTENCY = auto()
    REGULARIZED = auto()


class ComputationType(Enum):
    FFT = auto()
    REGULAR = auto()


class OperatorType(Enum):
    C = auto()
    S = auto()


class SVThresholdMethod(Enum):
    HARD_CUTOFF = 150.0
    RELU_SHIFT = 0.9
    RELU_SHIFT_AUTOMATIC = None

    def __init__(self, value: Optional[float]):
        self.threshold = value


@dataclass
class LoraksOptions:
    # rank: SVThresholdMethod = SVThresholdMethod.HARD_CUTOFF
    rank: int = 150
    # TODO: i think rank needs to be a number, as the AC LORAKS version does deduce the nullspace,
    #  but does not do thresholding
    regularization_lambda: float = 0.1
    loraks_neighborhood_radius: int = 3
    loraks_matrix_type: OperatorType = OperatorType.C
    fast_compute: ComputationType = ComputationType.REGULAR
    max_num_iter: int = 20
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# TODO: include indices, matrix shapes and eg. count matrices in the respective operators and make them classes?


class Loraks:
    @staticmethod
    def create(implementation: Optional[LoraksImplementation] = None, options: LoraksOptions = LoraksOptions(),
               **kwargs):
        """Factory method to instantiate the appropriate LORAKS implementation"""

        recon = None
        match implementation:
            case None:
                # Choose the implementation based on option characteristics
                if options.fast_compute == ComputationType.REGULAR:
                    recon = PSLoraks()
                else:
                    recon = LSTSQLoraks(**kwargs)
            case LoraksImplementation.P_LORAKS:
                recon = PSLoraks(**kwargs)
            case LoraksImplementation.AC_LORAKS:
                recon = LSTSQLoraks(**kwargs)
        if recon is None:
            raise RuntimeError("This should never happen. Please report this issue to the developers.")
        return recon.configure(options)


class LoraksBase:
    """
    Base class for all LORAKS implementations.
    Both P- and AC-Loraks will need to implement this interface.
    I believe then we can unify Loraks for the end-user.
    Additionally, each implementation can stay separate
    """

    def __init__(self, **kwargs):
        self.config = {}
        self.configure(**kwargs)

    def configure(self, **kwargs):
        """Configure the solver with the given parameters"""
        self.config.update(kwargs)
        return self

    def reconstruct(self, k_space, sampling_mask=None):
        """Reconstruct k-space data"""
        raise NotImplementedError("Subclasses must implement this method")
