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
    rank: SVThresholdMethod = SVThresholdMethod.HARD_CUTOFF
    regularization_lambda: float = 0.5
    loraks_neighborhood_radius: int = 3
    loraks_matrix_type: OperatorType = OperatorType.C
    fast_compute: ComputationType = ComputationType.REGULAR
    max_num_iter: int = 20
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
