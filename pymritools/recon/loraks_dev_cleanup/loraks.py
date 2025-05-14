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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import tqdm

from enum import Enum, auto
from typing import Optional, final, TypeVar

from simple_parsing.helpers import Serializable

logger = logging.getLogger("Loraks")


class LoraksImplementation(Enum):
    """
    Specifies which Loraks algorithm to use.
    """
    P_LORAKS = auto()
    AC_LORAKS = auto()


@dataclass
class RankReductionMethod(Enum):
    """
    Specifies the method used for cutting of ranks to calculate a low-rank representation.
    """
    HARD_CUTOFF = auto()
    RELU_SHIFT = auto()
    RELU_SHIFT_AUTOMATIC = auto()


@dataclass
class RankReduction(Serializable):
    """
    Specifies the method and value used for cutting of ranks to calculate a low-rank representation.
    """
    method: RankReductionMethod
    value: Optional[float | int] = None

    def set_value(self, value: int):
        self.value = value


class OperatorType(Enum):
    """
    Specifies the type of neighborhood matrix to use.
    """
    C = auto()
    S = auto()


@dataclass
class LoraksOptions(Serializable):
    """
    Base options available in each Loraks algorithm.
    Do not use this class directly but instead its subclasses for the particular algorithm.
    """
    loraks_type: LoraksImplementation = LoraksImplementation.P_LORAKS
    loraks_matrix_type: OperatorType = OperatorType.C
    rank: RankReduction = field(default_factory=lambda: RankReduction(method=RankReductionMethod.HARD_CUTOFF, value=150))
    regularization_lambda: float = 0.1
    max_num_iter: int = 20
    device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# We need this type definition for type hinting to later have a method that can take subclasses LoraksOptions
LoraksOptionsType = TypeVar('LoraksOptionsType', bound=LoraksOptions)


# TODO: include indices, matrix shapes and eg. count matrices in the respective operators and make them classes?
class LoraksBase(ABC):
    """
    Base class for all LORAKS implementations.
    Both P- and AC-Loraks will need to implement this interface.
    I believe then we can unify Loraks for the end-user.
    Additionally, each implementation can stay separate
    """

    # Steps needed in any implementation
    # init / configure
    # 1) set options -> AC/P, (fast / slow), S/C, Rank, NB Size, Lambda, optimizing stuff (tol, max numiter, optimizer?)
    # computations
    # 2) Prepare k-space -> unified input, controlled output shapes / batching
    # TODO: Combination method(s) ? channel batching, channel compression, channel & echo combinations?
    # TODO: We also need to prepare the operators here, including set up indices etc.
    # 3) Preparation (different for P vs AC) including extract mask
    # 4) create loss func
    # 5) iteration
    # 6) reverse shape / batch preparation

    # __ public
    @abstractmethod
    def configure(self, options: LoraksOptionsType):
        """Configure the solver with the given parameters"""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def reconstruct_batch(self, k_space_batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _initialize(self, k_space: torch.Tensor):
        """
        Used to initialize the reconstruction algorithm once before processing each batch.
        This method can be implemented by subclasses and is called once before the reconstruction starts.
        """
        pass

    @abstractmethod
    def _prepare_batch(self, batch):
        raise NotImplementedError("Subclasses must implement this method")

    @final
    def reconstruct(self, k_space):
        """
        Reconstructs k-space data.

        """
        # prepare / batch k-space
        k_space_prepared, input_shape, combined_shape = self._prep_k_space_to_batches(k_space)
        # allocate output
        k_space_recon = torch.zeros_like(k_space_prepared)
        for i, batch in tqdm.tqdm(enumerate(k_space_prepared)):
            # put on device - device management, thus _reconstruct_batch to be a device agnostic function?
            # batch = batch.to(self.device)
            k_space_recon[i] = self.reconstruct_batch(batch)
            # memory management?

        return self._unprep_batches_to_k_space(k_space_recon, input_shape, combined_shape)

    @final
    def _prep_k_space_to_batches(self, k_space: torch.Tensor) -> (torch.Tensor, tuple, tuple):
        """
        Prepare the input k-space tensor into batches for separate computations, adjusting for memory requirements.

        The input tensor must have the shape [nt, ne, nz, ny, nx].
        The output tensor will be reshaped into
        batches with a fixed output dimensionality of [b, -1], where b is the batch size.

        :param k_space: Input k-space tensor of shape [nt, ne, nz, ny, nx].
        :raises ValueError: If the input tensor does not match the expected shape.
        :return: Output tensor divided into batches, reshaped to [b, -1].
        """
        # Validate input shape
        if k_space.dim() > 5:
            raise ValueError("Input tensor must have 5 dimensions: [nt, ne, nz, ny, nx].")
        while k_space.dim() < 5:
            logger.warning(
                f"Input tensor has less than 5 dimensions ({k_space.shape})."
                f" We unfold last dimension, might create unwanted behaviour.")
            k_space = k_space.unsqueeze(-1)
        # TODO: we can include the padding here as well
        input_shape = k_space.shape

        # decide on matrix sizes, maybe from memory computations? first create matrices from channels and or contrasts.
        # thus we have spatial dimensions and some combination of channels and contrasts -> [bce, nxyz, nce]
        # ToDo: device combination strategy based on memory computations (those might differ per algorithm used),
        #  should this be an abstract method again?
        k_combined = NotImplemented
        # we flatten everything per batch as we work with linear indices
        b, nxyz, nce = k_combined.shape
        combined_shape = (b, nxyz, nce)

        k_batched = k_combined.reshape(b, -1)

        return k_batched, input_shape, combined_shape

    @final
    def _unprep_batches_to_k_space(self, k_batches: torch.Tensor, input_shape: tuple,
                                   combined_shape: tuple) -> torch.Tensor:
        """
        This method needs to reverse the above batching
        :param k_batches:
        :return:
        """
        # unflatten the shape dim (need some shapes here...)
        # need some information about channel batching (we might take this from above method as well?)
        k_combined = k_batches.reshape(combined_shape)
        # unfold individual dimensions - might need to process some channel / echo combinations
        k_space = k_combined.reshape(input_shape)
        # TODO: we need the unpadding here
        return k_space


class Loraks:
    @staticmethod
    def create(options: LoraksOptionsType) -> LoraksBase:
        """Factory method with support for implementation-specific options"""
        recon = None
        match options.loraks_type:
            case LoraksImplementation.P_LORAKS:
                from pymritools.recon.loraks_dev_cleanup.p_loraks import PLoraks
                recon = PLoraks()
            case LoraksImplementation.AC_LORAKS:
                from pymritools.recon.loraks_dev_cleanup.ac_loraks import AcLoraks
                recon = AcLoraks()
        recon.configure(options)
        return recon
