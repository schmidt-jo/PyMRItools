import logging

import torch
import tqdm
from typing import Callable, Tuple, Optional, Union
from enum import Enum, auto

from pymritools.recon.loraks_dev.matrix_indexing import get_linear_indices
from pymritools.utils.algorithms import subspace_orbit_randomized_svd, randomized_svd

log_module = logging.getLogger(__name__)


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
    HARD_CUTOFF = auto()
    RELU_SHIFT = auto()
    RELU_SHIFT_AUTOMATIC = auto()


def get_lowrank_algorithm_function(algorithm: LowRankAlgorithmType, args: Tuple):
    # Check if args is actually two values and both are integers
    if len(args) != 2 or not isinstance(args[0], int) or not isinstance(args[1], int):
        raise ValueError("Lowrank algorithm arguments must be two integers q and niter.")
    q, niter = args
    match algorithm:
        case LowRankAlgorithmType.TORCH_LOWRANK_SVD:
            def func(matrix: torch.Tensor):
                u, s, v = torch.svd_lowrank(A=matrix, q=q, niter=niter)
                return u, s, v.mH

            return func
        case LowRankAlgorithmType.RANDOM_SVD:
            return lambda matrix: randomized_svd(matrix=matrix, q=q, power_projections=niter)
        case LowRankAlgorithmType.SOR_SVD:
            return lambda matrix: subspace_orbit_randomized_svd(matrix=matrix, q=q, power_projections=niter)


def get_sv_threshold_function(method: SVThresholdMethod, args: Tuple, device: torch.device = "cpu"):
    match method:
        case SVThresholdMethod.HARD_CUTOFF:
            if len(args) != 2 or not isinstance(args[0], int) or not isinstance(args[1], int):
                raise ValueError("Hard cutoff method arguments must be two integers: "
                                 "the length of the singular value vector and the the  cut_off_index.")
            sv_multiplier = torch.ones(args[0], device=device)
            sv_multiplier[args[1]:] = 0
            return lambda singular_values: singular_values * sv_multiplier
        case SVThresholdMethod.RELU_SHIFT:
            if len(args) != 1 or not isinstance(args[0], float):
                raise ValueError("ReLU cutoff method arguments must be one float tau.")
            return lambda singular_values: torch.relu(singular_values - args[0])
        case SVThresholdMethod.RELU_SHIFT_AUTOMATIC:
            if len(args) != 0:
                raise ValueError("ReLU shift automatic method does not take arguments.")
            def func(singular_values: torch.Tensor):
                scaled_cum_sum = torch.cumsum(singular_values, dim=0)/torch.sum(singular_values)
                idx = torch.nonzero(scaled_cum_sum < 0.9)[-1].item()
                print(f"Using cutoff index {idx} with value {singular_values[idx]}")
                return torch.relu(singular_values - singular_values[idx])
            return func
        case _:
            raise ValueError(f"Unknown singular value cutoff method: {method}")


# define the data-consistency loss, if data consistency is selected, this is 0 as the sampled data remains unaffected by the algorithm
# otherwise its the distance of the candidate to input at the sampled points
def get_data_consistency_loss_function(method: RegularizationType, args: Tuple, device: torch.device = "cpu"):
    if len(args) != 1 or not isinstance(args[0], float):
        raise ValueError("Regularization parameter must be one float.")
    regularization_lambda = args[0]
    match method:
        case RegularizationType.DATACONSISTENCY:
            def dc_loss_func(k_space_candidate: torch.Tensor,
                             k_space_sampled: torch.Tensor,
                             sampling_mask: torch.Tensor
                             ):
                return 0.0

        case RegularizationType.REGULARIZED:
            def dc_loss_func(k_space_candidate: torch.Tensor,
                             k_space_sampled: torch.Tensor,
                             sampling_mask: torch.Tensor
                             ):
                return k_space_candidate * sampling_mask - k_space_sampled
            return lambda matrix: regularization_lambda * torch.linalg.norm(matrix, orf="fro")
        case _:
            raise ValueError(f"Unknown data consistency loss setting: {method}")
    return dc_loss_func

# define a data embedding operator: this puts the candidate into the k-space at non-sampled points
# if true data consistency is selected, otherwise just return the candidate
def get_candidate_data_embedding(method: RegularizationType, device: torch.device = "cpu"):
    match method:
        case RegularizationType.DATACONSISTENCY:
            def embed_data(k_space_candidate: torch.Tensor,
                           k_space_sampled: torch.Tensor):
                mask = k_space_sampled.abs() < 1e-10
                k_space_sampled[mask] = k_space_candidate
                return k_space_sampled
        case RegularizationType.REGULARIZED:

            def embed_data(k_space_candidate: torch.Tensor,
                           k_space_sampled):
                return k_space_candidate
        case _:
            raise ValueError(f"Unknown data consistency loss setting: {method}")
    return embed_data


# ToDo: We have a couple of avenues:
#   AC-LORAKS and P - LORAKS,
#   each with true data consistency or regularization.
#   AC additionally with FFT accelerated computations or normal,

# Data Consistency: We can define an embedding operator.
# This would either embed the candidate data into the full k-space or take the candidate data as full k-space,
# based on the data consistency setting

#
def create_loss_function(
        operator: Callable[[torch.Tensor, torch.Tensor, Tuple], torch.Tensor],
        svd_func: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        threshold_func: Callable[[torch.Tensor], torch.Tensor],
        data_consistency_loss: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], float],
        data_embedding: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    def loss_func(k_space_candidate: torch.Tensor,
                  indices: torch.Tensor,
                  matrix_shape: Tuple,
                  k_sampled_points: torch.Tensor,
                  sampling_mask: torch.Tensor,
                  lam_s: float,
                  device: torch.device = "cpu"):
        k_data = data_embedding(k_space_candidate, k_sampled_points)
        matrix = operator(k_data, indices, matrix_shape)
        u, s, vh = svd_func(matrix)
        s_r = threshold_func(s)
        matrix_recon_loraks = torch.matmul(u * s_r.to(u.dtype), vh)
        loss_1 = torch.linalg.norm(matrix - matrix_recon_loraks, ord="fro")
        loss_2 = data_consistency_loss(k_space_candidate, k_sampled_points, sampling_mask)
        return loss_2 + lam_s * loss_1, loss_1, loss_2

    # return torch.compile(loss_func, fullgraph=True)
    return loss_func



