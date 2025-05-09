import logging

import torch
import tqdm
from typing import Callable, Tuple, Optional, Union
import psutil


from pymritools.recon.loraks_dev.matrix_indexing import get_linear_indices
from pymritools.recon.loraks_dev.operators import c_operator, s_operator
from pymritools.recon.loraks_dev_cleanup.loraks import LowRankAlgorithmType, SVThresholdMethod, OperatorType

from pymritools.utils.algorithms import subspace_orbit_randomized_svd, randomized_svd

logger = logging.getLogger(__name__)


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


def create_loss_function(
        operator: Callable[[torch.Tensor, torch.Tensor, Tuple], torch.Tensor],
        svd_func: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        threshold_func: Callable[[torch.Tensor], torch.Tensor]):
    def loss_func(k_space_candidate: torch.Tensor,
                  indices: torch.Tensor,
                  matrix_shape: Tuple,
                  k_sampled_points: torch.Tensor,
                  sampling_mask: torch.Tensor,
                  lam_s: float,
                  device: torch.device = "cpu"):
        # lambda <= 0.0 means true data consistency
        if lam_s > 0.0:
            k_data = k_space_candidate
        else:
            k_data = k_sampled_points.clone()
            k_data[~sampling_mask] = k_space_candidate
        matrix = operator(k_data, indices, matrix_shape)
        u, s, vh = svd_func(matrix)
        s_r = threshold_func(s)
        matrix_recon_loraks = torch.matmul(u * s_r.to(u.dtype), vh)
        loss_1 = torch.linalg.norm(matrix - matrix_recon_loraks, ord="fro")
        if lam_s > 0.0:
            loss_2 = torch.linalg.norm(k_space_candidate * sampling_mask - k_sampled_points)
            return loss_2 + lam_s * loss_1, loss_1, loss_2
        else:
            return loss_1, loss_1, 0.0
    # return torch.compile(loss_func, fullgraph=True)
    return loss_func

def calculate_matrix_size(k_space_shape: Tuple,
                          patch_shape: Tuple,
                          sample_directions: Tuple,
                          matrix_type: OperatorType) -> tuple[int, int]:
    """
        Calculates the operator matrix size for a given k-space shape, patch shape, sample directions,
        and matrix type.

        Args:
        k_space_shape (Tuple): The shape of the k-space which impacts the number of sampling points.
        patch_shape (Tuple): The patch shape to extract from the k-space.
        sample_directions (Tuple): Directional sampling patterns for patches.
        matrix_type (OperatorType): Type of operator matrix

        Returns:
        tuple[int, int]: A tuple containing two integers representing the dimensions of the
        operator matrix based on the given inputs and the specified operator type.
    """
    k_space_shape = torch.tensor(k_space_shape)
    patch_shape = torch.tensor(patch_shape)
    sample_directions = torch.tensor(sample_directions)
    patch_sizes = torch.where(patch_shape == 0, torch.tensor(1), patch_shape)
    patch_sizes = torch.where(patch_shape == -1, k_space_shape, patch_sizes)

    sample_steps = torch.where(sample_directions == 0, torch.tensor(1), k_space_shape - patch_sizes + 1)
    if matrix_type == OperatorType.C:
        return torch.prod(patch_sizes).item(),  torch.prod(sample_steps).item()
    elif matrix_type == OperatorType.S:
        return 2 * torch.prod(patch_sizes).item(), 2 * torch.prod(sample_steps).item()
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")



class Loraks:
    def __init__(self, lambda_factor: float = 0.3, max_num_iter: int = 50):
        self.dirty_config = True
        self.dirty_indices = True

        self.patch_shape: Optional[Tuple] = None
        self.sample_directions: Optional[Tuple] = None
        self.k_space_shape: Optional[Tuple] = None
        self.svd_algorithm: Optional[LowRankAlgorithmType] = None
        self.svd_algorithm_args: Optional[Tuple] = None
        self.operator_type: Optional[OperatorType] = None
        self.sv_cutoff_method: Optional[SVThresholdMethod] = None
        self.sv_cutoff_args: Optional[Tuple] = None
        self.lambda_factor: Optional[float] = lambda_factor
        self.max_num_iter = max_num_iter
        self.learning_rate_func: Callable = lambda _: 1e-3
        self.device: Optional[torch.device] = torch.get_default_device()

        self.loss_func: Optional[Callable] = None
        self.indices: Optional[torch.Tensor] = None
        self.matrix_operator_shape: Optional[Tuple] = None

    def with_patch_shape(self, patch_shape: Tuple) -> "Loraks":
        if self.patch_shape != patch_shape:
            logger.info(f"Patch shape changed from {self.patch_shape} to {patch_shape}")
            self.dirty_config = True
            self.dirty_indices = True
            self.patch_shape = patch_shape
        else:
            logger.info(f"Patch shape unchanged: {self.patch_shape}")
        return self

    def with_sample_directions(self, sample_directions: Tuple) -> "Loraks":
        if self.sample_directions != sample_directions:
            logger.info(f"Sample directions changed from {self.sample_directions} to {sample_directions}")
            self.dirty_config = True
            self.dirty_indices = True
            self.sample_directions = sample_directions
        else:
            logger.info(f"Sample directions unchanged: {self.sample_directions}")
        return self

    def with_torch_lowrank_algorithm(self, q: int, niter: int) -> "Loraks":
        if self.svd_algorithm != LowRankAlgorithmType.TORCH_LOWRANK_SVD or self.svd_algorithm_args != (q, niter):
            logger.info(f"SVD algorithm changed from {self.svd_algorithm} to {LowRankAlgorithmType.TORCH_LOWRANK_SVD}")
            logger.info(f"SVD algorithm arguments changed from {self.svd_algorithm_args} to ({q}, {niter})")
            self.dirty_config = True
            self.svd_algorithm = LowRankAlgorithmType.TORCH_LOWRANK_SVD
            self.svd_algorithm_args = (q, niter)
        else:
            logger.info(f"SVD algorithm unchanged: {self.svd_algorithm}")
            logger.info(f"SVD algorithm arguments unchanged: {self.svd_algorithm_args}")
        return self

    def with_sor_svd_algorithm(self, q: int, niter: int) -> "Loraks":
        if self.svd_algorithm != LowRankAlgorithmType.SOR_SVD or self.svd_algorithm_args != (q, niter):
            logger.info(f"SVD algorithm changed from {self.svd_algorithm} to {LowRankAlgorithmType.SOR_SVD}")
            logger.info(f"SVD algorithm arguments changed from {self.svd_algorithm_args} to ({q}, {niter})")
            self.dirty_config = True
            self.svd_algorithm = LowRankAlgorithmType.SOR_SVD
            self.svd_algorithm_args = (q, niter)
        else:
            logger.info(f"SVD algorithm unchanged: {self.svd_algorithm}")
            logger.info(f"SVD algorithm arguments unchanged: {self.svd_algorithm_args}")
        return self

    def with_rand_svd_algorithm(self, q: int, niter: int) -> "Loraks":
        if self.svd_algorithm != LowRankAlgorithmType.RANDOM_SVD or self.svd_algorithm_args != (q, niter):
            logger.info(f"SVD algorithm changed from {self.svd_algorithm} to {LowRankAlgorithmType.RANDOM_SVD}")
            logger.info(f"SVD algorithm arguments changed from {self.svd_algorithm_args} to ({q}, {niter})")
            self.dirty_config = True
            self.svd_algorithm = LowRankAlgorithmType.RANDOM_SVD
            self.svd_algorithm_args = (q, niter)
        else:
            logger.info(f"SVD algorithm unchanged: {self.svd_algorithm}")
            logger.info(f"SVD algorithm arguments unchanged: {self.svd_algorithm_args}")
        return self

    def with_c_matrix(self) -> "Loraks":
        if self.operator_type != OperatorType.C:
            logger.info(f"Matrix type changed from {self.operator_type} to {OperatorType.C}")
            self.dirty_config = True
            self.operator_type = OperatorType.C
        else:
            logger.info(f"Matrix type unchanged: {self.operator_type}")
        return self

    def with_s_matrix(self) -> "Loraks":
        if self.operator_type != OperatorType.S:
            logger.info(f"Matrix type changed from {self.operator_type} to {OperatorType.S}")
            self.dirty_config = True
            self.operator_type = OperatorType.S
        else:
            logger.info(f"Matrix type unchanged: {self.operator_type}")
        return self

    def with_sv_hard_cutoff(self, q: int, rank: int) -> "Loraks":
        if self.sv_cutoff_method != SVThresholdMethod.HARD_CUTOFF or self.sv_cutoff_args != (q, rank):
            logger.info(
                f"Singular values cutoff method changed from {self.sv_cutoff_method} to {SVThresholdMethod.HARD_CUTOFF}")
            logger.info(f"Singular values cutoff arguments changed from {self.sv_cutoff_args} to ({q},)")
            self.dirty_config = True
            self.sv_cutoff_method = SVThresholdMethod.HARD_CUTOFF
            self.sv_cutoff_args = (q, rank)
        else:
            logger.info(f"Singular values cutoff method unchanged: {self.sv_cutoff_method}")
            logger.info(f"Singular values cutoff arguments unchanged: {self.sv_cutoff_args}")
        return self

    def with_sv_soft_cutoff(self, tau: float) -> "Loraks":
        if self.sv_cutoff_method != SVThresholdMethod.RELU_SHIFT or self.sv_cutoff_args != (tau,):
            logger.info(
                f"Singular values cutoff method changed from {self.sv_cutoff_method} to {SVThresholdMethod.RELU_SHIFT}")
            logger.info(f"Singular values cutoff arguments changed from {self.sv_cutoff_args} to ({tau},)")
            self.dirty_config = True
            self.sv_cutoff_method = SVThresholdMethod.RELU_SHIFT
            self.sv_cutoff_args = (tau,)
        else:
            logger.info(f"Singular values cutoff method unchanged: {self.sv_cutoff_method}")
            logger.info(f"Singular values cutoff arguments unchanged: {self.sv_cutoff_args}")
        return self


    def with_sv_auto_soft_cutoff(self) -> "Loraks":
        if self.sv_cutoff_method != SVThresholdMethod.RELU_SHIFT_AUTOMATIC:
            logger.info(
                f"Singular values cutoff method changed from {self.sv_cutoff_method} to {SVThresholdMethod.RELU_SHIFT}")
            self.dirty_config = True
            self.sv_cutoff_method = SVThresholdMethod.RELU_SHIFT_AUTOMATIC
            self.sv_cutoff_args = tuple()
        else:
            logger.info(f"Singular values cutoff method unchanged: {self.sv_cutoff_method}")
            logger.info(f"Singular values cutoff arguments unchanged: {self.sv_cutoff_args}")
        return self

    def with_constant_learning_rate(self, learning_rate: float) -> "Loraks":
        self.learning_rate_func = lambda _: learning_rate
        return self

    def with_linear_learning_rate(self, min_learning_rate: float, max_learning_rate: float) -> "Loraks":
        self.learning_rate_func = lambda i: (
                min_learning_rate + (max_learning_rate - min_learning_rate) * (1.0 - i / self.max_num_iter)
        )
        return self

    def with_device(self, device: Union[torch.device, str]) -> "Loraks":
        if isinstance(device, str):
            device = torch.device(device)
        if not isinstance(device, torch.device):
            raise ValueError("Device must be a torch.device object")
        if self.device != device:
            logger.info(f"Device changed from {self.device} to {device}")
            self.device = device
        else:
            logger.info(f"Device unchanged: {self.device}")
        return self

    def _initialize_matrix_indices(self):
        self.indices, self.matrix_operator_shape = get_linear_indices(
            self.k_space_shape[1:],
            self.patch_shape,
            self.sample_directions
        )
        if self.operator_type == OperatorType.S:
            self.matrix_operator_shape = tuple(2*d for d in self.matrix_operator_shape)

    def _get_operator_matrix_size(self) -> tuple[int, int]:
        """
        Returns the size of the C/S-matrix which is, in other words, the neighborhood size and the number
        of sampled points we use along the sampling directions.
        This method can only be called from within the reconstruct method when we have already set the shape of the
        k-space.
        """
        if self.k_space_shape is None:
            raise ValueError("k_space_shape must be set before calling this function")
        if self.patch_shape is None:
            raise ValueError("patch_shape must be set before calling this function")
        if self.sample_directions is None:
            raise ValueError("sample_directions must be set before calling this function")
        if self.operator_type is None:
            raise ValueError("operator_type must be set before calling this function")
        return calculate_matrix_size(self.k_space_shape[1:], self.patch_shape, self.sample_directions, self.operator_type)

    def _get_available_cpu_memory(self):
        """
        Returns the total available memory on the CPU in bytes.
        """
        return psutil.virtual_memory().available

    def _get_available_gpu_memory(self):
        """
        Returns the total available memory on the GPU in bytes.
        """
        return torch.cuda.get_device_properties(self.device).total_memory

    def _assert_enough_memory(self):
        m_width, m_height = self._get_operator_matrix_size()
        logger.info(f"Matrix size: {m_width} x {m_height}")
        # TODO: Calculate mem requirements

    def _prepare(self):
        """
        Sets up the necessary configuration for the reconstruction.
        This method can only be called from within the reconstruct method when we have already set the shape of the
        k-space.
        """
        if not self.dirty_config:
            logger.info("No configuration changes detected. Skipping preparation.")
            return

        if not isinstance(self.k_space_shape, tuple) or len(self.k_space_shape) < 3:
            raise ValueError("k_space_shape must be a tuple of at least 3 dimensions (batch, nx, ny, ...)")

        if self.patch_shape is None:
            logger.info("No patch shape specified. Using default.")
            self.with_patch_shape((5, 5, *[-1 for _ in range(len(self.k_space_shape) - 3)]))
        elif len(self.patch_shape) != len(self.k_space_shape) - 1:
            raise ValueError("patch_shape must have the same length as the number of spatial dimensions")

        if self.sample_directions is None:
            logger.info("No sample directions specified. Using default.")
            self.with_sample_directions((-1, -1, *[0 for _ in range(len(self.patch_shape) - 2)]))
        elif len(self.sample_directions) != len(self.patch_shape):
            raise ValueError("sample_directions must have the same length as patch_shape")

        if self.svd_algorithm is None:
            logger.info("No SVD algorithm specified. Using default.")
            # God, I hope this makes sense. For calculating the size of q, we calculate the
            # size of the c/s-matrix and take a portion of it.
            m_width, m_height = self._get_operator_matrix_size()
            neighborhood_size = min(m_width, m_height)
            # This gives a good reduction for size of the low-rank calculation
            self.with_torch_lowrank_algorithm(int(100 * neighborhood_size / (100 + neighborhood_size)), 2)

        if self.operator_type is None:
            logger.info("No operator type specified. Using default.")
            self.with_c_matrix()

        if self.sv_cutoff_method is None:
            logger.info("No singular values cutoff method specified. Using default.")
            # Hard to determine this automatically if it is not set.
            # We just assume we use one of the rand, sor or lowrank methods and use q-2
            self.with_sv_hard_cutoff(self.svd_algorithm_args[0] - 2)

    def reconstruct(self, k_space: torch.Tensor, sampling_mask: torch.Tensor):
        if k_space.shape != self.k_space_shape:
            self.dirty_config = True
            self.k_space_shape = k_space.shape
        self._prepare()
        self._assert_enough_memory()
        self._initialize_matrix_indices()

        operator_func = c_operator if self.operator_type == OperatorType.C else s_operator
        svd_func = get_lowrank_algorithm_function(self.svd_algorithm, self.svd_algorithm_args)
        sv_threshold_func = get_sv_threshold_function(self.sv_cutoff_method, self.sv_cutoff_args, self.device)
        loss_func = create_loss_function(operator_func, svd_func, sv_threshold_func)

        batch_size = self.k_space_shape[0]
        k_input = k_space.contiguous()
        k_out = torch.empty_like(k_input)

        for b in range(batch_size):
            progress_bar = tqdm.trange(self.max_num_iter, desc="Optimization")
            k = k_input[b].clone().to(self.device).requires_grad_()
            sampling_mask_batch = sampling_mask[b].to(self.device)
            k_sampled_points = k_input[b].to(self.device) * sampling_mask_batch

            # iterations
            for i in progress_bar:
                loss, loss_1, loss_2 = loss_func(
                    k,
                    self.indices,
                    self.matrix_operator_shape,
                    k_sampled_points,
                    sampling_mask_batch,
                    self.lambda_factor)
                loss.backward()

                # Use the optimal learning_rate to update parameters
                with torch.no_grad():
                    k -= self.learning_rate_func(i) * k.grad
                k.grad.zero_()

                progress_bar.postfix = (
                    f"LR: {self.learning_rate_func(i):.6f} -- "
                    f"LowRank Loss: {1e3 * loss_1.item():.2f} -- "
                    f"Data Loss: {1e3 * loss_2.item():.2f} -- "
                    f"Total Loss: {1e3 * loss.item():.2f}"
                )

            # k is our converged best guess candidate, need to unwrap / reshape
            k_out[b] = k.detach().cpu()
        return k_out

# l = (Loraks()
#      .with_patch_shape((5, 5, -1, -1, -1))
#      .with_sample_directions((-1, -1, 0, 0, 0))
#      .with_torch_lowrank_algorithm(50, 2)
#      .with_c_matrix()
#      .with_sv_soft_cutoff(1.234)
#      )
# l.reconstruct(my_k_space)


# loss = create_loss_func(randomized_svd, 10, soft_thresholding)
#
# k_space_shape = (256, 256, 2, 3, 4)
# k_space_candidate = torch.randn(k_space_shape)
# indices, matrix_shape = get_linear_indices(k_space_shape, (1, 1, -1, -1, -1), (1, 1, 0, 0, 0))
# sampling_mask = torch.randn(k_space_shape) > 0.9
# k_sampled_points = k_space_candidate*sampling_mask
# lam_s = 1.0
#
# loss(k_space_candidate, indices, matrix_shape, k_sampled_points, sampling_mask, lam_s)
