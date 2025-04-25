import logging
import pathlib as plib

import numpy as np
from scipy.io import loadmat
import torch
from torch.nn.functional import pad

import plotly.graph_objects as go
import plotly.subplots as psub
import tqdm

from pymritools.recon.loraks_dev.matrix_indexing import get_linear_indices
from pymritools.recon.loraks_dev.operators import c_operator, s_operator, c_adjoint_operator, s_adjoint_operator
from pymritools.utils import Phantom, fft, ifft, root_sum_of_squares, torch_load, nifti_save, torch_save
from pymritools.recon.loraks.algorithms import cgd

log_module = logging.getLogger(__name__)


def get_circular_nb_indices(nb_radius):
    # want a circular neighborhood, i.e. find all indices within a radius
    nb_x, nb_y = torch.meshgrid(
        torch.arange(-nb_radius, nb_radius + 1),
        torch.arange(-nb_radius, nb_radius + 1),
        indexing='ij'
    )
    # Create a mask for the circular neighborhood
    nb_r = nb_x ** 2 + nb_y ** 2 <= nb_radius ** 2
    # Get the indices of the circular neighborhood
    return torch.nonzero(nb_r).squeeze()


def get_indices(k_space_shape_2d: tuple, nb_radius: int, reversed: bool = False):

    # want a circular neighborhood radius and convert to linear indices
    neighborhood_indices = get_circular_nb_indices(nb_radius=nb_radius)

    # Calculate offsets relative to the center
    offsets = neighborhood_indices - nb_radius

    y, x = torch.meshgrid(torch.arange(k_space_shape_2d[-2]), torch.arange(k_space_shape_2d[-1]))

    yx = torch.concatenate((y.unsqueeze(-1), x.unsqueeze(-1)), dim=-1)
    yx = torch.reshape(yx, (-1, 2))

    yxnb = yx.unsqueeze(0) + offsets.unsqueeze(1)
    idx = torch.all(
        (yxnb >= torch.tensor([1 - s % 2 for s in k_space_shape_2d[:2]])) &
        (yxnb < torch.tensor(k_space_shape_2d[:2])),
        dim=(0, 2)
    )

    if reversed:
        yxnb = torch.tensor(k_space_shape_2d[:2]).unsqueeze(0).unsqueeze(1) - yx.unsqueeze(0) + offsets.unsqueeze(1)
    else:
        yxnb = yx.unsqueeze(0) + offsets.unsqueeze(1)

    yxnb = yxnb[:, idx]
    # convert to linear indices
    neighborhood_linear_indices = yxnb[..., 0] * k_space_shape_2d[1] + yxnb[..., 1]
    return neighborhood_linear_indices


def new_matlike_s_operator_rev(k_space: torch.Tensor, device: torch.device = torch.get_default_device()):
    indices = get_indices(
        k_space_shape_2d=k_space.shape[-2:], nb_radius=3, reversed=False
    ).to(device)
    indices_rev = get_indices(
        k_space_shape_2d=k_space.shape[-2:], nb_radius=3, reversed=True
    ).to(device)
    matrix_shape = indices.shape

    match k_space.dtype:
        case torch.float32:
            dtype = torch.complex64
        case torch.float64:
            dtype = torch.complex128
        case _:
            dtype = torch.complex128

    k_space = k_space.view(*k_space.shape[:-2], -1)
    # effectively c - matrix in each channel
    s_p = k_space[..., indices].view(-1, *matrix_shape)
    s_m = k_space[..., indices_rev].view(-1, *matrix_shape)

    s_p_m = (s_p - s_m).to(dtype)
    s_u = torch.concatenate([s_p_m.real, -s_p_m.imag], dim=1)

    s_p_m = (s_p + s_m).to(dtype)
    s_d = torch.concatenate([s_p_m.imag, s_p_m.real], dim=1)

    s = torch.concatenate([s_u, s_d], dim=-1).contiguous()
    return s.view(-1, s.shape[-1])


class AC_LORAKS:
    def __init__(
            self,
            k_space_xyzct: torch.Tensor,
            rank: int,
            regularization_lambda: float = 0.0,
            loraks_neighborhood_side_size: int = 5,
            loraks_matrix_type: str = "S",
            fast_compute: bool = True,
            batch_channel_dim: int = -1,
            sampling_mask_xyt: torch.Tensor = None,
            max_num_iter: int = 20,
            conv_tol: float = 1e-3,
            process_slice: bool = False,
            device: torch.device = torch.get_default_device()):
        log_module.info("Initialize LORAKS reconstruction algortihm")
        self.rank: int = rank
        self.regularization_lambda: float = regularization_lambda
        self.loraks_matrix_type = loraks_matrix_type.capitalize()
        self.loraks_neighborhood_side_size: int = loraks_neighborhood_side_size
        self.max_num_iter: int = max_num_iter
        self.conv_tol: float = conv_tol
        self.device: torch.device = device
        self._log(f"Using device: {device}")

        if k_space_xyzct.shape.__len__() < 5:
            self._log_error(
                "Found less than 5 dimensional tensors, "
                "shape tensor e.g. unsqueeze all singular dimensions to fit x-y-z-c-t shape."
            )
        if process_slice:
            self._log(f"Processing single slice toggled (usually for testing performance or parameters).")
            k_space_xyzct = k_space_xyzct[:, :, k_space_xyzct.shape[2] // 2, None]
        if sampling_mask_xyt is None:
            sampling_mask_xyt = torch.abs(k_space_xyzct) > 1e-11
            sampling_mask_xyt = sampling_mask_xyt[:, :, 0, 0]
        else:
            if sampling_mask_xyt.shape.__len__() != 3:
                self._log_error(
                    "sampling_mask_xyt has wrong shape, ensure 3D tensor e.g. unsqueeze x-y-t shape. "
                    "It is assumed that slices and channels are sampled with the same sampling scheme."
                )
        self.sampling_mask_xyt: torch.Tensor = sampling_mask_xyt.to(torch.int)

        # using channel wise computation, save shape
        self._log(f"Input k-space shape: {k_space_xyzct.shape}")
        nx, ny, nz, nc, nt = k_space_xyzct.shape
        self.k_us_xyzct: torch.Tensor = k_space_xyzct.to(torch.complex128)

        # find ac region
        # self.ac_data = self._find_ac_region()
        # self.ac_data = self.ac_data.mT
        self.ac_region_slices = self._find_ac_region()
        ac_data = self.k_us_xyzct[self.ac_region_slices]

        if torch.numel(ac_data) > 1e2:
            self._log(f"Using AC Region - detected region of shape {ac_data.shape}")
        else:
            self._log_error(f"No AC Region found or Region too small ({torch.numel(ac_data)}")

        # check batching
        if 1 < batch_channel_dim < nc:
            if nc % batch_channel_dim > 1e-9:
                self._log_error(
                    f"Channel dimension must be divisible by channel batch-size, "
                    f"otherwise batched matrices will have varying dimensions and would need varying rank settings."
                )
            self.batch_channel_size: int = batch_channel_dim
            self.num_batches = nc // batch_channel_dim
            self._log(f"Using batched channel dimension (size: {batch_channel_dim} / {nc})")
        else:
            self.num_batches = 1
            self.batch_channel_size: int = nc

        # set some shapes
        ac_nx, ac_ny = ac_data.shape[:2]

        self._log("Merging channel and time dimension, set spatial dims to end")
        # we reverse axis order since we will do all computations on the 2D spatial axes and
        # want to benefit from python / torch ordering
        # want to avoid shaping or ordering fuckups
        nce = self.batch_channel_size * nt

        sampling_mask_xyct = self.sampling_mask_xyt[:, :, None].expand(nx, ny, nc, nt)
        # self.sampling_mask_batch = self._combine_channel_echo_dim(
        #     sampling_mask_xyct.mT
        # )

        self.sampling_mask_batch = torch.reshape(
            sampling_mask_xyct.permute(3, 2, 1, 0), (-1, ny, nx)
        )
        self.sampling_mask_batch_bool = self.sampling_mask_batch.to(torch.bool)

        self.ac_shape = (nce, ac_ny, ac_nx)
        self.shape_batch = (nce, ny, nx)

        # set indices of ac region - remember this is slice wise (hence 3D with batched channel and time)
        # indices, ac_matrix_shape = get_linear_indices(
        #     k_space_shape=self.ac_shape[:2],
        #     patch_shape=(loraks_neighborhood_side_size, loraks_neighborhood_side_size),
        #     sample_directions=(1, 1)
        # )
        # self.ac_indices = indices.to(self.device)
        # # set indices for k space - remember this is slice wise (hence 4D)
        # indices, matrix_shape = get_linear_indices(
        #     k_space_shape=self.shape_batch[:2],
        #     patch_shape=(loraks_neighborhood_side_size, loraks_neighborhood_side_size),
        #     sample_directions=(1, 1)
        # )
        # self.indices = indices.to(self.device)

        if self.loraks_matrix_type == "S":
            self._log("Using S - Matrix formulation")
            self.operator = new_matlike_s_operator_rev
            self.operator_adjoint = s_adjoint_operator
            # self.ac_matrix_shape = tuple([2*i for i in ac_matrix_shape])
            # self.ac_matrix_shape = ac_matrix_shape
            # self.matrix_shape = tuple([2*i for i in matrix_shape])
            # self.matrix_shape = matrix_shape
        elif self.loraks_matrix_type == "C":
            self._log("Using C - Matrix formulation")
            self.operator = c_operator
            self.operator_adjoint = c_adjoint_operator
            # self.ac_matrix_shape = ac_matrix_shape
            # self.matrix_shape = matrix_shape
        else:
            self._log_error(f"Matrix formulation ({loraks_matrix_type}) not implemented, choose either 'S' or 'C'")

        if self.regularization_lambda < 1e-9:
            # set strict data consistency
            self._log("Using strict data consistency algorithm")
            self.use_data_consistency = True
        else:
            # set regularization version
            self._log(f"Using regularized algorithm (lambda: {self.regularization_lambda:.3f})")
            self.use_data_consistency = False

        if fast_compute:
            # set fft algorithm
            self._log("Using fast computation via FFTs")
            self.fast_compute = True
        else:
            self._log("Using operator based (slower) algorithm")
            self.fast_compute = False

        # in the end we need an M operator and a b vector to solve Mf = b least squares via cgd
        # self.count_matrix = self._get_count_matrix()
        torch.cuda.empty_cache()

    @staticmethod
    def _combine_channel_echo_dim(x: torch.Tensor):
        shape = x.shape
        if shape.__len__() != 4:
            msg = (f"called function to combine channel and echo dimension on tensor of shape ({shape}). "
                   f"This function assumes a 4D tensor with channel and echo dimension in last two.")
            log_module.error(msg)
            raise AttributeError(msg)
        return torch.reshape(x, (-1, *shape[-2:]))

    @staticmethod
    def _log(msg):
        msg = f"\t\t- {msg}"
        log_module.info(msg)

    @staticmethod
    def _log_error(msg):
        log_module.error(msg)
        raise AttributeError(msg)

    # def _get_count_matrix(self):
    #     in_ones = torch.ones(self.shape_batch, device=self.device, dtype=torch.complex64)
    #     ones_matrix = self.operator(
    #         k_space=in_ones, indices=self.indices, matrix_shape=self.matrix_shape
    #     )
    #     indices, _ = get_linear_indices(
    #         k_space_shape=self.shape_batch,
    #         patch_shape=(self.loraks_neighborhood_side_size, self.loraks_neighborhood_side_size, -1),
    #         sample_directions=(1, 1, 0)
    #     )
    #     count_matrix = self.operator_adjoint(
    #         ones_matrix,
    #         indices=indices.to(self.device), k_space_dims=self.shape_batch
    #     ).real.to(torch.int)
    #     return count_matrix

    def _find_ac_region(self) -> torch.Tensor:
        """
        Find the central rectangular region by expanding from the center,
        ensuring full sampling and consistency across dimensions.

        Args:
            tensor (torch.Tensor): Input tensor of 3 or more dimensions

        Returns:
            tuple: A slice tuple representing the AC region
        """

        # Reduced mask across all but first two dimensions to check consistency
        reduced_mask_2d = self.sampling_mask_xyt.all(dim=tuple(range(2, 3)))

        # Function to find central fully-sampled region in a dimension
        def find_central_region(dim_mask):
            # Find indices of non-zero elements
            non_zero_indices = torch.where(dim_mask)[0]

            if len(non_zero_indices) == 0:
                raise ValueError("No non-zero elements in dimension")

            # Compute dimension size and center
            dim_size = len(dim_mask)
            center = dim_size // 2

            # Initialize region from center
            left = right = center

            # Expand region outwards, ensuring consistent sampling
            while True:
                # Check if we can expand left
                can_expand_left = left > 0 and dim_mask[left - 1]
                # Check if we can expand right
                can_expand_right = right < dim_size - 1 and dim_mask[right + 1]

                # If can't expand either direction, we're done
                if not (can_expand_left or can_expand_right):
                    break

                # Prioritize symmetric expansion
                if can_expand_left and (not can_expand_right or
                                        (center - (left - 1)) <= (right + 1 - center)):
                    left -= 1
                elif can_expand_right:
                    right += 1

            return left, right + 1

        # Find central region for first two dimensions
        ac_slices = []
        for dim in range(2):
            # Reduce to current dimension
            dim_mask = reduced_mask_2d.any(dim=1 - dim)
            start, end = find_central_region(dim_mask)
            ac_slices.append(slice(start, end))

        # Add full slices for additional dimensions
        ac_slices.extend([slice(None) for _ in range(2, 5)])

        # Create and verify the region
        return tuple(ac_slices)
        # ac_region = self.k_us_xyzct[ac_region_slices]
        #
        # # Ensure fully non-zero
        # if (ac_region == 0).any():
        #     raise ValueError("Found region contains zero values")
        #
        # return ac_region

    def _get_k_batch(self, idx_slice: int, idx_batch: int):
        return self._combine_channel_echo_dim(
            # self.k_us_xyzct[
            #     :, :, idx_slice, idx_batch * self.batch_channel_size:(idx_batch + 1 * self.batch_channel_size)
            # ]
            self.k_us_xyzct[
                :, :, idx_slice, idx_batch * self.batch_channel_size:(idx_batch + 1 * self.batch_channel_size)
            ].permute(3, 2, 1, 0).to(self.device)
        )

    def _get_m_operator_orig(self, vvh: torch.Tensor):
        # original implementation of operators
        if self.use_data_consistency:

            def _m_op(x):
                m = self.count_matrix[~self.sampling_mask_batch_bool] * x

                tmp = torch.zeros(self.shape_batch, dtype=self.k_us_xyzct.dtype, device=self.device)
                tmp[~self.sampling_mask_batch_bool] = x
                mvs = s_adjoint_operator(
                    torch.matmul(
                        s_operator(tmp, indices=self.indices, matrix_shape=self.matrix_shape), vvh
                    ),
                    indices=self.indices, k_space_dims=self.shape_batch
                )[~self.sampling_mask_batch_bool]
                return m - mvs

        else:
            aha = (
                    # self.sampling_mask_xyt.unsqueeze(-2).to(self.device) +
                    self.sampling_mask_batch.to(self.device) +
                    self.regularization_lambda * self.count_matrix
            )

            def _m_op(x):
                m_v = s_adjoint_operator(
                    torch.matmul(
                        s_operator(x, indices=self.indices, matrix_shape=self.matrix_shape),
                        vvh
                    ),
                    indices=self.indices, k_space_dims=self.shape_batch
                )
                return aha * x - self.regularization_lambda * m_v

        return _m_op

    def _get_b_vector_orig(self, idx_slice: int, idx_batch: int, vvh: torch.Tensor):
        k_batch = self._get_k_batch(idx_slice=idx_slice, idx_batch=idx_batch).to(self.device).contiguous()
        if self.use_data_consistency:
            return s_adjoint_operator(
                torch.matmul(
                    s_operator(
                        k_batch,
                        indices=self.indices,
                        matrix_shape=self.matrix_shape),
                    vvh
                ),
                indices=self.indices, k_space_dims=self.shape_batch
            )[~self.sampling_mask_batch_bool]
        else:
            return k_batch

    def _m_op_base(self, x: torch.Tensor, v_c: torch.Tensor, v_s: torch.Tensor):
        # dims [nx, ny, nce]
        pad_x = pad(
            x,
            (0, self.loraks_neighborhood_side_size - 1, 0, self.loraks_neighborhood_side_size - 1),
            mode="constant", value=0.0
        )
        fft_x = torch.fft.fft2(pad_x, dim=(-2, -1))
        # dims [nx + nb - 1, ny + nb - 1, nce]
        mv_c = torch.sum(v_c * fft_x.unsqueeze(0), dim=1)
        mv_s = torch.sum(v_s * torch.conj(fft_x).unsqueeze(0), dim=1)

        imv_c = torch.fft.ifft2(mv_c, dim=(-2, -1))[..., :self.shape_batch[-2], :self.shape_batch[-1]]
        imv_s = torch.fft.ifft2(mv_s, dim=(-2, -1))[..., :self.shape_batch[-2], :self.shape_batch[-1]]

        return imv_c - imv_s

    def _get_m_operator_fft(self, v_s: torch.Tensor, v_c: torch.Tensor):
        if self.use_data_consistency:
            def _m_op(x):
                tmp = torch.zeros(self.shape_batch, dtype=self.k_us_xyzct.dtype, device=self.device)
                tmp[~self.sampling_mask_batch_bool] = x
                m = self._m_op_base(tmp, v_c=v_c, v_s=v_s)
                return 2 * m[~self.sampling_mask_batch_bool]

        else:
            # aha = self.sampling_mask_xyt.unsqueeze(-2).to(self.device)
            aha = self.sampling_mask_batch.to(self.device)
            # x has shape [mx, ny, nc, ne]

            def _m_op(x):
                x = torch.reshape(x, self.shape_batch)
                m = self._m_op_base(x, v_c=v_c, v_s=v_s)
                return (aha * x + 2 * self.regularization_lambda * m).flatten()

        return _m_op

    def _get_b_vector_fft(self, idx_slice: int, idx_batch: int, v_s: torch.Tensor, v_c: torch.Tensor):
        k_batch = self._get_k_batch(idx_slice=idx_slice, idx_batch=idx_batch).to(self.device).contiguous()
        if self.use_data_consistency:
            m = self._m_op_base(k_batch, v_c=v_c, v_s=v_s)
            return - 2 * m[~self.sampling_mask_batch_bool]
        else:
            return k_batch.flatten()

    def _get_ac_matrix(self, idx_slice: int, idx_batch: int):
        # build s-matrix
        k_in = self._get_k_batch(idx_slice=idx_slice, idx_batch=idx_batch)
        mask_in = (torch.abs(k_in) > 1e-11).to(torch.int)

        indices = get_indices(k_space_shape_2d=k_in.shape[-2:], nb_radius=3, reversed=False)
        indices_rev = get_indices(k_space_shape_2d=k_in.shape[-2:], nb_radius=3, reversed=True)

        nb_size = indices.shape[0] * k_in.shape[0]
        ac_matrix = self.operator(k_in, device=self.device)

        mask_p = torch.reshape(mask_in.view(*k_in.shape[:-2], -1)[:, indices], (-1, indices.shape[-1]))
        mask_f = torch.reshape(mask_in.view(*k_in.shape[:-2], -1)[:, indices_rev], (-1, indices.shape[-1]))

        idx = (torch.sum(mask_p, dim=0) == nb_size) & (torch.sum(mask_f, dim=0) == nb_size)
        idx = torch.concatenate([idx, idx], dim=0)
        ac_matrix = ac_matrix[:, idx]
        return ac_matrix

    def _get_ac_matrix_arxv(self, idx_slice: int, idx_batch: int):
        # one_in = torch.zeros(self.shape_batch, dtype=torch.complex64, device=self.device)
        # one_in[self.sampling_mask_batch_bool] = torch.randn(torch.count_nonzero(self.sampling_mask_batch_bool), dtype=torch.complex64, device=self.device)
        # matrix = self.operator(
        #     one_in, indices=self.indices, matrix_shape=self.matrix_shape
        # )
        # # want to find fully contained neighborhoods throughout the whole matrix
        # int_mask_matrix = (torch.abs(matrix)>1e-9).to(torch.int)
        # idx = torch.sum(int_mask_matrix, dim=-1) == matrix.shape[-1]

        # ac_data = self.ac_data[
        #         :, :, idx_slice, idx_batch * self.batch_channel_size:(idx_batch + 1 * self.batch_channel_size)
        #     ].mT
        ac_data = self.ac_data[
                :, :, idx_slice, idx_batch * self.batch_channel_size:(idx_batch + 1 * self.batch_channel_size)
            ].permute(3, 2, 1, 0)
        # retrieve data in reversed axis order
        k_ac = self._combine_channel_echo_dim(ac_data).contiguous().to(self.device)
        ac_matrix = self.operator(
            k_space=k_ac,
            # indices=self.ac_indices,
            # matrix_shape=self.ac_matrix_shape
        )
        # ac_matrix = self.operator(
        #     self._get_k_batch(idx_slice=idx_slice, idx_batch=idx_batch).to(self.device), indices=self.indices, matrix_shape=self.matrix_shape
        # )
        # ac_matrix = ac_matrix[idx]

        return ac_matrix

    def _get_ac_subspace_v(self, idx_slice: int, idx_batch: int, use_nullspace: bool = False):
            # find the V matrix for the ac subspaces
        m_ac = self._get_ac_matrix(idx_slice=idx_slice, idx_batch=idx_batch)
        # via eigh
        aha = torch.matmul(m_ac, m_ac.mH)
        assert torch.allclose(aha, aha.mH)
        eig_vals, eig_vecs = torch.linalg.eigh(aha, UPLO="U")
        m_ac_rank = eig_vals.shape[-1]
        # get subspaces from svd of subspace matrix
        # eigenvalues are in ascending order but not absolute value wise!
        idx = torch.argsort(torch.abs(eig_vals), dim=-1, descending=True)
        eig_vecs = eig_vecs[:, idx]
        # eigenvectors are contained as columns
        v_sub = eig_vecs[:, self.rank:] if use_nullspace else eig_vecs[:, :self.rank]
        if m_ac_rank < self.rank:
            err = f"loraks rank parameter is too large, cant be bigger than ac matrix dimensions."
            log_module.error(err)
            raise ValueError(err)
        return v_sub.mH

    def _get_vvh_matrix_of_ac_subspace(self, idx_slice: int, idx_batch: int, use_nullspace: bool = False):
        v_sub = self._get_ac_subspace_v(idx_slice=idx_slice, idx_batch=idx_batch, use_nullspace=use_nullspace)
        return torch.matmul(v_sub.mH, v_sub)

    def _reshape_mat(self, v: torch.Tensor):
        nfilt, filt_size = v.shape
        nc = self.shape_batch[-1]

        v_new = torch.zeros(
            (self.loraks_neighborhood_side_size, self.loraks_neighborhood_side_size, nc, nfilt),
            dtype=v.dtype, device=self.device
        )

        # create indices for circle inside patch of radius R
        x, y = torch.meshgrid(
            [
                torch.arange(-np.floor(self.loraks_neighborhood_side_size // 2), self.loraks_neighborhood_side_size // 2 + 1),
                torch.arange(-np.floor(self.loraks_neighborhood_side_size // 2), self.loraks_neighborhood_side_size // 2 + 1)
            ],
            indexing='ij'
        )
        idx = x ** 2 + y ** 2 <= (self.loraks_neighborhood_side_size // 2) ** 2

        v_new[idx] = torch.reshape(v.mT, (-1, nc, nfilt))
        return v_new

    def _complex_subspace_rep(self, v: torch.Tensor):
        # v = v.mT
        # # extract sizes
        # # nfilt, filt_size = v.shape
        # filt_size, nfilt = v.shape
        # # get neighborhood and channel sizes
        # nb = self.loraks_neighborhood_side_size**2
        # nce = torch.prod(torch.tensor(self.shape_batch[-1])).item()
        #
        # v = torch.reshape(
        #     v,
        #     (2 * nce, nb, nfilt)
        # )
        # # # complex subspace representation
        # v = torch.squeeze(v[::2] + 1j * v[1::2])
        # # v = v[:nce] + 1j * v[nce:]
        #
        # # shape to patch size
        # v = torch.reshape(
        #     v,
        #     (nce, self.loraks_neighborhood_side_size, self.loraks_neighborhood_side_size, nfilt)
        # ).permute(
        #         2, 1, 0, 3
        #     )
        nfilt, filt_size = v.shape
        nce = self.shape_batch[0]
        nb_size = filt_size // nce
        nss_c = torch.reshape(v, (nfilt, -1, nb_size))
        nss_c = nss_c[:, ::2] + 1j * nss_c[:, 1::2]
        return torch.reshape(nss_c, (nfilt, -1))

    def _embed_circular_patch(self, v: torch.Tensor):
        nfilt, filt_size = v.shape

        # get indices
        circular_nb_indices = get_circular_nb_indices(nb_radius=3)
        # find neighborhood size
        nb_size = circular_nb_indices.shape[0]

        # build squared patch
        v = torch.reshape(v, (nfilt, -1, nb_size))
        nc = v.shape[1]

        v_patch = torch.zeros(
            (nfilt, nc, self.loraks_neighborhood_side_size, self.loraks_neighborhood_side_size),
            dtype=v.dtype, device=self.device
        )
        v_patch[:, :, circular_nb_indices[:, 0], circular_nb_indices[:, 1]] = v

        return v_patch

    def _v_0_phase_filter(self, v: torch.Tensor, matrix_type: str = "S"):
        # Conjugate of filters
        cfilt = torch.conj(v)

        # Determine ffilt based on opt
        if matrix_type == 'S':  # for S matrix
            ffilt = torch.conj(v)
        else:  # for C matrix
            ffilt = torch.flip(v, dims=(-2, -1))

        # Perform 2D FFT
        ccfilt = torch.fft.fft2(
            cfilt,
            dim=(-2, -1),
            s=(
                2 * self.loraks_neighborhood_side_size - 1,
                2 * self.loraks_neighborhood_side_size - 1
            )
        )
        fffilt = torch.fft.fft2(
            ffilt,
            dim=(-2, -1),
            s=(
                2 * self.loraks_neighborhood_side_size - 1,
                2 * self.loraks_neighborhood_side_size - 1
            )
        )

        # Compute patch via inverse FFT of element-wise multiplication and sum
        patch = torch.fft.ifft2(
            torch.sum(ccfilt.unsqueeze(2) * fffilt.unsqueeze(1), dim=0),
            dim=(-2, -1)
        )

        return patch

    def _v_pad(self, v_patch: torch.Tensor):
        # assumed dims of v_patch [px, py, nce, nce]
        pad_x = self.shape_batch[-1] - self.loraks_neighborhood_side_size
        pad_y = self.shape_batch[-2] - self.loraks_neighborhood_side_size
        return pad(
            v_patch,
            (
                0, pad_x,
                0, pad_y
            ),
            mode='constant', value=0.0
        )

    def _v_shift(self, v_pad: torch.Tensor, matrix_type: str = "S"):
        if matrix_type == 'S':
            return torch.roll(
                v_pad,
                dims=(-2, -1),
                shifts=(
                    -2 * self.loraks_neighborhood_side_size + 2 - self.shape_batch[-2] % 2,
                    -2 * self.loraks_neighborhood_side_size + 2 - self.shape_batch[-1] % 2
                )
            )
        else:
            return torch.roll(
                v_pad,
                dims=(-2, -1),
                shifts=(
                    - self.loraks_neighborhood_side_size + 1,
                    - self.loraks_neighborhood_side_size + 1
                )
            )

    def _v_fft_prep(self, v: torch.Tensor, matrix_type: str = "S"):
        v_cplx = self._complex_subspace_rep(v)
        v_embed = self._embed_circular_patch(v_cplx)
        v_filtered = self._v_0_phase_filter(v_embed, matrix_type=matrix_type)
        v_pad = self._v_pad(v_filtered)
        v_shift = self._v_shift(v_pad, matrix_type=matrix_type)
        return torch.fft.fft2(v_shift, dim=(-2, -1))

    def _prep_batch(self, idx_slice: int, idx_batch: int):
        # for each batch we do one time computations
        # in AC LORAKS terms we compute the subspace or nullspace from AC data and do some manipulation
        # on this space based on which algorithm method we use
        if self.fast_compute:
            v = self._get_ac_subspace_v(idx_slice=idx_slice, idx_batch=idx_batch, use_nullspace=True)
            # ## we prepare the space via FFT for fast computation
            # mat_s = loadmat("/data/pt_np-jschmidt/code/PyMRItools/resources/vs_matlab_tests/loraks_test_data/filtfilt_s.mat")
            # v = torch.from_numpy(mat_s["filtfilt"]).to(device=self.device, dtype=self.k_us_xyzct.dtype)
            v_s = self._v_fft_prep(v, matrix_type="S")
            v_c = self._v_fft_prep(v, matrix_type="C")

            m_op = self._get_m_operator_fft(v_s=v_s, v_c=v_c)
            b = self._get_b_vector_fft(idx_slice=idx_slice, idx_batch=idx_batch, v_s=v_s, v_c=v_c)

        else:
            # if not fast computation is enabled, we use the subspace in all cases
            v = self._get_vvh_matrix_of_ac_subspace(idx_slice=idx_slice, idx_batch=idx_batch, use_nullspace=False)
            m_op = self._get_m_operator_orig(vvh=v)
            b = self._get_b_vector_orig(idx_slice=idx_slice, idx_batch=idx_batch, vvh=v)
        return m_op, b

    def reconstruct(self):
        log_module.info("Reconstructing k - space data")
        k_recon = torch.zeros_like(self.k_us_xyzct, device=torch.device("cpu"))
        for idx_s in range(self.k_us_xyzct.shape[2]):
            log_module.info(f"\t* Processing slice: {idx_s + 1} / {self.k_us_xyzct.shape[2]}")
            bar = tqdm.trange(self.num_batches, desc="Processing batch")

            for idx_b in bar:
                # for each batch we precompute one time computations and chose operators
                m_op, b = self._prep_batch(idx_slice=idx_s, idx_batch=idx_b)
                if self.use_data_consistency:
                    x_in = torch.zeros_like(b)
                else:
                    x_in = b.clone()
                # and solve via cgd

                xmin, resi, results = cgd(
                    func_operator=m_op, x=x_in, b=b,
                    iter_bar=bar, max_num_iter=self.max_num_iter, conv_tol=self.conv_tol
                )
                if self.use_data_consistency:
                    tmp = torch.zeros(self.shape_batch, dtype=self.k_us_xyzct.dtype, device=xmin.device)
                    tmp[~self.sampling_mask_batch_bool] = xmin

                    result = self._get_k_batch(idx_slice=idx_s, idx_batch=idx_b).cpu() + tmp.cpu()
                else:
                    result = xmin.cpu()
                # reshape to all dims
                result = torch.reshape(result, (-1, self.batch_channel_size, *self.shape_batch[-2:]))
                # get in input order
                result = result.permute(3, 2, 1, 0)
                k_recon[
                    :, :, idx_s, idx_b * self.batch_channel_size:(idx_b + 1) * self.batch_channel_size
                ] = result
        return k_recon


def recon_data_consistency(
        k_space: torch.Tensor, sampling_mask: torch.Tensor,
        rank: int, loraks_neighborhood_size: int = 5, matrix_type: str = "S",
        max_num_iter: int = 200,
        device: torch.device = torch.get_default_device()):
    if matrix_type.capitalize() == "S":
        use_s_matrix = True
    elif matrix_type.capitalize() == "C":
        use_s_matrix = False
    else:
        err = "Currently only S or C matrix LORAKS types are supported."
        log_module.error(err)
        raise ValueError(err)
    k_space = k_space.to(dtype=torch.complex64)

    mask = sampling_mask.to(torch.bool)
    mask_unsampled = ~mask

    logging.info("Set Matrix Indices and AC Matrix")
    # ToDo: Calculate size of AC subregion if whole region is to big for CPU?
    sample_dir = torch.zeros(len(k_space.shape), dtype=torch.int)
    sample_dir[:2] = 1
    patch_shape = torch.full(size=(len(k_space.shape),), fill_value=-1, dtype=torch.int)
    patch_shape[:2] = loraks_neighborhood_size

    indices, matrix_shape = get_linear_indices(
        k_space_shape=k_space.shape,
        patch_shape=tuple(patch_shape.tolist()),
        sample_directions=tuple(sample_dir.tolist())
    )

    if use_s_matrix:
        matrix_shape = tuple((torch.tensor(matrix_shape, dtype=torch.int) * torch.tensor([2, 2])).tolist())

    logging.info("Init optimization")
    # we optimize only unmeasured data
    # find read direction
    read_dir = -1
    for i in range(2):
        tmp_mask = torch.movedim(mask[:, :, 0].to(torch.int), i, 0)
        shape = list(tmp_mask.shape)
        shape.pop(1)
        if torch.sum(tmp_mask[:, tmp_mask.shape[1] // 2]) == torch.prod(torch.tensor(shape, dtype=torch.int)):
            read_dir = i
    if read_dir < 0:
        err = "Could not find read direction."
        log_module.error(err)
        raise ValueError(err)

    shape = torch.tensor(k_space.shape)
    if read_dir == 0:
        shape[1] = torch.count_nonzero(mask_unsampled) / (shape[0] * torch.prod(shape[2:]))
    else:
        shape[0] = torch.count_nonzero(mask_unsampled) / torch.prod(shape[1:])
    k = torch.zeros(tuple(shape), dtype=k_space.dtype, device=device, requires_grad=True)

    lr = torch.linspace(1e-3, 1e-4, max_num_iter, device=device)
    mask_unsampled = mask_unsampled.to(device)

    # we take the 0 filled sampled data to the device
    k_sampled = k_space.to(device)
    # build truncation vector
    oversampling = 10
    q = rank + oversampling
    s_tr = torch.zeros(q, device=device)
    s_tr[:-oversampling] = 1

    progress_bar = tqdm.trange(max_num_iter, desc="Optimization")
    for i in progress_bar:
        # we need a zero filling operator to compute the losses
        tmp = torch.zeros_like(k_sampled)
        tmp[mask_unsampled] = k.flatten()

        if use_s_matrix:
            matrix = s_operator(k_space=k_sampled+tmp, indices=indices, matrix_shape=matrix_shape)
        else:
            matrix = c_operator(k_space=k_sampled+tmp, indices=indices, matrix_shape=matrix_shape)

        # do svd
        u, s, v = torch.svd_lowrank(A=matrix, q=rank+10, niter=2)
        # truncate
        s = s_tr * s
        # build lr-matrix
        matrix_lr = torch.matmul(u * s.to(u.dtype), v.mH)

        loss = torch.linalg.norm(matrix - matrix_lr, ord="fro")

        loss.backward()

        with torch.no_grad():
            k -= lr[i] * k.grad

        norm_grad = torch.linalg.norm(k.grad).item()
        progress_bar.postfix = (
            f"loss: {loss.item():.2f}, norm k: {torch.linalg.norm(k).item():.3f}, norm grad: {norm_grad:.3f}"
        )
        k.grad.zero_()
        if norm_grad < 1e-3:
            msg = f"Optimization converged at step: {i+1}"
            log_module.info(msg)
            break
    # build data
    tmp = torch.zeros_like(k_sampled)
    tmp[mask_unsampled] = k.flatten().detach()
    k_recon = k_sampled + tmp
    return k_recon.cpu()


def main_pbp():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Set Device: {device}")

    logging.info(f"Set Paths")
    path = plib.Path(__name__).parent.absolute()
    path_fig = path.joinpath("figures").absolute()
    path_fig.mkdir(exist_ok=True, parents=True)

    logging.info("Set Phantom")
    # nx, ny, nc, ne = (256, 256, 4, 2)
    # phantom = Phantom.get_shepp_logan(shape=(nx, ny), num_coils=nc, num_echoes=ne)
    # sl_us = phantom.sub_sample_ac_random_lines(acceleration=3, ac_lines=40)

    # mat = loadmat("/data/pt_np-jschmidt/code/PyMRItools/resources/vs_matlab_tests/loraks_test_data/ac_loraks_s-matrix-testing.mat")
    sl_us = torch.load("/data/pt_np-jschmidt/code/PyMRItools/resources/vs_matlab_tests/loraks_phantom/sl_phantom_256x224x4x2.pt")
    # sl_us = torch.from_numpy(mat["kData"]) + 1e-9
    sl_us = sl_us.contiguous()

    img_us = torch.abs(fft(sl_us, dims=(0, 1)))
    # img_us = root_sum_of_squares(img_us, dim_channel=-2)

    ac_loraks = AC_LORAKS(
        k_space_xyzct=sl_us.unsqueeze(2), rank=50, regularization_lambda=0.0,
        loraks_neighborhood_side_size=7,
        fast_compute=True, device=device, process_slice=True, max_num_iter=20,
        conv_tol=1e-3
    )
    k_recon = ac_loraks.reconstruct()[:, :, 0]

    img_recon = torch.abs(fft(k_recon, dims=(0, 1)))

    fig = psub.make_subplots(
        rows=3, cols=4
    )
    for i, d in enumerate([sl_us, img_us, k_recon, img_recon, img_recon - img_us]):
        if i % 2 == 0 and i < 4:
            d = torch.log(torch.abs(d))
        else:
            d = torch.abs(d)
        row = int(i / 2) + 1
        for c in range(2):
            col = int(i % 2) + 2 * c + 1
            fig.add_trace(
                go.Heatmap(z=d[:, :, c, 0], showscale=False),
                row=row, col=col
            )
            fig.update_xaxes(visible=False, row=row, col=col)
            xaxis = fig.data[-1].xaxis
            fig.update_yaxes(visible=False, scaleanchor=xaxis, row=row, col=col)
    fig_name = plib.Path("scratches/figures/js_loraks_test.html").absolute()
    log_module.info(f"Saving figure: {fig_name}")
    fig.write_html(fig_name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main_pbp()
