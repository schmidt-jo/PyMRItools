import logging
import pathlib as plib
import torch
import tqdm
from torch.nn.functional import pad
from pymritools.utils import Phantom, ifft

from pymritools.utils.algorithms import cgd
import plotly.graph_objects as go
import plotly.subplots as psub

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


def get_indices(k_space_shape: tuple, nb_radius: int, reversed: bool = False):

    # want a circular neighborhood radius and convert to linear indices
    neighborhood_indices = get_circular_nb_indices(nb_radius=nb_radius)

    # Calculate offsets relative to the center
    offsets = neighborhood_indices - nb_radius

    y, x = torch.meshgrid(torch.arange(k_space_shape[-2]), torch.arange(k_space_shape[-1]))

    yx = torch.concatenate((y.unsqueeze(-1), x.unsqueeze(-1)), dim=-1)
    yx = torch.reshape(yx, (-1, 2))

    yxnb = yx.unsqueeze(0) + offsets.unsqueeze(1)
    idx = torch.all(
        (yxnb >= torch.tensor([1 - s % 2 for s in k_space_shape[:2]])) &
        (yxnb < torch.tensor(k_space_shape[:2])),
        dim=(0, 2)
    )

    if reversed:
        yxnb = torch.tensor(k_space_shape[:2]).unsqueeze(0).unsqueeze(1) - yx.unsqueeze(0) + offsets.unsqueeze(1)
    else:
        yxnb = yx.unsqueeze(0) + offsets.unsqueeze(1)

    yxnb = yxnb[:, idx]
    # convert to linear indices
    neighborhood_linear_indices = yxnb[..., 0] * k_space_shape[1] + yxnb[..., 1]
    return neighborhood_linear_indices


def new_matlike_s_operator_rev(k_space: torch.Tensor, indices: torch.Tensor, indices_rev: torch.Tensor, matrix_shape: tuple):
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
            # loraks_neighborhood_side_size: int = 5,
            loraks_neighborhood_radius: int = 3,
            loraks_matrix_type: str = "S",
            fast_compute: bool = True,
            batch_channel_dim: int = -1,
            max_num_iter: int = 20,
            conv_tol: float = 1e-3,
            process_slice: bool = False,
            device: torch.device = torch.get_default_device()):

        log_module.info("Initialize LORAKS reconstruction algortihm")
        self.rank: int = rank
        self.regularization_lambda: float = regularization_lambda
        self.loraks_matrix_type = loraks_matrix_type.capitalize()
        self.loraks_neighborhood_radius: int = loraks_neighborhood_radius
        self.loraks_neighborhood_side_size: int = 2 * loraks_neighborhood_radius + 1
        self.max_num_iter: int = max_num_iter
        self.conv_tol: float = conv_tol
        self.device: torch.device = device
        self._log(f"Using device: {device}")

        self._log(f"Input k-space shape: {k_space_xyzct.shape}")
        if k_space_xyzct.shape.__len__() < 5:
            self._log_error(
                "Found less than 5 dimensional tensors, "
                "shape tensor e.g. unsqueeze all singular dimensions to fit x-y-z-c-t shape."
            )
        if process_slice:
            self._log(f"Processing single slice toggled (usually for testing performance or parameters).")
            k_space_xyzct = k_space_xyzct[:, :, k_space_xyzct.shape[2] // 2, None]
        # First lets get the shape
        self.in_shape_xyzct = k_space_xyzct.shape

        # check if we need to batch the channel dimension
        self.batch_channel_size, self.num_channel_batches = self._check_channel_batching(
            batch_channel_dim=batch_channel_dim, nc=k_space_xyzct.shape[-2]
        )
        self.k_batched: torch.Tensor = self._reshape_input_to_batches(k_data_xyzct=k_space_xyzct)
        self.mask_batched: torch.Tensor = torch.abs(self.k_batched) > 1e-10
        # save shape of batch (only [nce, ny, nx])
        self.shape_batch: tuple = self.k_batched.shape[-3:]

        # build operator indices
        self.indices: torch.Tensor = get_indices(
            k_space_shape=self.shape_batch[-2:], nb_radius=loraks_neighborhood_radius, reversed=False
        )
        self.indices_rev: torch.Tensor = get_indices(
            k_space_shape=self.shape_batch[-2:], nb_radius=loraks_neighborhood_radius, reversed=True
        )

        if not self.loraks_matrix_type == "S":
            self._log_error("Other matrix types not yet implemented")

        def s_operator(k_space: torch.Tensor):
            return new_matlike_s_operator_rev(
                k_space=k_space,
                indices_rev=self.indices_rev,
                indices=self.indices,
                matrix_shape=self.indices.shape
            )

        self.operator = s_operator

        if self.regularization_lambda < 1e-9:
            # set strict data consistency
            self._log("Using strict data consistency algorithm")
            self.use_data_consistency = True
        else:
            self._log_error("Using regularization not yet implemented")
            # set regularization version
            self._log(f"Using regularized algorithm (lambda: {self.regularization_lambda:.3f})")
            self.use_data_consistency = False

        if fast_compute:
            # set fft algorithm
            self._log("Using fast computation via FFTs")
            self.fast_compute = True
        else:
            self._log_error("Using slow compute not yet implemented")
            self._log("Using operator based (slower) algorithm")
            self.fast_compute = False

        # in the end we need an M operator and a b vector to solve Mf = b least squares via cgd
        # self.count_matrix = self._get_count_matrix()
        torch.cuda.empty_cache()

    @staticmethod
    def _log(msg):
        msg = f"\t\t- {msg}"
        log_module.info(msg)

    @staticmethod
    def _log_error(msg):
        log_module.error(msg)
        raise AttributeError(msg)

    def _reshape_input_to_batches(self, k_data_xyzct) -> torch.Tensor:
        # we want to take input data arranged in x-y-z-c-t (like nifti or from raw data)
        # and implement a slice wise computation (batch slices), additionally, we might need to
        # compress channels or batch them additionally
        # thus in this method we want to end up with batched k-space: [b, nct, ny, nx],
        # where nct is combining echos and (potentially batched) channels.
        # reverse axis
        k_data = k_data_xyzct.permute(4, 3, 2, 1, 0)
        # assume we checked the batching of channels already, we do the batching
        k_data_bc = torch.tensor_split(k_data.unsqueeze(0), self.num_channel_batches, dim=2)
        # we now have a tuple of channel batched k tensors
        k_data_bc = torch.concatenate(k_data_bc, dim=0)
        # should be dims [bc, nt, ncb, nz, ny, nx]
        bc, nt, ncb, nz, ny, nx = k_data_bc.shape
        # we pull the z dim to front
        k_data_bcz = k_data_bc.permute(0, 3, 1, 2, 4, 5)
        # dims [bc, nz, nt, ncb, ny, nx]
        # we reshape echo and batched channels, and batch dimension
        k_data_b = torch.reshape(k_data_bcz, (bc * nz, nt*ncb, ny, nx))
        return k_data_b

    def _reshape_batches_to_input(self, k_data_b):
        # basically reversing above steps
        # in dims [b, ncbt, ny, nx]
        # reshape into all individual dimensions
        k_data_bcz = torch.reshape(
            k_data_b,
            (
                self.num_channel_batches, self.in_shape_xyzct[2],
                self.in_shape_xyzct[-1], self.batch_channel_size,
                self.in_shape_xyzct[1], self.in_shape_xyzct[0])
        )
        # dims [bc, nz, nt, ncb, ny, nx]
        k_data_bc = k_data_bcz.permute(0, 2, 3, 1, 4, 5)
        # now dims [bc, nt, ncb, nz, ny, nx]
        # concatenate split channel batches
        k_data = torch.concatenate([k for k in k_data_bc], dim=1)
        # now dims [nt, nc, nz, ny, nx]
        # reverse order
        k_data = k_data.permute(4, 3, 2, 1, 0)
        return k_data

    def _check_channel_batching(self, batch_channel_dim, nc) -> (torch.Tensor, torch.Tensor):
        # check batching
        if 1 < batch_channel_dim < nc:
            if nc % batch_channel_dim > 1e-9:
                self._log_error(
                    f"Channel dimension must be divisible by channel batch-size, "
                    f"otherwise batched matrices will have varying dimensions and "
                    f"possibly would need varying rank settings."
                )
            batch_channel_size: int = batch_channel_dim
            num_batches = nc // batch_channel_dim
            self._log(f"Using batched channel dimension (size: {batch_channel_dim} / {nc})")
        else:
            num_batches = 1
            batch_channel_size: int = nc
        return batch_channel_size, num_batches

    def _get_ac_matrix(self, k_data: torch.Tensor, mask: torch.tensor):

        nb_size = self.indices.shape[0] * k_data.shape[0]
        ac_matrix = self.operator(k_data)

        mask_p = torch.reshape(mask.view(*k_data.shape[:-2], -1)[:, self.indices], (-1, self.indices.shape[-1]))
        mask_f = torch.reshape(mask.view(*k_data.shape[:-2], -1)[:, self.indices_rev], (-1, self.indices.shape[-1]))

        idx = (torch.sum(mask_p, dim=0) == nb_size) & (torch.sum(mask_f, dim=0) == nb_size)
        idx = torch.concatenate([idx, idx], dim=0)
        ac_matrix[:, ~idx] = 0.0
        return ac_matrix

    def _get_nullspace(self, m_ac):
        mmh = m_ac @ m_ac.mH

        e_vals, e_vecs = torch.linalg.eigh(mmh, UPLO="U")

        idx = torch.argsort(torch.abs(e_vals), descending=True)
        um = e_vecs[:, idx]

        return um[:, self.rank:].mH

    def _complex_subspace_representation(self, v: torch.Tensor):
        nb_size = self.indices.shape[0]
        nfilt, filt_size = v.shape
        nss_c = torch.reshape(v, (nfilt, -1, nb_size))
        nss_c = nss_c[:, ::2] + 1j * nss_c[:, 1::2]
        return torch.reshape(nss_c, (nfilt, -1))

    def _embed_circular_patch(self, v: torch.Tensor):
        nfilt, filt_size = v.shape

        # get indices
        circular_nb_indices = get_circular_nb_indices(nb_radius=self.loraks_neighborhood_radius)
        # find neighborhood size
        nb_size = circular_nb_indices.shape[0]

        # build squared patch
        v = torch.reshape(v, (nfilt, -1, nb_size))
        nc = v.shape[1]

        v_patch = torch.zeros(
            (nfilt, nc, self.loraks_neighborhood_side_size, self.loraks_neighborhood_side_size),
            dtype=v.dtype
        )
        v_patch[:, :, circular_nb_indices[:, 0], circular_nb_indices[:, 1]] = v

        return v_patch

    def _zero_phase_filter(self, v: torch.Tensor, matrix_type: str = "S"):
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

    def _get_m_operator_fft(self, v_s: torch.Tensor, v_c: torch.Tensor, mask: torch.Tensor):
        def _m_op(x):
            tmp = torch.zeros(self.shape_batch, dtype=v_s.dtype)
            tmp[~mask] = x
            m = self._m_op_base(tmp, v_c=v_c, v_s=v_s)
            return 2 * m[~mask]

        return _m_op

    def _get_b_vector_fft(self, k: torch.Tensor, mask: torch.Tensor, v_s: torch.Tensor, v_c: torch.Tensor):
        m = self._m_op_base(k, v_c=v_c, v_s=v_s)
        return - 2 * m[~mask]

    def reconstruct(self):
        # allocate space
        k_recon = torch.zeros_like(self.k_batched)
        # set up progress bar
        bar = tqdm.trange(self.k_batched.shape[0], desc="Reconstruction")
        for idx_b in bar:
            k_in = self.k_batched[idx_b].to(self.device)
            mask = self.mask_batched[idx_b].to(self.device)

            # get ac matrix
            m_ac = self._get_ac_matrix(k_data=k_in, mask=mask)

            # eigenvalue decomposition
            v = self._get_nullspace(m_ac=m_ac)

            # complexify nullspace
            v = self._complex_subspace_representation(v)

            # prep zero phase filter input
            v_patch = self._embed_circular_patch(v)

            # zero phase filter
            vs_filt = self._zero_phase_filter(v_patch.clone(), matrix_type="S")
            vc_filt = self._zero_phase_filter(v_patch.clone(), matrix_type="C")

            # pad and fft shift
            vs_shift = self._v_shift(self._v_pad(vs_filt), matrix_type="S")
            vc_shift = self._v_shift(self._v_pad(vc_filt), matrix_type="C")

            # fft
            vs = torch.fft.fft2(vs_shift, dim=(-2, -1))
            vc = torch.fft.fft2(vc_shift, dim=(-2, -1))

            # get operators
            m_op = self._get_m_operator_fft(v_s=vs, v_c=vc, mask=mask)
            b = self._get_b_vector_fft(k=k_in, mask=mask, v_s=vs, v_c=vc)

            # reconstruct
            recon_k_missing, _, _ = cgd(
                func_operator=m_op, x=torch.zeros_like(b), b=b, max_num_iter=20, conv_tol=1e-3
            )

            # embed data
            recon_k = k_in.clone()
            recon_k[~mask] = recon_k_missing

            k_recon[idx_b] = recon_k
        return self._reshape_batches_to_input(k_recon)


def main():
    log_module.info("K Space creation")
    torch.manual_seed(10)
    nx = 256
    ny = 224
    nc = 4
    ne = 2

    rank = 50
    r = 3

    sl_phantom = Phantom.get_shepp_logan(shape=(nx, ny), num_coils=nc, num_echoes=ne)
    k_data = sl_phantom.sub_sample_ac_random_lines(ac_lines=40, acceleration=3)

    in_k = k_data.clone()
    in_k = torch.reshape(in_k, (nx, ny, -1))
    in_img = ifft(in_k, dims=(0, 1))

    ac_loraks = AC_LORAKS(
        k_space_xyzct=k_data.unsqueeze(2), rank=rank, loraks_neighborhood_radius=r, process_slice=True,
        device=torch.device("cpu")
    )
    k_recon = torch.squeeze(ac_loraks.reconstruct())
    k_recon = torch.reshape(k_recon, (nx, ny, -1))

    out_img = ifft(k_recon, dims=(0, 1))

    fig = psub.make_subplots(
        rows=4, cols=out_img.shape[-1]
    )
    for i, d in enumerate([in_k, k_recon, in_img, out_img]):
        d = torch.abs(d)
        if i < 2:
            d = torch.log(d)
        for c in range(out_img.shape[-1]):
            fig.add_trace(
                go.Heatmap(z=d[:, :, c], showscale=False),
                row=i + 1, col=c + 1
            )
    f_name = plib.Path("/data/pt_np-jschmidt/code/PyMRItools/scratches/figures/js_ac_loraks_test.html")
    log_module.info(f"write file: {f_name}")
    fig.write_html(f_name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
