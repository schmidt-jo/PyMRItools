"""
Gibbs unringing algorithm based on doi/10.1002/mrm.26054
Kellner et al. 2015 MRM
_____
Implementation, Jochen Schmidt, 26.09.2024

"""
import pathlib as plib
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as psub
import logging
import tqdm
import torch
from pymritools.utils.funtions import fft
log_module = logging.getLogger(__name__)


def gibbs_unring_1d(
        image_data_1d: np.ndarray,
        m: int = 100, k: int = 3,
        visualize: bool = True,
        gpu: bool = False, gpu_device: int = 0) -> np.ndarray:
    log_module.info("Gibbs un-ring 1D")
    # take last dimension as the dimension to process,
    # combine rest
    shape = image_data_1d.shape
    n1d_data = np.reshape(image_data_1d, (-1, shape[-1]))
    n_dim = n1d_data.shape[0]

    # get number of coefficients and k indices
    n_c_k = shape[-1]
    k_ind = np.arange(-int(n_c_k / 2), int(n_c_k / 2))
    # we build M voxel shifted images
    s = np.arange(-m, m)
    # build array axis
    x = k_ind.copy()
    # dims: x [x], c_k [k], s [2m]
    # we build shifted images
    # want i_s [b, 2m, x, k] # want fft, hence sum over k

    # want to build neighborhoods around each voxel from the left and right without inclusion with size k
    # set neighborhood
    nb_lr = np.array([[-i, i] for i in np.arange(1, k + 1)])
    nb_pts = np.clip(np.arange(n_c_k)[:, None, None] + nb_lr[None, :], 0, n_c_k - 1)

    # we need to batch the data
    batch_size = 200
    log_module.info(f"\t\t - batch data, (batch size: {batch_size})")
    batch_sections = int(np.ceil(n_dim / batch_size))
    batch_idxs = np.array_split(np.arange(n_dim), batch_sections)
    batched_data = np.array_split(n1d_data, batch_sections)
    # allocate space
    ur_data = np.zeros_like(n1d_data, dtype=float)

    if gpu:
        device = torch.device(f"cuda:{gpu_device}")
        to_k_ind = torch.from_numpy(k_ind).to(device)
        to_x = torch.from_numpy(x).to(device)
        to_s = torch.from_numpy(s).to(device)
        to_nb_pts = torch.from_numpy(nb_pts).to(device)
    else:
        device = None
        to_k_ind = None
        to_x = None
        to_s = None
        to_nb_pts = None
    # batch process
    log_module.info(f"\t\t - use gpu: {gpu}, (device: {gpu_device})")

    for idx_batch in tqdm.trange(batched_data.__len__(), desc="Batch Process Data"):
        data_batch = batched_data[idx_batch]
        if gpu:
            data_batch = torch.from_numpy(data_batch).to(device)
            i_unring = batch_process_torch(
                data_batch=data_batch, k_ind=to_k_ind, x=to_x, s=to_s, m=m, nb_pts=to_nb_pts
            )
        else:
            i_unring = batch_process_numpy(
                data_batch=data_batch, k_ind=k_ind, x=x, s=s, m=m, nb_pts=nb_pts
            )
        batch_idx = batch_idxs[idx_batch]
        n_dim_batch = data_batch.shape[0]

        ur_data[batch_idx] = i_unring

        if visualize and idx_batch == 0:
            mid_l = int(n_dim_batch / 2)
            # plot
            fig = go.Figure()
            fig.add_trace(
                go.Scattergl(x=x, y=np.abs(n1d_data[mid_l]), name="initial image")
            )
            fig.add_trace(
                go.Scattergl(x=x, y=np.abs(i_unring[mid_l]), name="unring")
            )
            file_name = path.joinpath("i_ur").with_suffix(".html")
            logging.info(f"Write file: {file_name}")
            fig.write_html(file_name.as_posix())
    image_data_1d = np.reshape(ur_data, shape)
    return image_data_1d


def batch_process_numpy(
        data_batch: np.ndarray, k_ind: np.ndarray,
        x: np.ndarray, s: np.ndarray, m: int, nb_pts: np.ndarray):
    n_dim_batch = data_batch.shape[0]
    n_c_k = data_batch.shape[-1]
    # we want to get the fourier coefficients of the initial image
    c_k = fft(data_batch, inverse=False, axes=(-1,))
    i_s = np.sum(
        c_k[:, None, None, :] * np.exp(
            2 * np.pi * 1j / n_c_k * k_ind[None, None, None, :] * (
                    x[None, None, :, None] + s[None, :, None, None] / (2 * m)
            )
        ),
        axis=-1
    ) / n_c_k

    # build matrix, dim [b, 2m, x, k_nb, 2]
    d_mat = i_s[:, :, nb_pts]
    # now we sum across the abs difference within left and right sided nbs
    d_mat = np.sum(
        np.abs(
            np.diff(d_mat, axis=3)
        ),
        axis=3
    )
    # dim [b, 2m, x, 2]
    # the optimal shifts are found independently for left and right to be minimal oscillation measure d,
    # hence reduce for s
    t_opt = np.argmin(d_mat, axis=1)
    d_mat_min = np.min(d_mat, axis=1)
    # find whether left or right is minimal
    t_opt_lr = np.argmin(d_mat_min, axis=-1)

    # combine batch dims for now
    b_t_opt = np.reshape(t_opt, (-1, 2))
    b_t_opt_lr = t_opt_lr.flatten()
    # find s indices
    b_t_s = b_t_opt[np.arange(b_t_opt.shape[0]), b_t_opt_lr]
    # reshape
    t_opt_s = np.reshape(b_t_s, (n_dim_batch, -1))
    # reduce by this dimension
    # get shifts
    shift_t = s[t_opt_s] / (2 * m)
    # find corresponding shift image and axis
    shift_i_t = np.abs(
        i_s[np.arange(n_dim_batch)[:, None], t_opt_s, np.arange(x.shape[0])[None, :]]
    )

    # interpolate back to grid and fill in original size
    # dims [nd, x] want to find grid points of i at x coming from x + shifts and i_t(x+shifts)
    x_idxs = np.tile(np.arange(n_c_k - 1)[None, :], (n_dim_batch, 1))
    x_idxs[shift_t[:, 0] >= 0] += 1
    nd_idxs = np.tile(np.arange(n_dim_batch)[:, None], (1, n_c_k - 1))

    i_unring = np.zeros_like(shift_t)
    i_unring[nd_idxs, x_idxs] = np.abs(shift_i_t[:, :-1]) + (
            x[x_idxs] - shift_t[:, :-1]
    ) / np.diff(shift_t, axis=-1) * np.abs(np.diff(shift_i_t, axis=-1))
    return i_unring


def batch_process_torch(
        data_batch: torch.tensor, k_ind: torch.tensor,
        x: torch.tensor, s: torch.tensor, m: int, nb_pts: torch.tensor):
    n_dim_batch = data_batch.shape[0]
    n_c_k = data_batch.shape[-1]
    # we want to get the fourier coefficients of the initial image
    c_k = fft(data_batch, inverse=False, axes=(-1,), use_torch=True)
    i_s = torch.sum(
        c_k[:, None, None, :] * torch.exp(
            2 * torch.pi * 1j / n_c_k * k_ind[None, None, None, :] * (
                    x[None, None, :, None] + s[None, :, None, None] / (2 * m)
            )
        ),
        dim=-1
    ) / n_c_k

    # build matrix, dim [b, 2m, x, k_nb, 2]
    d_mat = i_s[:, :, nb_pts]
    # now we sum across the abs difference within left and right sided nbs
    d_mat = torch.sum(
        torch.abs(
            torch.diff(d_mat, dim=3)
        ),
        dim=3
    )
    # dim [b, 2m, x, 2]
    # the optimal shifts are found independently for left and right to be minimal oscillation measure d,
    # hence reduce for s
    d_mat_min, t_opt = torch.min(d_mat, dim=1)
    # find whether left or right is minimal
    _, t_opt_lr = torch.min(d_mat_min, dim=-1)

    # combine batch dims for now
    b_t_opt = torch.reshape(t_opt, (-1, 2))
    b_t_opt_lr = torch.flatten(t_opt_lr)
    # find s indices
    b_t_s = b_t_opt[torch.arange(b_t_opt.shape[0]), b_t_opt_lr]
    # reshape
    t_opt_s = torch.reshape(b_t_s, (n_dim_batch, -1))
    # reduce by this dimension
    # get shifts
    shift_t = s[t_opt_s] / (2 * m)
    # find corresponding shift image and axis
    shift_i_t = torch.abs(
        i_s[torch.arange(n_dim_batch)[:, None], t_opt_s, torch.arange(x.shape[0])[None, :]]
    ).to(dtype=shift_t.dtype)

    # interpolate back to grid and fill in original size
    # we take x + shift as our new grid
    xs_grid = shift_t + x[None, :]
    # we dont need to sort the grid as shifts are going from -0.5 - 0.499 of voxel size,
    # hence its supposed to be sorted
    # however due to the irregularity, it can happen that a regular x does not lie between two adjacent shifts,
    # thus we need to find the left neighbor of all grid points x, we can do this easily by computing the difference
    # between x and xs, if its positive, the shift is already to the left, if not its to the right,
    # first we assume x always between two adjacent xs, just increasing indices
    ind_left_nb = torch.arange(n_c_k)[None, :].expand(n_dim_batch, -1).to(shift_t.device)
    # now compute the difference and set index to one grid point left if above condition given
    ind_left_nb[shift_t > 0] -= 1
    # dont allow mapping outside of tensor
    ind_left_nb = torch.clamp(ind_left_nb, 0, n_c_k - 2)
    # create nd idx array for indexing
    nd_idxs = torch.arange(n_dim_batch)[:, None].expand(-1, n_c_k).to(shift_t.device)

    i_unring = shift_i_t[nd_idxs, ind_left_nb] + torch.nan_to_num(
        torch.divide(
            x[None, :] - xs_grid[nd_idxs, ind_left_nb],
            xs_grid[nd_idxs, ind_left_nb + 1] - xs_grid[nd_idxs, ind_left_nb]
        ), nan=0.0
    ) * (shift_i_t[nd_idxs, ind_left_nb + 1] - shift_i_t[nd_idxs, ind_left_nb])
    return i_unring.detach().cpu().numpy()


def gibbs_unring_nd(
        image_data_nd: np.ndarray,
        m: int = 100, k: int = 3,
        visualize: bool = True,
        gpu: bool = False, gpu_device: int = 0) -> np.ndarray:
    log_module.info(f"Gibbs un-ring multidim")
    # check data shape
    if not 2 <= image_data_nd.shape.__len__() <= 4:
        err = f"implementation covers 2D - 4D data (xy (z) (t)), but found {image_data_nd.shape.__len__()} dims."
        log_module.error(err)
        raise AttributeError(err)
    while image_data_nd.shape.__len__() < 4:
        image_data_nd = np.expand_dims(image_data_nd, -1)
    # save dims
    nx, ny, nz, nt = image_data_nd.shape

    # introduce weighting filter
    k_x = np.linspace(-np.pi, np.pi, nx)
    k_y = np.linspace(-np.pi, np.pi, ny)

    log_module.info(f"\t\t - calc weighting filters")

    denom = 2 + np.cos(k_y[None, :]) + np.cos(k_x[:, None])
    g_x = np.divide(
        1 + np.cos(k_y[None, :]),
        denom,
        where=denom > 1e-9, out=np.zeros_like(denom)
    )
    g_y = np.divide(
        1 + np.cos(k_x[:, None]),
        denom,
        where=denom > 1e-9, out=np.zeros_like(denom)
    )
    if visualize:
        # plot
        fig = psub.make_subplots(
            rows=1, cols=2,
            shared_xaxes=True, shared_yaxes=True,
            column_titles=["G<sub>x</sub>", "G<sub>y</sub>"]
        )
        fig.add_trace(
            go.Heatmap(z=g_x, colorscale="Magma", transpose=True),
            row=1, col=1
        )
        fig.add_trace(
            go.Heatmap(z=g_y, colorscale="Magma", transpose=True),
            row=1, col=2
        )
        file_name = path.joinpath("g_xy").with_suffix(".html")
        logging.info(f"Write file: {file_name}")
        fig.write_html(file_name.as_posix())

    # move nd dims to front
    nd_data = np.moveaxis(image_data_nd, (2, 3), (0, 1))
    # combine batches
    nd_data = np.reshape(nd_data, (-1, nx, ny))
    nd_dim = nd_data.shape[0]

    log_module.info(f"\t\t - filter image")
    # create modified filter images
    i_x = fft(fft(nd_data, inverse=False, axes=(-2, -1)) * g_x[None, :, :], inverse=True, axes=(-2, -1))
    i_y = fft(fft(nd_data, inverse=False, axes=(-2, -1)) * g_y[None, :, :], inverse=True, axes=(-2, -1))

    log_module.info(f"\t\t --> process ix ____")
    # process x dim of ix
    ur_ix = gibbs_unring_1d(
        image_data_1d=np.moveaxis(i_x, 1, -1), m=m, k=k, visualize=visualize,
        gpu=gpu, gpu_device=gpu_device
    )
    ur_ix = np.moveaxis(ur_ix, -1, 1)

    log_module.info(f"\t\t --> process iy ____")
    # process y dim of iy
    ur_iy = gibbs_unring_1d(
        image_data_1d=i_y, m=m, k=k, visualize=visualize,
        gpu=gpu, gpu_device=gpu_device
    )

    # average
    ur_i = (ur_ix + ur_iy)
    # reshape
    ur_i = np.reshape(ur_i, (nz, nt, nx, ny))
    # move batch dims back
    ur_i = np.moveaxis(ur_i, (0, 1), (2, 3))
    log_module.info(f"\t\t - done!")
    return ur_i


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # prepare working path
    path = plib.Path("/data/pt_np-jschmidt/code/gibun/dev")
    path.mkdir(exist_ok=True, parents=True)

    # load in data
    path_to_phantom_data = plib.Path(
        "/data/pt_np-jschmidt/data/00_phantom_scan_data/semc_2024-04-19/processed/semc/semc_r0p7_echo_mag.nii.gz"
    )
    log_module.info(f"load file: {path_to_phantom_data}")
    phantom_img = nib.load(path_to_phantom_data)
    phantom_data = phantom_img.get_fdata()

    phantom_unring = gibbs_unring_nd(image_data_nd=phantom_data, visualize=False, gpu=True)

    # save data
    path_to_save = path_to_phantom_data.with_stem(f"ur_{path_to_phantom_data.stem}").with_suffix(".nii")
    log_module.info(f"save file: {path_to_save}")
    img = nib.Nifti1Image(phantom_unring, phantom_img.affine)
    nib.save(img, path_to_save.as_posix())
