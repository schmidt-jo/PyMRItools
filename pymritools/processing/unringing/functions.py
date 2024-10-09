"""
Functions for unringing module.
Algorithm based on doi/10.1002/mrm.26054
Kellner et al. 2015 MRM
"""
import torch
import numpy as np
import logging
import tqdm
from pymritools.utils.funtions import fft

log_module = logging.getLogger(__name__)


def batch_process_torch(
        data_batch: torch.Tensor, k_ind: torch.Tensor,
        x: torch.Tensor, s: torch.Tensor, m: int, nb_pts: torch.Tensor):
    # get size of batch dimension
    n_dim_batch = data_batch.shape[0]
    n_c_k = data_batch.shape[-1]
    # we want to get the fourier coefficients of the initial image
    c_k = fft(data_batch, img_to_k=False, axes=(-1,))
    # compute shifted images
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
    return i_unring


def gibbs_unring_1d(
        image_data_1d: torch.Tensor,
        m: int = 100, k: int = 3,
        visualize: bool = True,
        gpu: bool = False, gpu_device: int = 0) -> torch.Tensor:
    log_module.info("Gibbs un-ring 1D")

    if gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_device}")
    else:
        device = torch.device("cpu")

    # take last dimension as the dimension to process, combine rest
    shape = image_data_1d.shape
    n1d_data = torch.reshape(image_data_1d, (-1, shape[-1]))
    n_dim = n1d_data.shape[0]

    # get number of coefficients and k indices
    n_c_k = shape[-1]
    k_ind = torch.arange(-int(n_c_k / 2), int(n_c_k / 2)).to(device)

    # we build M voxel shifted images
    s = torch.arange(-m, m).to(device)
    # build array axis
    x = k_ind.clone().to(device)
    # dims: x [x], c_k [k], s [2m]
    # we build shifted images
    # want i_s [b, 2m, x, k] # want fft, hence sum over k

    # want to build neighborhoods around each voxel from the left and right without inclusion with size k
    # set neighborhood
    nb_lr = torch.tensor([[-i, i] for i in np.arange(1, k + 1)])
    nb_pts = torch.clip(torch.arange(n_c_k)[:, None, None] + nb_lr[None, :], 0, n_c_k - 1).to(device)

    # we need to batch the data
    batch_size = 200
    log_module.info(f"\t\t- batch data, (batch size: {batch_size})")
    num_batches = int(np.ceil(n_dim / batch_size))

    # allocate space
    ur_data = torch.zeros_like(n1d_data)

    # batch process
    log_module.info(f"\t\t- set gpu: {gpu}, (available device: {gpu_device})")

    for idx_batch in tqdm.trange(num_batches, desc="Batch Process Data"):
        start = idx_batch * batch_size
        end = np.min([n_dim, start + batch_size])
        i_unring = batch_process_torch(
                data_batch=n1d_data[start:end], k_ind=k_ind, x=x, s=s, m=m, nb_pts=nb_pts
        )
        ur_data[start:end] = i_unring

        # if visualize and idx_batch == 0:
        #     mid_l = int(n_dim_batch / 2)
        #     # plot
        #     fig = go.Figure()
        #     fig.add_trace(
        #         go.Scattergl(x=x, y=np.abs(n1d_data[mid_l]), name="initial image")
        #     )
        #     fig.add_trace(
        #         go.Scattergl(x=x, y=np.abs(i_unring[mid_l]), name="unring")
        #     )
        #     file_name = path.joinpath("i_ur").with_suffix(".html")
        #     logging.info(f"Write file: {file_name}")
        #     fig.write_html(file_name.as_posix())
    image_data_1d = torch.reshape(ur_data, shape)
    return image_data_1d


def gibbs_unring_nd(
        image_data_nd: torch.Tensor,
        m: int = 100, k: int = 3,
        visualize: bool = True,
        gpu: bool = False, gpu_device: int = 0) -> torch.Tensor:
    """
    Multidimensional Gibbs unringing function.

    This function applies the Gibbs unringing algorithm to multidimensional image data to reduce ringing artifacts.
    The algorithm is based on the paper by Kellner et al. (2015).

    Parameters:
        image_data_nd (torch.Tensor):
            The input image data. Expected dimensions are 2 to 4 (xy (z) (t)).
        m (int, optional): 
            The number of shifts per voxel. Default is 100.
        k (int, optional): 
            The size of the neighborhood around each voxel for which to find the optimal image. Default is 3.
        visualize (bool, optional): 
            Whether to visualize intermediate results or not. Default is True.
        gpu (bool, optional): 
            Whether to use GPU acceleration if available. Default is False.
        gpu_device (int, optional): 
            The GPU device index to use if GPU acceleration is enabled. Default is 0.

    Returns:
        torch.Tensor: The unringed image data with the same shape as the input.
    """
    log_module.info(f"Gibbs un-ring multidim")
    # check data shape
    if not 2 <= image_data_nd.shape.__len__() <= 4:
        err = f"implementation covers 2D - 4D data (xy (z) (t)), but found {image_data_nd.shape.__len__()} dims."
        log_module.error(err)
        raise AttributeError(err)
    while image_data_nd.shape.__len__() < 4:
        image_data_nd = np.expand_dims(image_data_nd, -1)

    # ensure tensor
    if not torch.is_tensor(image_data_nd):
        image_data_nd = torch.from_numpy(image_data_nd)
    # save dims
    nx, ny, nz, nt = image_data_nd.shape

    # setup device
    if gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_device}")
    else:
        device = torch.device("cpu")
    log_module.info(f"\t\t- set device: {device}")
    # introduce weighting filter
    k_x = torch.linspace(-np.pi, np.pi, nx).to(device)
    k_y = torch.linspace(-np.pi, np.pi, ny).to(device)

    log_module.info(f"\t\t - calc weighting filters")

    denom = 2 + torch.cos(k_y[None, :]) + torch.cos(k_x[:, None])
    g_x = torch.nan_to_num(
        torch.divide(
            1 + torch.cos(k_y[None, :]),
            denom,
        ),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    g_y = torch.nan_to_num(
        torch.divide(
            1 + torch.cos(k_x[:, None]),
            denom,
        ),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    # if visualize:
    #     # plot
    #     fig = psub.make_subplots(
    #         rows=1, cols=2,
    #         shared_xaxes=True, shared_yaxes=True,
    #         column_titles=["G<sub>x</sub>", "G<sub>y</sub>"]
    #     )
    #     fig.add_trace(
    #         go.Heatmap(z=g_x, colorscale="Magma", transpose=True),
    #         row=1, col=1
    #     )
    #     fig.add_trace(
    #         go.Heatmap(z=g_y, colorscale="Magma", transpose=True),
    #         row=1, col=2
    #     )
    #     file_name = path.joinpath("g_xy").with_suffix(".html")
    #     logging.info(f"Write file: {file_name}")
    #     fig.write_html(file_name.as_posix())

    # move nd dims to front
    nd_data = torch.movedim(image_data_nd, (2, 3), (0, 1)).to(device)
    # combine batches
    nd_data = torch.reshape(nd_data, (-1, nx, ny))
    nd_dim = nd_data.shape[0]

    log_module.info(f"\t\t - filter image")
    # create modified filter images
    i_x = fft(fft(nd_data, img_to_k=False, axes=(-2, -1)) * g_x[None, :, :], img_to_k=True, axes=(-2, -1))
    i_y = fft(fft(nd_data, img_to_k=False, axes=(-2, -1)) * g_y[None, :, :], img_to_k=True, axes=(-2, -1))

    log_module.info(f"\t\t --> process ix ____")
    # process x dim of ix
    ur_ix = gibbs_unring_1d(
        image_data_1d=torch.movedim(i_x, 1, -1), m=m, k=k, visualize=visualize,
        gpu=gpu, gpu_device=gpu_device
    )
    ur_ix = torch.movedim(ur_ix, -1, 1)

    log_module.info(f"\t\t --> process iy ____")
    # process y dim of iy
    ur_iy = gibbs_unring_1d(
        image_data_1d=i_y, m=m, k=k, visualize=visualize,
        gpu=gpu, gpu_device=gpu_device
    )

    # average
    ur_i = (ur_ix + ur_iy)
    # reshape
    ur_i = torch.reshape(ur_i, (nz, nt, nx, ny))
    # move batch dims back
    ur_i = torch.movedim(ur_i, (0, 1), (2, 3))
    log_module.info(f"\t\t - done!")
    return ur_i
