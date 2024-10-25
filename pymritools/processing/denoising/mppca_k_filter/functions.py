import logging
import pathlib as plib

import tqdm
import numpy as np
import torch
import plotly.colors as plc
import plotly.graph_objects as go

log_module = logging.getLogger(__name__)

def misc():
    ext_k_sampling_mask = np.tile(k_sampling_mask[:, :, None, None, :], (1, 1, *k_space.shape[2:4], 1))

    filter_input = np.reshape(
        k_space[ext_k_sampling_mask],
        (k_space.shape[0], -1, *k_space.shape[2:])
    )
    k_space_filt = np.zeros_like(k_space)
    filtered_input = matched_filter_noise_removal(noise_data=noise_scans, k_space_lines=filter_input)
    k_space_filt[ext_k_sampling_mask] = filtered_input.flatten()


def distribution_mp(
        x: int | float | torch.Tensor, sigma: int | float | torch.Tensor, gamma: int | float) -> torch.Tensor:
    """
    Marchenko-Pastur distribution given sigma and gamma, calculated for some input value or axes x.
    Can cast for x dim [c, d, ...] and sigma [a, b, ...] to [a, b, ..., c, d, ...]. Gamma is assumed to be scalar.
    :param x: Input tensor or scalar value. Could be of type int, float, or a torch.Tensor.
    :param sigma: Scale parameter tensor or scalar value. Could be of type int, float, or a torch.Tensor.
    :param gamma: Scalar value for the shape parameter of the distribution. Must be an int or float.
    :return: A torch.Tensor representing the calculated distribution's probability densities.
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma)

    # allocate output
    shape_sig = sigma.shape
    shape_x = x.shape
    shape = (*shape_sig, *shape_x)
    result = torch.zeros(shape)

    # cast to shape of sigma and x
    sigma = sigma.unsqueeze(-1).expand(*shape_sig, *shape_x)
    x = x.unsqueeze(0).expand(*shape_sig, *shape_x)

    # calculate lambda boundaries
    lam_p = sigma ** 2 * (1 + np.sqrt(gamma)) ** 2
    lam_m = sigma ** 2 * (1 - np.sqrt(gamma)) ** 2

    # build mask
    mask = (lam_m < x) & (x < lam_p)
    # fill probabilities
    result[mask] =  (
            torch.sqrt(
                (lam_p - x) * (x - lam_m)
            ) / (2 * torch.pi * gamma * x * sigma**2)
    )[mask]
    result /= torch.sum(result)
    return torch.squeeze(result)


def find_approx_squared_matrix_form(shape_1d: tuple | int):
    """
    small helper that gives the side length of one side of a matrix closest to squared form of 1D input.
    Can be used to shuffle some 1D input into a 2D matrix that is as unskewed as possible using reshape((output, -1))

    :param shape_1d: some shape, if more than 1D, using the last dimension
    :return:
    """
    if isinstance(shape_1d, int):
        shape_1d = (shape_1d,)
    a = int(np.ceil(np.sqrt(shape_1d[-1])))
    for i in range(a):
        if shape_1d[-1] % a == 0:
            break
        a -= 1
    return a


def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    Interpolate a function fp at points xp in a multidimensional context

    Parameters:
    x (torch.Tensor): Tensor of the new sampling points with shape [batch, a, b]
    xp (torch.Tensor): 1D Tensor of original sample points with shape [c]
    fp (torch.Tensor): 2D Tensor of function values at xp with shape [a, c]

    Returns:
    torch.Tensor: Interpolated values with shape [batch, a, b]
    """
    batch, a, b = x.shape
    # find closest upper adjacent indices of x in xp, then the next lower one
    indices = torch.searchsorted(xp, x.view(-1, b))
    indices = torch.clamp(indices, 1, xp.shape[0] - 1)
    # find adjacent left and right points on originally sampled axes xp
    x0 = xp[indices - 1]
    x1 = xp[indices]
    # find values of originally sampled function considering its differing for each idx_a
    fp_expanded = fp.unsqueeze(0).expand(batch, -1, -1)
    y0 = fp_expanded.gather(2, indices.view(batch, a, b) - 1)
    y1 = fp_expanded.gather(2, indices.view(batch, a, b))
    # get the slope
    slope = (y1 - y0) / (x1 - x0).view(batch, a, b)
    interpolated_values = slope * (x - x0.view(batch, a, b)) + y0
    return interpolated_values


def matched_filter_noise_removal(
        noise_data_n_ch_samp: np.ndarray | torch.Tensor, k_space_lines_read_ph_sli_ch_t: np.ndarray | torch.Tensor,
        noise_histogram_depth: int = 100):
    """
    Function to filter noise using the PCA of readout lines / samples.
    First this function extracts a noise MP distribution from noise scans,
    then it looks for this distribution in sampled k-space lines after rearranging them into blocks and
    doing a SVD. The singular values identified to lay within the noise spectrum of the noise scans are down-weighted.
    The line is then reconstructed with adopted singular values.
    It is coded such that it does the noise characterization per channel.
    Channels could however be treated as approximately identical regarding noise.
    This way even fewer noise scans can suffice to use the approach.
    """
    log_module.info(f"pca denoising matched filter")
    # get to pytorch tensors
    if not torch.is_tensor(noise_data_n_ch_samp):
        noise_data_n_ch_samp = torch.from_numpy(noise_data_n_ch_samp)
    if not torch.is_tensor(k_space_lines_read_ph_sli_ch_t):
        k_space_lines_read_ph_sli_ch_t = torch.from_numpy(k_space_lines_read_ph_sli_ch_t)

    # we got noise data, assumed dims [num_noise_scans, num_channels, num_samples]
    # want to use this to calculate a np distribution of noise singular values per channel
    # start rearranging, channels to front, combine num scans
    noise_data_n_ch_samp = np.moveaxis(noise_data_n_ch_samp, 0, -1)
    noise_data_n_ch_samp = np.reshape(noise_data_n_ch_samp, (noise_data_n_ch_samp.shape[0], -1))
    shape = noise_data_n_ch_samp.shape
    # should be dims [channels, num_samples * num_scans]

    # want to make the sampled lines as square as possible
    matrix_short_side = find_approx_squared_matrix_form(shape)
    # reshape again - spread the last dim aka the line into a approx square matrix
    noise_data_n_ch_samp = np.reshape(noise_data_n_ch_samp, (shape[0], matrix_short_side, -1))
    m = matrix_short_side
    n = noise_data_n_ch_samp.shape[-1]

    # calculate singular values of noise distribution across all channels - dims [channels, m]
    noise_s = torch.linalg.svdvals(noise_data_n_ch_samp)
    # get eigenvalues
    s_lam = noise_s ** 2 / n
    # take some percentage bigger than biggest eigenvalue
    noise_s_max = 1.2 * torch.max(s_lam)
    # build some axis for which we calculate the mp - distribution
    noise_ax = torch.linspace(0, noise_s_max, noise_histogram_depth)

    gamma = m / n
    # get biggest and lowest s to approximate noise characteristics from
    sigma = torch.sqrt(
        (torch.max(s_lam, dim=1).values - torch.min(s_lam, dim=1).values) / 4 / np.sqrt(gamma)
    )

    # get mp distribution of noise values for all channels
    p_noise = distribution_mp(noise_ax, sigma, gamma)
    p_noise /= torch.sum(p_noise, dim=len(sigma.shape), keepdim=True)

    # We want to convert this distribution to a weighting.
    # ie: the more probable singular values of our data are in this distribution,
    # the more we want to threshold them
    # do some adjustments to convert to weighting factor. This needs some testing of optimality!
    p_noise_w = torch.clamp(
        p_noise / torch.max(p_noise, dim=len(sigma.shape), keepdim=True),
        0, 0.25
    )
    # scale back to 1
    p_noise_w /= torch.max(p_noise_w, dim=len(sigma.shape), keepdim=True)
    # fill in ones in front
    p_noise_w[:, :int(noise_histogram_depth / 10)] = 1

    # invert distribution to create weighting
    p_weight = 1 - p_noise_w / torch.max(p_noise_w, dim=len(sigma.shape), keepdim=True).values
    p_weight_ax = noise_ax

    colors = plc.sample_colorscale("Turbo", np.linspace(0.1, 0.9, p_weight.shape[0]))
    # quick testing visuals
    fig = go.Figure()
    for idx_c, p in enumerate(p_weight):
        fig.add_trace(
            go.Scattergl(
                x=noise_ax, y=p_noise[idx_c],
                marker=dict(color=colors[idx_c]), name=f"channel-{idx_c}",
                showlegend=True
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=noise_ax, y=p.numpy(),
                marker=dict(color=colors[idx_c]),
                name=f"weighting function, ch-{idx_c}", showlegend=False
            )
        )
    fig_path = plib.Path("./examples/raw_data/rd_file/figs/").absolute()
    path = fig_path.joinpath("noise_dist_weighting_per_channel").with_suffix(".html")
    log_module.info(f"write file: {path}")
    fig.write_html(path.as_posix())

    # we want a svd across a 2d matrix spanned by readout line samples
    # the rational is that basically over all matrix entries there could only be a smooth signal sampled,
    # hence a low rank representation of the matrix should exist.
    # At the same time we know the exact noise mp-distribution for such matrices spanned by
    # adjacent samples from the noise scans (just calculated)
    # we aim at removing this part of the singular values

    # rearrange k-space, get a batch dimension and channels separated [batch dims..., num_channels, num_samples readout]
    k_space_lines_read_ph_sli_ch_t = torch.swapdims(k_space_lines_read_ph_sli_ch_t, 0, -1)
    shape = k_space_lines_read_ph_sli_ch_t.shape
    # flatten batch dims [batch dim, num_channels, num_samples readout]
    k_space_lines_read_ph_sli_ch_t = torch.reshape(k_space_lines_read_ph_sli_ch_t, (-1, *shape[-2:]))
    # find once again close to square form of the last dimension
    a = find_approx_squared_matrix_form(shape)
    # and build matrix from line
    k_space_lines_read_ph_sli_ch_t = torch.reshape(
        k_space_lines_read_ph_sli_ch_t,
        (*k_space_lines_read_ph_sli_ch_t.shape[:2], a, -1)
    )
    # save matrix dimensions
    m, n = k_space_lines_read_ph_sli_ch_t.shape[-2:]

    # allocate output space
    k_space_filt = torch.zeros_like(k_space_lines_read_ph_sli_ch_t)

    # batch svd
    batch_size = 200
    num_batches = int(np.ceil(k_space_lines_read_ph_sli_ch_t.shape[0] / batch_size))
    # using gpu - test how much we can put there and how it scales for speed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for idx_b in tqdm.trange(num_batches, desc="filter svd"):
    # for idx_b in tqdm.trange(5, desc="filter svd"):
        start = idx_b * batch_size
        end = min((idx_b + 1) * batch_size, k_space_lines_read_ph_sli_ch_t.shape[0])
        # send batch to GPU
        batch = k_space_lines_read_ph_sli_ch_t[start:end].to(device)
        # svd batch
        u, s, v = torch.linalg.svd(
            batch,
            full_matrices=False
        )
        # process batch (can do straight away with tensor on device)
        # get eigenvalue from singular values
        bs_eigv = s**2 / n
        # can calculate the weighting for the whole batched singular values at once
        weighting = interpolate(bs_eigv, xp=p_weight_ax, fp=p_weight)
        # we could also try original idea: make histogram and find / correlate the noise
        # MP-distribution in the histogram, then remove, for now we just weight

        # weight original singular values (not eigenvalues)
        s_w = s * weighting
        # reconstruct signal with filtered singular values, dims [batch, channel, m]
        signal_filt = torch.matmul(
            torch.einsum("iklm, ikm -> iklm", u, s_w.to(u.dtype)),
            v)
        # normalize to previous signal levels - taking maximum of absolute value across channels
        # assign and move off GPU
        k_space_filt[start:end] = signal_filt.cpu()

        if idx_b % 10 == 0:
            # quick visuals for reference
            noisy_line = batch[0].reshape((batch.shape[1], -1)).cpu().numpy()
            noisy_line_filt = k_space_filt[start].reshape((batch.shape[1], -1))
            fig = go.Figure()
            for idx_c in [5, 15]:
                fig.add_trace(
                    go.Scattergl(y=np.abs(noisy_line[idx_c]), name=f"ch-{idx_c}, noisy line mag")
                )
                fig.add_trace(
                    go.Scattergl(y=np.abs(noisy_line_filt[idx_c]), name=f"ch-{idx_c}, noisy line mag filtered")
                )
            path = fig_path.joinpath(f"line_batch-{idx_b}_noise_filterin").with_suffix(".html")
            log_module.info(f"write file: {path}")
            fig.write_html(path.as_posix())

            fig = go.Figure()
            for idx_c in [5, 15]:
                fig.add_trace(
                    go.Scattergl(y=weighting[0, idx_c], name=f"ch-{idx_c}, noisy line mag")
                )
            path = fig_path.joinpath(f"weighting_batch-{idx_b}_noise_filtering").with_suffix(".html")
            log_module.info(f"write file: {path}")
            fig.write_html(path.as_posix())

    # reshape - get matrix shuffled line back to the an actual 1D line
    k_space_filt = np.reshape(k_space_filt, (*k_space_filt.shape[:-2], -1))
    # deflate batch dimensions
    k_space_filt = np.reshape(k_space_filt, shape)
    # move read dimension back to front
    k_space_filt = np.swapaxes(k_space_filt, -1, 0)
    return k_space_filt

