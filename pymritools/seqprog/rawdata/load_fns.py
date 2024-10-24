import logging
import pathlib as plib

import torch
from scipy.spatial.transform import Rotation
import numpy as np
import polars as pl
import tqdm
import plotly.graph_objects as go
import plotly.colors as plc

from twixtools.geometry import Geometry
from twixtools.mdb import Mdb
from pymritools.config.seqprog import Sampling, PulseqParameters2D
from pymritools.seqprog.rawdata.utils import remove_oversampling

log_module = logging.getLogger(__name__)


def distribution_mp(x, sigma, gamma):
    # allocate output
    shape_sig = sigma.shape
    shape_x = x.shape
    shape = (*shape_sig, *shape_x)
    result = np.zeros(shape)
    # cast to shape of sigma and x

    sigma = np.tile(
        np.expand_dims(sigma, (-np.arange(1, len(shape_x) + 1)).tolist()),
        (*np.ones(len(shape_sig), dtype=int).tolist(), *shape_x)
    )
    # if we have more than one sigma dimension, we want to mask the axis x for each of the sigmas
    x = np.tile(
        np.expand_dims(x, np.arange(len(shape_sig)).tolist()),
        (*shape_sig, *np.ones(len(shape_x), dtype=int).tolist())
    )
    # calculate lambda boundaries
    lam_p = sigma**2 * (1 + np.sqrt(gamma))**2
    lam_m = sigma**2 * (1 - np.sqrt(gamma))**2

    # build mask
    mask = (lam_m < x) & (x < lam_p)
    # fill probabilities
    result[mask] =  (np.sqrt((lam_p - x) * (x - lam_m)) / (2 * np.pi * gamma * x * sigma**2))[mask]
    result /= np.sum(result)
    return np.squeeze(result)


def get_k_space_indices_from_trajectory_grid(trajectory: np.ndarray, n_read: int, os_factor: int):
    """
    Calculate the closest points on grid from a trajectory array (ranging from -0.5, to 0.5).
    We assume the total number of samples is given by n_read * os_factor.
    We shift the samples onto the next available position on the grid by rounding.
    Theres a check for multiple counts of the same position.
    The function returns the index positions.
    """
    # calculate the closest points on grid from trajectory. get positions
    # We could use gridding algorithms here in case we have no close grid positions.
    # I.e. for fancy sampling or use of field probes
    traj_pos = (0.5 + trajectory) * n_read * os_factor
    # just round to the nearest point in 1d
    traj_pos = np.round(traj_pos).astype(int)
    # find indices of points that are actually within the k-space in case the trajectory maps too far out
    traj_valid_index = np.squeeze(
        np.nonzero(
            (traj_pos >= 0) & (traj_pos < os_factor * n_read)
        )
    )
    # find positions of those indices
    traj_valid_pos = traj_pos[traj_valid_index].astype(int)
    indices_in_k_sampling_img = (traj_valid_pos[::os_factor] / os_factor).astype(int)
    return traj_valid_pos, indices_in_k_sampling_img


def get_whitening_matrix(noise_data_n_samples_channel):
    """
    Calculate pre-whitening matrix for channel decorrelation.
    Input all noise data with dimensions [batch, num_samples_adc, num_channels].
    """
    # whiten data, psi_l_inv dim [nch, nch], k-space dims [nfe, npe, nsli, nch, nechos]
    log_module.info("noise decorrelation")
    acq_type = "noise_scan"
    if noise_data_n_samples_channel.shape.__len__() < 2:
        err = (f"Need at least one noise scan with multiple ADC samples for all channels. "
               f"But found noise data shape: ({noise_data_n_samples_channel.shape})!")
        log_module.error(err)
        raise AttributeError(err)
    if noise_data_n_samples_channel.shape.__len__() > 3:
        msg = ("\t\t - noise covariance calculation only implemented for max 3D arrays of "
               "[num-noise scans, num scanned samples, num channels]. But got > 3D array")
        log_module.error(msg)
        raise AttributeError(msg)
    while noise_data_n_samples_channel.shape.__len__() < 3:
        # if less than 3D assume only one noise scan [num_samples, num_channels]
        noise_data_n_samples_channel = noise_data_n_samples_channel[None]
    if noise_data_n_samples_channel.shape[-1] > 100 > noise_data_n_samples_channel.shape[-2]:
        log_module.warning(
            f"found noise data shape: {noise_data_n_samples_channel.shape}, "
            f"but assume dimensions [batch, samples, channels]. "
            f"Try to swap last two dimensions."
        )
        noise_data_n_samples_channel = np.swapaxes(noise_data_n_samples_channel, -1, -2)

    # get covariance matrix for all scans, i.e. noise dimension
    # we can calculate the batched covariance matrix for all individual noise channels
    bcov = np.einsum(
        'ijk, ijl -> ikl',
        noise_data_n_samples_channel,
        noise_data_n_samples_channel.conj(),
        optimize=True
    ) / noise_data_n_samples_channel.shape[1]

    # we average over the number of scans
    psi = np.mean(bcov, axis=0)

    psi_l = np.linalg.cholesky(psi)
    if not np.allclose(psi, np.dot(psi_l, psi_l.T.conj())):
        # verify that L * L.H = A
        err = "cholesky decomposition error"
        logging.error(err)
        raise AssertionError(err)
    return np.linalg.inv(psi_l)


def get_affine(
        geom: Geometry,
        voxel_sizes_mm: np.ndarray,
        fov_mm: np.ndarray,
        slice_gap_mm: float = 0.0):
    # handle rotations
    # get rotation matrix from geom
    rot_mat = Rotation.from_matrix(geom.rotmatrix)
    # handle in plane rotations
    # get normal vector of slab / slice
    normal_vec = np.array(geom.normal)
    # get rotation around plane
    inp_rot_mat = Rotation.from_rotvec(geom.inplane_rot * normal_vec, degrees=False)
    # update affine with rotation part, calculate whole rotation
    affine_rot = (inp_rot_mat * rot_mat).as_matrix()

    # set zoom
    scaling_mat = np.diag(voxel_sizes_mm)
    # add gap
    scaling_mat[-1, -1] += slice_gap_mm
    # update affine
    affine_rot_zoom = np.matmul(scaling_mat, affine_rot)

    # add offset
    # we have for one the center offset saved in geometry
    center_offset = np.array(geom.offset)
    # additionally, we have the origin to be defined as left-posterior-inferior position of the data array
    # we can get this via the fov, set the origin to lpi corner, we need to make up for the asymmetric k-space center
    voxel_offset = - fov_mm / 2
    # if the normal vector of the slice plane is aligned with z dimension, this is the offset already.
    # if deviating from the z direction we can calculate the offset shift by applying a rotation that would yield
    # the normal vector, if starting from the e_z basis
    # for this we calculate the vector perpendicular to e_z and the normal vector and the angle between them
    # around which we need to rotate
    if np.linalg.norm(normal_vec - np.array([0, 0, 1])) > 1e-5:
        normal_perp = np.cross(np.array([0, 0, 1]), normal_vec)
        perp_norm = np.linalg.norm(normal_perp)
        # since both vectors are normalized to 1, we get the angle from the sin
        perp_rot_angle = np.arcsin(perp_norm)
        # normalize the rotation vector
        normal_perp = np.abs(normal_perp / perp_norm)
        # we get the rotation matrix
        perp_rot_mat = Rotation.from_rotvec(perp_rot_angle * normal_perp)
    else:
        # just identity matrix
        perp_rot_mat = Rotation.from_matrix(np.eye(3))
    # apply rotation to voxel origin vector
    voxel_offset = (inp_rot_mat * perp_rot_mat).apply(voxel_offset)
    # add both offsets
    offset = voxel_offset + center_offset

    # construct affine
    aff_matrix = np.zeros((4, 4))
    aff_matrix[:3, :3] = affine_rot_zoom
    aff_matrix[:-1, -1] = offset
    aff_matrix[-1, -1] = 1.0

    # experimentally we found adjustments necessary, data seems to be reoriented after rotation as compared to RAS+
    aff_matrix = np.matmul(np.diag([-1, 1, 1, 1]), aff_matrix)
    aff_matrix[:3, :3] = aff_matrix[:3, :3].T

    return aff_matrix


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
        noise_data: np.ndarray, k_space_lines: np.ndarray, hist_depth: int = 20,
        noise_hist_depth: int = 100):
    log_module.info(f"pca denoising matched filter")
    # we got noise data, assumed dims [num_noise_scans, num_channels, num_samples]
    # want to use this to calculate a np distribution of noise singular values per channel
    # start rearranging, channels to front, combine num scans
    noise_data = np.moveaxis(noise_data, 0, -1)
    noise_data = np.reshape(noise_data, (noise_data.shape[0], -1))
    shape = noise_data.shape
    # should be dims [channels, num_samples * num_scans]

    # want to make the sampled lines as square as possible
    a = find_approx_squared_form(shape)

    # reshape again - spread the last dim aka the line into a approx square matrix
    noise_data = np.reshape(noise_data, (shape[0], a, -1))
    m = a
    n = noise_data.shape[-1]

    # calculate singular values of noise distribution across all channels - dims [channels, m]
    noise_s = np.linalg.svdvals(noise_data)
    # get eigenvalues
    s_lam = noise_s ** 2 / n
    noise_s_max = 1.2 * np.max(s_lam)
    noise_ax = np.linspace(0, noise_s_max, noise_hist_depth)

    gamma = m / n
    # get biggest and lowest s
    sigma = np.sqrt((np.max(s_lam, axis=1) - np.min(s_lam, axis=1)) / 4 / np.sqrt(gamma))

    # get mp distribution of noise values for all channels
    p_noise = distribution_mp(noise_ax, sigma, gamma)
    p_noise /= np.sum(p_noise, axis=len(sigma.shape), keepdims=True)
    # do some adjustments to convert to weighting
    p_noise_w = np.clip(p_noise / np.max(p_noise, axis=len(sigma.shape), keepdims=True), 0, 0.5)
    p_noise_w /= np.max(p_noise_w, axis=len(sigma.shape), keepdims=True)
    p_noise_w[:, :5] = 1
    p_noise_w = torch.from_numpy(p_noise_w)

    # invert distribution to create weighting
    p_weight = 1 - p_noise_w / torch.max(p_noise_w, dim=len(sigma.shape), keepdim=True).values
    p_weight_ax = torch.from_numpy(noise_ax)

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
    k_space_lines = np.swapaxes(k_space_lines, 0, -1)
    shape = k_space_lines.shape
    # flatten batch dims [batch dim, num_channels, num_samples readout]
    k_space_lines = np.reshape(k_space_lines, (-1, *shape[-2:]))
    # find once again close to square form of the last dimension
    a = find_approx_squared_form(shape)
    # and build matrix from line
    k_space_lines = np.reshape(k_space_lines, (*k_space_lines.shape[:2], a, -1))
    # save matrix dimensions
    m, n = k_space_lines.shape[-2:]

    # allocate output space
    k_space_filt = np.zeros_like(k_space_lines)

    # batch svd
    batch_size = 200
    num_batches = int(np.ceil(k_space_lines.shape[0] / batch_size))
    # using gpu - test how much we can put there and how it scales for speed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for idx_b in tqdm.trange(num_batches, desc="filter svd"):
    # for idx_b in tqdm.trange(5, desc="filter svd"):
        start = idx_b * batch_size
        end = min((idx_b + 1) * batch_size, k_space_lines.shape[0])
        # send batch to GPU
        batch = torch.from_numpy(k_space_lines[start:end]).to(device)
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
        signal_filt = torch.matmul(torch.einsum("iklm, ikm -> iklm", u, s_w.to(u.dtype)), v)
        # normalize to previous signal levels - taking maximum of absolute value across channels
        # assign and move off GPU
        k_space_filt[start:end] = signal_filt.cpu().numpy()

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


def find_approx_squared_form(shape_1d: tuple):
    a = int(np.ceil(np.sqrt(shape_1d[-1])))
    for i in range(a):
        if shape_1d[-1] % a == 0:
            break
        a -= 1
    return a


def load_pulseq_rd(
        pulseq_config: PulseqParameters2D, sampling_config: Sampling,
        data_mdbs: list[Mdb], geometry: Geometry, hdr: dict,
        device: torch.device = torch.device("cpu")):
    if device != torch.device("cpu") and not torch.cuda.is_available():
        device = torch.device("cpu")
    log_module.info(f"setup device")

    log_module.debug(f"setup dimension info")
    # find number of coils
    num_coils = data_mdbs[-1].block_len

    log_module.debug(f"allocate img arrays")
    etl = sampling_config.df_sampling_pattern["echo_number"].unique().max() + 1

    # build image array
    # x is rl, y is pa
    if pulseq_config.acq_phase_dir == "RL":
        transpose_xy = True
    else:
        # the pulseq conf class should make sure its only those two options possible
        transpose_xy = False

    # get dimensions
    n_read = pulseq_config.resolution_n_read
    n_phase = pulseq_config.resolution_n_phase
    n_slice = pulseq_config.resolution_slice_num
    os_factor = pulseq_config.oversampling

    # allocate space
    k_space = np.zeros(
        (n_read * os_factor, n_phase, n_slice, num_coils, etl),
        dtype=complex
    )
    k_sampling_mask = np.zeros(
        (n_read, n_phase, etl),
        dtype=bool
    )

    log_module.debug(f"Remove SYNCDATA acquisitions and get Arrays".rjust(20))
    # remove data not needed, i.e. SYNCDATA acquisitions
    data_mdbs = [d for d in data_mdbs if "SYNCDATA" not in d.get_active_flags()]

    # check number of scans
    if not len(sampling_config.samples) == len(data_mdbs):
        msg = (f"sampling pattern info assumes different number of scans "
               f"({len(sampling_config.samples)}) then input data ({len(data_mdbs)})!")
        log_module.warning(msg)

    log_module.debug(f"Start sorting".ljust(20))
    # save noise scans separately, used later
    noise_scans = None

    for acq in sampling_config.df_sampling_pattern["acquisition"].unique():
        log_module.info(f"Processing acquisition type: {acq}")
        # take all sampled lines belonging to respective acquisition
        sampled_lines = sampling_config.df_sampling_pattern.filter(pl.col("acquisition") == acq)
        # get scan numbers
        sampled_scan_numbers = sampled_lines["scan_number"].to_list()
        # get corresponding data
        data = np.array([data_mdbs[k].data for k in sampled_scan_numbers])
        if acq == "noise_scan":
            # save separately, no further sorting necessary
            noise_scans = data
            continue
        if "nav" in acq:
            log_module.info("not yet implemented any navigator scans".rjust(50))
            continue
        # sort data into k-space and sampling mask
        # get corresponding phase encodes
        sampled_phase_encodes = sampled_lines["phase_encode_number"].to_numpy()
        # get corresponding echo numbers
        sampled_echo_numbers = sampled_lines["echo_number"].to_numpy()
        # get corresponding slice numbers
        sampled_slice_numbers = sampled_lines["slice_number"].to_numpy()

        # if not noise scan we want to retrieve trajectory
        log_module.debug(f"Loading trajectory {acq}")
        k_traj = [t for t in sampling_config.trajectories if t.name == acq][0]

        # get indices on grid (no gridding algorithms or similar)
        k_pos, sampling_idxs = get_k_space_indices_from_trajectory_grid(
            trajectory=k_traj.trajectory, n_read=n_read, os_factor=os_factor
        )

        # sort k_space data
        k_space[
        k_pos[None, :],
        sampled_phase_encodes[:, None],
        sampled_slice_numbers[:, None],
        :,
        sampled_echo_numbers[:, None]
        ] = np.moveaxis(data, -2, -1)

        # sampling equal for all slice numbers, we can reduce to one of the slices
        sampled_mask_vals = sampled_lines.filter(pl.col("slice_number") == 0)
        sampled_mask_pe = sampled_mask_vals["phase_encode_number"].to_numpy()
        sampled_mask_echoes = sampled_mask_vals["echo_number"].to_numpy()
        k_sampling_mask[sampling_idxs[:, None], sampled_mask_pe[None, :], sampled_mask_echoes[None, :]] = True

    log_module.info(f"Finished extracting all acquisitions!")
    #
    # # remove oversampling, use gpu if set
    # k_space = remove_oversampling(
    #     data=k_space, data_input_sampled_in_time=True, read_dir=0, os_factor=os_factor
    # )
    #
    # # fft bandpass filter for oversampling removal not consistent
    # # with undersampled in the 0 filled regions data, remove artifacts
    # k_space *= k_sampling_mask[:, :, None, None, :]

    # decorrelate channels
    if noise_scans is not None:
        psi_l_inv = get_whitening_matrix(noise_data_n_samples_channel=np.swapaxes(noise_scans, -2, -1))
        k_space = np.einsum("ijkmn, lm -> ijkln", k_space, psi_l_inv, optimize=True)
        noise_scans = np.einsum("imn, lm -> iln", noise_scans, psi_l_inv, optimize=True)

    # remove oversampling, use gpu if set
    k_space = remove_oversampling(
        data=k_space, data_input_sampled_in_time=True, read_dir=0, os_factor=os_factor
    )

    # fft bandpass filter for oversampling removal not consistent
    # with undersampled in the 0 filled regions data, remove artifacts
    # extend mask to full dims
    k_space *= k_sampling_mask[:, :, None, None, :]
    ext_k_sampling_mask = np.tile(k_sampling_mask[:, :, None, None, :], (1, 1, *k_space.shape[2:4], 1))

    filter_input = np.reshape(
        k_space[ext_k_sampling_mask],
        (k_space.shape[0], -1, *k_space.shape[2:])
    )
    k_space_filt = np.zeros_like(k_space)
    filtered_input = matched_filter_noise_removal(noise_data=noise_scans, k_space_lines=filter_input)
    k_space_filt[ext_k_sampling_mask] = filtered_input.flatten()

    # correct gradient directions - at the moment we have reversed z dir
    k_space = np.flip(k_space, axis=2)
    k_space_filt = np.flip(k_space_filt, axis=2)

    # # scale values
    # ks_max = np.max(np.abs(k_space))
    # k_space *= scale_to_max_val / ks_max

    log_module.info(f"Extract geometry & affine information")
    # this is very dependent on the geom object from pulseq, can change with different pulseg.dll on scanner,
    # which defines resolution and matrix sizes
    # gap = (geometry.voxelsize[-1] - n_slice * resolution_slice) / (n_slice - 1)
    gap = pulseq_config.resolution_slice_gap / 100 * pulseq_config.resolution_slice_thickness
    # # for affine
    voxel_dims = np.array([
        pulseq_config.resolution_voxel_size_read,
        pulseq_config.resolution_voxel_size_phase,
        pulseq_config.resolution_slice_thickness
    ])
    fov = np.array([*(k_space.shape[:2] * voxel_dims[:2]), geometry.voxelsize[-1]])

    # swap dims if phase dir RL
    if transpose_xy:
        k_space = np.swapaxes(k_space, 0, 1)
        k_space_filt = np.swapaxes(k_space_filt, 0, 1)
        k_sampling_mask = np.swapaxes(k_sampling_mask, 0, 1)
        voxel_dims = voxel_dims[[1, 0, 2]]
        fov = fov[[1, 0, 2]]

    # get affine
    aff = get_affine(
        geometry,
        voxel_sizes_mm=voxel_dims,
        fov_mm=fov,
        slice_gap_mm=gap
    )

    return k_space, k_sampling_mask, aff, k_space_filt


def load_siemens_rd():
    log_module.error(f"Siemens raw data extraction not yet implemented!")
    return None, None, None