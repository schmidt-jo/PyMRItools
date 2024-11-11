import logging

import torch
from scipy.spatial.transform import Rotation
import numpy as np
import polars as pl
import tqdm

from twixtools.geometry import Geometry
from twixtools.mdb import Mdb
from pymritools.config.seqprog import Sampling, PulseqParameters2D
from pymritools.seqprog.rawdata.utils import remove_oversampling

log_module = logging.getLogger(__name__)


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


def load_pulseq_rd(
        pulseq_config: PulseqParameters2D, sampling_config: Sampling,
        data_mdbs: list[Mdb], geometry: Geometry, hdr: dict, split_read_polarity: bool = True,
        use_gpu: bool = True, gpu_device: int = 0):
    if torch.cuda.is_available() and use_gpu:
        device = torch.device(f'cuda:{gpu_device}')
    else:
        device = torch.device('cpu')
    log_module.info(f"setup device")

    log_module.debug(f"Remove SYNCDATA acquisitions and get Arrays".rjust(20))
    # remove data not needed, i.e. SYNCDATA acquisitions
    data_mdbs = [d for d in data_mdbs if "SYNCDATA" not in d.get_active_flags()]

    log_module.debug(f"setup dimension info")
    # find number of coils
    num_coils = data_mdbs[-1].mdh.UsedChannels

    log_module.debug(f"allocate img arrays")
    etl = sampling_config.df_sampling_pattern.filter(
        pl.col("acquisition") != "noise_scan"
    )["echo_number"].unique().max() + 1

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

    # check number of scans
    if not len(sampling_config.samples) == len(data_mdbs):
        msg = (f"sampling pattern info assumes different number of scans "
               f"({len(sampling_config.samples)}) then input data ({len(data_mdbs)})!")
        log_module.warning(msg)

    log_module.debug(f"Start sorting".ljust(20))
    # save noise scans separately, used later
    noise_scans = None
    # save echo numbers in case we want to split the readout polarities
    echo_numbers_bu = None
    echo_numbers_bd = None

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
        if split_read_polarity:
            # we want to split read polarity, for this we want to safe the echo numbers belonging to each readout dir
            if "bu" in acq:
                echo_numbers_bu = torch.from_numpy(sampled_lines["echo_number"].unique().sort().to_numpy().copy())
            elif "bd" in acq:
                echo_numbers_bd = torch.from_numpy(sampled_lines["echo_number"].unique().sort().to_numpy().copy())
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
    if noise_scans is not None:
        psi_l_inv = torch.from_numpy(
            get_whitening_matrix(noise_data_n_samples_channel=np.swapaxes(noise_scans, -2, -1))
        ).to(device=device, dtype=torch.complex128)
        noise_scans = torch.einsum(
            "imn, lm -> iln",
            torch.from_numpy(noise_scans).to(device=device, dtype=torch.complex128),
            psi_l_inv
        )
    log_module.info(f"remove oversampling")

    k_space_rm_os = np.zeros((n_read, n_phase, n_slice, num_coils, etl), dtype=k_space.dtype)
    # do some batched processing slice wise use gpu if available
    for idx_slice in tqdm.trange(n_slice, desc="slice wise processing"):
        batch_k = torch.from_numpy(k_space[:, :, idx_slice]).to(device=device, dtype=torch.complex128)
        if noise_scans is not None:
            batch_k = torch.einsum("ijmn, lm -> ijln", batch_k, psi_l_inv)
        # remove oversampling, use gpu if set
        batch_rmos = remove_oversampling(
            data=batch_k, data_input_sampled_in_time=True, read_dir=0, os_factor=os_factor
        )
        k_space_rm_os[:, :, idx_slice] = batch_rmos.cpu().numpy()

    # fft bandpass filter for oversampling removal not consistent
    # with undersampled in the 0 filled regions data, remove artifacts
    # extend mask to full dims
    k_space = k_space_rm_os * k_sampling_mask[:, :, None, None, :]

    # correct gradient directions - at the moment we have reversed z dir
    k_space = np.flip(k_space, axis=2)

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

    return k_space, k_sampling_mask, aff, noise_scans, echo_numbers_bu, echo_numbers_bd


def load_siemens_rd(
        data_mdbs: list[Mdb], geometry: Geometry, hdr: dict, split_read_polarity: bool = True,
        device: torch.device = torch.device("cpu")):
    if device != torch.device("cpu") and not torch.cuda.is_available():
        device = torch.device("cpu")
    log_module.info(f"setup device")

    log_module.debug(f"Remove SYNCDATA acquisitions and get Arrays".rjust(20))
    # remove data not needed, i.e. SYNCDATA acquisitions
    data_mdbs = [d for d in data_mdbs if "SYNCDATA" not in d.get_active_flags()]

    log_module.debug(f"setup dimension info")

    # find number of coils
    num_coils = data_mdbs[-1].mdh.UsedChannels

    tes = np.array(hdr["MeasYaps"]["alTE"]) * 1e-6
    trs = np.array(hdr["MeasYaps"]["alTR"]) * 1e-6
    fa = np.array(hdr["MeasYaps"]["adFlipAngleDegree"])
    # get dimensions
    n_slice = max(int(hdr["Config"]["NParMeas"]), int(hdr["Config"]["NSlcMeas"]))
    n_phase = int(hdr["Config"]["NLinMeas"])
    n_read = int(data_mdbs[-1].mdh.SamplesInScan)
    n_echoes = int(hdr["Config"]["NEcoMeas"])
    slice_pos = []
    for mdb in tqdm.tqdm(data_mdbs):
        if mdb.cEco > n_echoes:
            msg = f"found echo counter extending set number of echoes"
            log_module.error(msg)
            raise ValueError(msg)
        if mdb.cPar > n_slice or mdb.cSlc > n_slice:
            msg = f"found slice counter extending set number of slices"
            log_module.error(msg)
            raise ValueError(msg)
        if mdb.cLin > n_phase:
            msg = f"found phase counter extending set number of phases"
            log_module.error(msg)
            raise ValueError(msg)
        if mdb.mdh.SliceData.SlicePos not in slice_pos:
            slice_pos.append(mdb.mdh.SliceData.SlicePos)
    slice_pos_array = np.array([[sp.Cor, sp.Sag, sp.Tra] for sp in slice_pos])
    slice_pos_sort = np.sort(np.sum(slice_pos_array, axis=-1))

    os_factor = 2

    log_module.debug(f"allocate img arrays")

    # build image array
    # read_dir = None
    # x is rl, y is pa
    # if read_dir == "AP":
    #     transpose_xy = True
    # else:
    #     # the pulseq conf class should make sure its only those two options possible
    #     transpose_xy = False

    # allocate space
    k_space = np.zeros(
        (n_read * os_factor, n_phase, n_slice, num_coils, n_echoes),
        dtype=complex
    )
    noise_scans = []

    for mdb in tqdm.tqdm(data_mdbs, desc="Sorting"):
        if "NOISEADJSCAN" in mdb.get_active_flags():
            noise_scans.append(mdb.data)
            continue
        else:
            phase = mdb.cLin
            slice_num = max(mdb.cPar, mdb.cSlc)
            mmss = mdb.mdh.SliceData.SlicePos
            sli_p = np.sum(np.array([mmss.Sag, mmss.Cor, mmss.Tra]), axis=-1)
            sli = np.where(slice_pos_sort == sli_p)[0]
            echo = mdb.cEco
            k_space[:, phase, sli, :, echo] = mdb.data.T
    # save noise scans separately, used later
    noise_scans = np.array(noise_scans)
    k_sampling_mask = (np.abs(k_space) > 1e-9).astype(int)
    log_module.info("done")

    # decorrelate channels
    if noise_scans is not None:
        psi_l_inv = torch.from_numpy(
            get_whitening_matrix(noise_data_n_samples_channel=np.swapaxes(noise_scans, -2, -1))
        ).to(device)
        noise_scans = torch.einsum("imn, lm -> iln", torch.from_numpy(noise_scans).to(device), psi_l_inv)

    k_space_rm_os = np.zeros((n_read, n_phase, n_slice, num_coils, n_echoes), dtype=k_space.dtype)
    # do some batched processing slice wise use gpu if available
    for idx_slice in tqdm.trange(n_slice, desc="slice wise processing"):
        batch_k = torch.from_numpy(k_space[:, :, idx_slice]).to(device)
        if noise_scans is not None:
            batch_k = torch.einsum("ijmn, lm -> ijln", batch_k, psi_l_inv)
        # remove oversampling, use gpu if set
        batch_rmos = remove_oversampling(
            data=batch_k, data_input_sampled_in_time=True, read_dir=0, os_factor=os_factor
        )
        k_space_rm_os[:, :, idx_slice] = batch_rmos.cpu().numpy()

    k_sampling_mask = k_sampling_mask[::os_factor]

    # fft bandpass filter for oversampling removal not consistent
    # with undersampled in the 0 filled regions data, remove artifacts
    # extend mask to full dims
    k_space *= k_sampling_mask

    # correct gradient directions - at the moment we have reversed z dir
    k_space = np.flip(k_space, axis=(0, 1))
    k_sampling_mask = np.flip(k_sampling_mask, axis=(0, 1))

    log_module.info(f"Extract geometry & affine information")
    # this is very dependent on the geom object from pulseq, can change with different pulseg.dll on scanner,
    # which defines resolution and matrix sizes
    # # for affine
    voxel_dims = np.array(geometry.voxelsize)
    fov = np.array(geometry.fov)

    # k_space = np.swapaxes(k_space, 0, 1)
    # fov = fov[[1, 0, 2]]
    # swap dims if phase dir RL
    # if transpose_xy:
    #     k_space = np.swapaxes(k_space, 0, 1)
    #     k_sampling_mask = np.swapaxes(k_sampling_mask, 0, 1)
    #     voxel_dims = voxel_dims[[1, 0, 2]]
    #     fov = fov[[1, 0, 2]]

    # get affine
    aff = get_affine(
        geometry,
        voxel_sizes_mm=voxel_dims,
        fov_mm=fov,
        slice_gap_mm=0.0
    )

    return k_space, k_sampling_mask, aff, noise_scans

