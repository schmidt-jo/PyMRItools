import logging
from twixtools.geometry import Geometry
from twixtools.mdb import Mdb
from pymritools.config.seqprog import Sampling, PulseqParameters2D
import numpy as np
import tqdm
import polars as pl

log_module = logging.getLogger(__name__)


def load_pulseq_rd(
        pulseq_config: PulseqParameters2D, sampling_config: Sampling,
        data_mdbs: list[Mdb], geometry: Geometry, hdr: dict):
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

    log_module.info(f"Remove SYNCDATA acquisitions and get Arrays".rjust(20))
    # remove data not needed, i.e. SYNCDATA acquisitions
    data_mdbs = [d for d in data_mdbs if "SYNCDATA" not in d.get_active_flags()]

    # check number of scans
    if not len(sampling_config.samples) == len(data_mdbs):
        msg = (f"sampling pattern info assumes different number of scans "
               f"({len(sampling_config.samples)}) then input data ({len(data_mdbs)})!")
        log_module.warning(msg)

    log_module.info(f"Start sorting".ljust(20))
    # save noise scans separately, used later
    noise_scans = None

    for a in sampling_config.df_sampling_pattern["acquisition"].unique():
        log_module.info(f"Processing acquisition type: {a}")
        # take all sampled lines belonging to respective acquisition
        sampled_lines = sampling_config.df_sampling_pattern.filter(pl.col("acquisition") == a)
        # get scan numbers
        sampled_scan_numbers = sampled_lines["scan_number"].to_list()
        # get corresponding data
        data = np.array([data_mdbs[k].data for k in sampled_scan_numbers])
        if a == "noise_scan":
            # save separately, no further sorting necessary
            noise_scans = data
            continue
        # sort data into k-space and sampling mask

        # # get the corresponding line sample from the config object
        # sample_line = sampling_config.samples[idx_s]
        # # get acquisition type
        # acq = sample_line.acquisition_type
        # # retrieve data as numpy array, dim [num_channels, adc samples]
        # data = np.array(d.data)
        # # raise sample line counter
        # idx_s += 1
        # # check for noise scan
        # if acq
        #     # we want to save noise scans for later use
        #     noise_scans.append(data)
        #     # then exit
        #     continue
        # # otherwise we want to retrieve trajectory, in case acquisition type changed
        # if acq != acquisition:
        #     # new acquisition, load acquisition trajectory
        #     log_module.info(f"Loading trajectory {acq}")
        #     k_traj = [t for t in sampling_config.trajectories if t.name == acq][0]
        #     acquisition = acq
        #     # calculate the closest points on grid from trajectory. get positions
        #     k_traj_pos = (0.5 + k_traj.trajectory) * n_read * os_factor
        #     # just round to the nearest point in 1d
        #     k_traj_pos = np.round(k_traj_pos).astype(int)
        #     # find indices of points that are actually within our k-space in case the trajectory maps too far out
        #     k_traj_valid_index = np.squeeze(
        #         np.nonzero((k_traj_pos >= 0) & (k_traj_pos < os_factor * n_read)))
        #     # find positions of those indices
        #     k_traj_valid_pos = k_traj_pos[k_traj_valid_index].astype(int)
        #     indices_in_k_sampling_img = (k_traj_valid_pos[::os_factor] / os_factor).astype(int)
        #
        # # get all sampled lines
        # # get indices needed from sampling conf of the sampled line
        # n_echo = sample_line.echo_number
        # n_slice = sample_line.slice_number
        # n_phase = sample_line.phase_encode_number
        #
        # # use trajectory to populate k_space data
        # k_space[k_traj_valid_pos, n_phase, n_slice, :, n_echo] = np.moveaxis(data, 0, 1)
        # k_sampling_mask[indices_in_k_sampling_img, n_phase, n_echo] = True

    log_module.info("Finished sorting".ljust(20))






    # log_module.debug(f"allocate navigator arrays")
    # nav = nav_params.n_read > 0
    # if nav:
    #     nav_shape = (nav_params.n_read * nav_params.os_factor, nav_params.n_phase, nav_params.n_slice, num_coils)
    #     k_nav = np.zeros(nav_shape, dtype=complex)
    #     k_nav_mask = np.zeros(nav_shape[:2], dtype=bool)
    # else:
    #     k_nav = None
    #     k_nav_mask = None

    # setup tkb
    # # use tkbnufft interpolation to get k-space data from trajectory
    # device = torch.device("cpu")
    # # we want to regrid the data to the regular grid but have oversampling in read direction
    # img_size = (img_params.os_factor * img_params.n_read,)
    # grid_size = (img_params.os_factor * img_params.n_read,)
    # tkbn_interp = tkbn.KbInterpAdjoint(
    #     im_size=img_size,
    #     grid_size=grid_size,
    #     device=device
    # )

    log_module.debug("loop through acquisition types")
    for acq_type in sampling_config.acquisitions:
        pyp_processing_acq_data(
            acq_type=acq_type, interface=interface, nav=nav, mdb_list=mdb_list,
            k_space=k_space, k_sampling_mask=k_sampling_mask, img_params=img_params,
            k_nav=k_nav, k_nav_mask=k_nav_mask, nav_params=nav_params
        )
    log_module.info(f"\t -- done!")
    #
    # # remove oversampling
    # k_space = helper_fns.remove_os(
    #     data=k_space, data_input_sampled_in_time=True, read_dir=0, os_factor=img_params.os_factor
    # )
    # # fft bandpass filter not consistent with undersampled in the 0 filled regions data, remove artifacts
    # k_space *= k_sampling_mask[:, :, None, None, :]
    #
    # # whiten data, psi_l_inv dim [nch, nch], k-space dims [nfe, npe, nsli, nch, nechos]
    # log_module.info("noise decorrelation")
    # acq_type = "noise_scan"
    # # get all sampling pattern entries matching acq. type and no navigator
    # sp_sub = interface.sampling_k_traj.sampling_pattern[
    #     (interface.sampling_k_traj.sampling_pattern["acq_type"] == acq_type)
    # ]
    # # get line numbers
    # line_nums = sp_sub.index.to_list()
    # # get cov
    # noise_lines = np.array([mdb_list[i].data.T for i in line_nums])
    # while noise_lines.shape.__len__() < 3:
    #     noise_lines = noise_lines[None]
    # psi_l_inv = 0
    # noise_cov = 0
    # for k in range(noise_lines.shape[0]):
    #     psi, cov = get_noise_cov_matrix(noise_data_line=noise_lines[k])
    #     psi_l_inv += psi
    #     noise_cov += cov
    # psi_l_inv /= noise_lines.shape[0]
    # noise_cov /= noise_lines.shape[0]
    #
    # k_space = np.einsum("ijkmn, lm -> ijkln", k_space, psi_l_inv, optimize="optimal")
    # if nav:
    #     k_nav = np.einsum("ijkm, lm -> ijkl", k_nav, psi_l_inv, optimize="optimal")
    #
    # # correct gradient directions - at the moment we have reversed z dir
    # k_space = np.flip(k_space, axis=2)
    # # k_sampling_mask = np.flip(k_sampling_mask, axis=(0, 1))
    #
    # # scale values
    # ks_max = np.max(np.abs(k_space))
    # k_space *= scale_to_max_val / ks_max
    #
    # # this is very dependent on the geom object from pulseq, can change with different pulseg.dll on scanner,
    # # which defines resolution and matrix sizes
    # gap = (geom.voxelsize[-1] - img_params.n_slice * img_params.resolution_slice) / (img_params.n_slice - 1)
    # # for affine
    # voxel_dims = np.array([
    #     interface.recon.multi_echo_img.resolution_read,
    #     interface.recon.multi_echo_img.resolution_phase,
    #     interface.recon.multi_echo_img.resolution_slice
    # ])
    # fov = np.array([*(k_space.shape[:2] * voxel_dims[:2]), geom.voxelsize[-1]])
    #
    # if nav:
    #     # remove oversampling
    #     k_nav = helper_fns.remove_os(
    #         data=k_nav, data_input_sampled_in_time=True, read_dir=0, os_factor=nav_params.os_factor
    #     )
    #     # correct gradient directions
    #     # k_nav = np.flip(k_nav, axis=(0, 1, 2))
    #     # k_nav_mask = np.flip(k_nav_mask, axis=(0, 1))
    #
    #     # scale values
    #     kn_max = np.max(np.abs(k_nav))
    #     k_nav *= scale_to_max_val / kn_max
    #
    # # swap dims if phase dir RL
    # if transpose_xy:
    #     k_space = np.swapaxes(k_space, 0, 1)
    #     k_sampling_mask = np.swapaxes(k_sampling_mask, 0, 1)
    #     voxel_dims = voxel_dims[[1, 0, 2]]
    #     fov = fov[[1, 0, 2]]
    #     if nav:
    #         k_nav = np.swapaxes(k_nav, 0, 1)
    #         k_nav_mask = np.swapaxes(k_nav_mask, 0, 1)
    #
    # aff = get_affine(
    #     geom,
    #     voxel_sizes_mm=voxel_dims,
    #     fov_mm=fov,
    #     slice_gap_mm=gap
    # )
    #
    # if nav:
    #     # navigators scaled (lower resolution)
    #     nav_scale_x = k_space.shape[0] / k_nav.shape[0]
    #     nav_scale_y = k_space.shape[1] / k_nav.shape[1]
    #     aff_nav = np.matmul(
    #         np.diag([nav_scale_x, nav_scale_y, 1, 1]),
    #         aff
    #     )
    # else:
    #     aff_nav = None
    #
    #
    # pass

def load_siemens_rd():
    pass