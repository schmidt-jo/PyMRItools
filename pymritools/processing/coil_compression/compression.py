"""
using
Huang F, Vijayakumar S, Li Y, Hertel S, Duensing GR.
A software channel compression technique for faster reconstruction with many channels.
Magn Reson Imaging 2008; 26: 133–141

and gcc -> idea that coil sensitivities are varying dependent on spatial position, but are assumed to be unknown.
We benefit here from using a coil compression in each slice.

Tao Zhang, John M. Pauly, Shreyas S. Vasanawala, and Michael Lustig
Coil  Compression for Accelerated Imaging with Cartesian Sampling
Magn Reson Med. 2013 69

We do not need to compute inverse fourier transform along slice dimension as we have a slice wise acquisition.
Do PCA based channel compression for faster computation of LORAKS
"""

import torch
import logging
import tqdm

log_module = logging.getLogger(__name__)


def compress_channels(input_k_space: torch.tensor, sampling_pattern: torch.tensor,
                      num_compressed_channels: int, use_ac_data: bool = True, use_gcc_along_read: bool = False
                      ) -> torch.tensor:
    """ k-space is assumed to be provided in dims [x, y, z, ch, t (optional)]"""
    # set path for plotting
    # out_path = plib.Path(opts.output_path).absolute()
    # fig_path = out_path.joinpath("plots/")
    # check input
    while input_k_space.shape.__len__() < 5:
        input_k_space = torch.unsqueeze(input_k_space, -1)
    while sampling_pattern.shape.__len__() < 5:
        sampling_pattern = torch.unsqueeze(sampling_pattern, -2)
    nx, ny, nz, nch, nt = input_k_space.shape
    # check if we are actually provided fewer channels
    if nch <= num_compressed_channels:
        msg = (f"input data has fewer channels ({nch}) than set for compression ({num_compressed_channels})! "
               f"No compression done.")
        log_module.info(msg)
        return input_k_space
    # slice wise processing
    log_module.info(f"extract ac region from sampling mask")
    # find ac data - we look for fully contained neighborhoods starting from middle position of the sampling mask
    # assuming ACs data is equal for all echoes (potentially except first)
    # Here we have a specific implementation for the sampling patterns used in jstmc mese and megesse sequences.
    # In the mese case, the second echo usually has higher SNR. In the megesse case,
    # the first echo is sampled with partial fourier in read direction.
    # Hence we skip the first echo.
    mid_x = int(nx / 2)
    mid_y = int(ny / 2)
    # make sure sampling pattern is bool
    sampling_pattern = sampling_pattern.to(torch.bool)
    if use_ac_data:
        # move from middle out
        lrbt = [mid_x, mid_x, mid_y, mid_y]
        boundaries = [nx, nx, ny, ny]
        dir = [-1, 1, -1, 1]
        cont_lrbt = [True, True, True, True]
        for _ in torch.arange(1, max(mid_x, mid_y) + 1):
            for idx_dir in range(len(lrbt)):
                pos = lrbt[idx_dir] + dir[idx_dir]
                # for each direction, check within range
                if 0 <= pos < boundaries[idx_dir]:
                    if idx_dir < 2:
                        pos_x = pos
                        pos_y = mid_y
                    else:
                        pos_x = mid_x
                        pos_y = pos
                    if sampling_pattern[pos_x, pos_y, 0, 0, 1] and cont_lrbt[idx_dir]:
                        # sampling pattern true and still looking
                        lrbt[idx_dir] = pos
                    elif not sampling_pattern[pos_x, pos_y, 0, 0, 1] and cont_lrbt[idx_dir]:
                        # sampling pattern false, toggle looking
                        cont_lrbt[idx_dir] = False
        # extract ac region
        ac_mask = torch.zeros_like(sampling_pattern[:, :, 0, 0, 0])
        ac_mask[lrbt[0]:lrbt[1], lrbt[2]:lrbt[3]] = True
        # check detected region
        ac_mask = ac_mask[:, :, None, None, None].expand(-1, -1, nz, nch, nt)
    else:
        # use all available sampled data
        ac_mask = sampling_pattern.expand(-1, -1, nz, nch, -1)
    if torch.nonzero(ac_mask).shape[0] < 100:
        err = f"Number of available sampled data / AC region detected too small (<100 voxel). exiting"
        log_module.error(err)
        raise AttributeError(err)
    ac_data = input_k_space.clone()
    ac_data[~ac_mask] = 0
    s_y = ac_mask[mid_x, :, 0, 0, 1].to(torch.int).sum()
    s_x = ac_mask[:, mid_y, 0, 0, 1].to(torch.int).sum()

    # find readout direction, assume fs (edges can be unsampled):
    if s_y < ny - 5 and s_x < nx - 5:
        err = f"neither read nor phase ACs region is fully sampled."
        log_module.error(err)
        raise AttributeError(err)
    elif s_y < ny - 5:
        read_dim = 0
        n_read = nx
        n_phase = ny
        mid = mid_x
    else:
        read_dim = 1
        n_read = ny
        n_phase = nx
        mid = mid_y

    # move fs dim first
    ac_data = torch.movedim(ac_data, read_dim, 0)
    ac_mask = torch.movedim(ac_mask, read_dim, 0)
    in_comp_data = torch.movedim(input_k_space, read_dim, 0)

    # reduce non fs dimension
    ac_data = ac_data[:, ac_mask[mid].expand(-1, -1, nch, -1)]
    ac_data = torch.reshape(ac_data, (n_read, -1, nz, nch, nt))
    # we do the compression slice wise and additionally try to deduce the compression matrix from highest snr data,
    # chose first third of echoes here but skip first one.
    dim_t = max(int(input_k_space.shape[-1] / 3), 2)
    ac_data = ac_data[:, :, :, :, 1:dim_t]
    if use_gcc_along_read:
        # we want to compute coil compression along slice and fs read dim, create hybrid data with slice
        # and read in img domain
        in_comp_data = torch.fft.fftshift(
            torch.fft.ifft(
                torch.fft.ifftshift(
                    in_comp_data,
                    dim=0
                ),
                dim=0
            ),
            dim=0
        )
        ac_data = torch.fft.fftshift(
            torch.fft.ifft(
                torch.fft.ifftshift(
                    ac_data,
                    dim=0
                ),
                dim=0
            ),
            dim=0
        )
    # allocate output_data
    compressed_data = torch.zeros(
        (n_read, n_phase, nz, num_compressed_channels, nt),
        dtype=input_k_space.dtype
    )

    log_module.info(f"start pca -> building compression matrix from calibration data -> use gcc")
    for idx_slice in tqdm.trange(nz):
        # chose slice data
        sli_data = ac_data[:, :, idx_slice]
        # we do virtual coil alignment after computation of the compression matrix with 0 transform p_x = I
        # then we compress data
        a_l_matrix_last = 1
        if use_gcc_along_read:
            for idx_read in range(n_read):
                gcc_data = sli_data[idx_read]
                # set channel dimension first and rearrange rest
                gcc_data = torch.moveaxis(gcc_data, -2, 0)
                gcc_data = torch.reshape(gcc_data, (nch, -1))
                # substract mean from each channel vector
                # gcc_data = gcc_data - torch.mean(gcc_data, dim=1, keepdim=True)
                # calculate covariance matrix
                # cov = torch.cov(gcc_data)
                # cov_eig_val, cov_eig_vec = torch.linalg.eig(cov)
                # # get coil compression matrix
                # a_l_matrix = cov_eig_vec[:num_compressed_channels]
                u_x_0, _, _ = torch.linalg.svd(gcc_data, full_matrices=False)
                a_l_matrix = torch.conj(u_x_0).T[:num_compressed_channels]

                # virtual coil alignment
                if idx_read > 0:
                    # we transform with identy for 0 element,
                    # i.e nothing to do. we keep the initial calculated compression matrix
                    # after that
                    # transform computed from previous read compression matrix
                    # define cx
                    c_x = torch.matmul(a_l_matrix, torch.conj(a_l_matrix_last).T)
                    # compute svd
                    ux, _, vxh = torch.linalg.svd(c_x, full_matrices=False)
                    # calc transform
                    p_x = torch.linalg.matmul(torch.conj(vxh).T, torch.conj(ux).T)
                else:
                    p_x = torch.eye(num_compressed_channels, dtype=torch.complex128)
                # align compression matrix
                a_l_matrix = torch.matmul(p_x, a_l_matrix)

                # compress data -> coil dimension over a_l
                compressed_data[idx_read, :, idx_slice] = torch.einsum(
                    "imn, om -> ion",
                    in_comp_data[idx_read, :, idx_slice],
                    a_l_matrix
                )
                # keep last matrix
                a_l_matrix_last = a_l_matrix.clone()
        else:
            # use gcc only along slice dim
            # set channel dimension first and rearrange rest
            gcc_data = torch.moveaxis(sli_data, -2, 0)
            gcc_data = torch.reshape(gcc_data, (nch, -1))
            # substract mean from each channel vector
            # gcc_data = gcc_data - torch.mean(gcc_data, dim=1, keepdim=True)
            # calculate covariance matrix
            # cov = torch.cov(gcc_data)
            # cov_eig_val, cov_eig_vec = torch.linalg.eig(cov)
            # # get coil compression matrix
            # a_l_matrix = cov_eig_vec[:num_compressed_channels]
            u_x_0, _, _ = torch.linalg.svd(gcc_data, full_matrices=False)
            a_l_matrix = u_x_0[:num_compressed_channels]
            # compress data -> coil dimension over a_l
            compressed_data[:, :, idx_slice] = torch.einsum(
                "ikmn, om -> ikon",
                in_comp_data[:, :, idx_slice],
                a_l_matrix
            )

    if use_gcc_along_read:
        # transform back
        compressed_data = torch.fft.fftshift(
            torch.fft.fft(
                torch.fft.ifftshift(
                    compressed_data,
                    dim=0
                ),
                dim=0
            ),
            dim=0
        )
    # move back
    compressed_data = torch.movedim(compressed_data, 0, read_dim)

    # remove fft "bleed"
    compressed_data[~sampling_pattern.expand(-1, -1, nz, num_compressed_channels, -1)] = 0

    # if opts.visualize and opts.debug:
    #     sli, ch, t = (torch.tensor([*compressed_data.shape[2:]]) / 2).to(torch.int)
    #     plot_k = compressed_data[:, :, sli, ch, 0]
    #     plotting.plot_img(img_tensor=plot_k.clone().detach().cpu(), log_mag=True,
    #                       out_path=fig_path, name=f"fs_k_space_compressed")
    #     fs_img_recon = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(plot_k)))
    #     plotting.plot_img(img_tensor=fs_img_recon.clone().detach().cpu(),
    #                       out_path=fig_path, name="fs_recon_compressed")
    return compressed_data
