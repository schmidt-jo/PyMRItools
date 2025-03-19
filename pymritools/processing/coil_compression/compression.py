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
import numpy as np
import logging
import tqdm

from pymritools.utils import fft

log_module = logging.getLogger(__name__)


def shape_input(input_k_space: torch.Tensor, sampling_pattern: torch.Tensor):
    # check input
    while input_k_space.shape.__len__() < 5:
        input_k_space = torch.unsqueeze(input_k_space, -1)
    while sampling_pattern.shape.__len__() < 5:
        sampling_pattern = torch.unsqueeze(sampling_pattern, -2)
    read_dir = -1
    for i in range(2):
        sp = torch.movedim(sampling_pattern, i, 0)
        num_samples = torch.sum(sp[:, sp.shape[1] // 2, sp.shape[2] // 2], dim=0)
        num_samples = torch.unique(num_samples)[0].item()
        if num_samples == sp.shape[0] and read_dir < 0:
            read_dir = i
    if read_dir < 0:
        err = (f"Could not deduce read dimension from input "
               f"(no middle sampled line in x-y-z corresponds to the shape dim).")
        log_module.error(err)
        raise AttributeError(err)
    log_module.info(f"found read dir for channel compression: {['x', 'y'][i]}")

    input_k_space = torch.movedim(input_k_space, read_dir, 0)
    sampling_pattern = torch.movedim(sampling_pattern, read_dir, 0)
    return input_k_space, sampling_pattern, read_dir


def compress_channels_2d(
        input_k_space: torch.tensor, sampling_pattern: torch.tensor,
        num_compressed_channels: int, use_ac_data: bool = True, visualize: bool = False,
        device: torch.device = torch.get_default_device()
    ) -> torch.tensor:
    log_module.info(f"GCC channel compression")
    nx, ny, nz, nc, ne = input_k_space.shape
    # check if we are actually provided fewer channels
    if nc <= num_compressed_channels:
        msg = (f"\t\tinput data has fewer channels ({nc}) than set for compression ({num_compressed_channels})! "
               f"No compression done.")
        log_module.info(msg)
        return input_k_space.cpu()
    """ k-space is assumed to be provided in dims [x, y, z, ch, t (optional)]"""
    input_k_space, sampling_pattern, read_dir = shape_input(
        input_k_space=input_k_space, sampling_pattern=sampling_pattern
    )
    nx, ny, nz, nc, ne = input_k_space.shape

    # get sampled data
    sampled_data = get_sampled_data(
        input_k_space=input_k_space, sampling_pattern=sampling_pattern, use_ac_data=use_ac_data
    )
    # check reduction - our use case always will have reduced y dims, x dim should be fully sampled but in k-space
    # z dim should be fully sampled but in img space (2d sequence and sorting)

    # deduce here the sampled dims. we want to do gcc in all fs dimensions, for know we hardcode the use case
    # need to revisit this if we want to automate this more
    # get all fs dimensions into img space
    in_comp_data = fft(input_data=sampled_data, img_to_k=False, axes=(0,))

    # we do slice wise processing (2d function), hence we do not account for geometric offsets in slice direction
    # get slice dim as batch dim up front
    # in_comp_data = torch.movedim(in_comp_data, 2, 0)
    # we suppose the channel sensitivities are geometrically similar across echoes.
    # We decide here to combine phase and echo data as to use all available data per slice
    in_comp_data = torch.movedim(in_comp_data, -1, 2)
    in_comp_data = torch.reshape(in_comp_data, (nx, -1, nz, nc))

    # could also try to use the echo mean:
    # in_comp_data = torch.mean(in_comp_data, dim=-1)
    # we should be left with [nx, ny, nz, nc]

    # check dims, we are using gcc along read direction across channels and phase encodes,
    # hence our maximum compression level is the min out of both quantities
    if num_compressed_channels > in_comp_data.shape[1]:
        msg = (f"Using GCC on AC data along read dimension. "
               f"Hence, compression matrix is calculated from matrices spanned by dims {in_comp_data.shape[-2:]}.\n"
               f"Chosen compression level ({num_compressed_channels}) is above maximum achievable by this method. "
               f"(Usually the number of AC lines).")
        log_module.info(msg)
        num_compressed_channels = in_comp_data.shape[1]
        log_module.info(f"Changing compression level to: {num_compressed_channels}")

    # allocate
    out_comp_data = torch.zeros((nx, ny, nz, num_compressed_channels, ne), dtype=input_k_space.dtype, device="cpu")

    # do slice wise computation
    iter_bar = tqdm.trange(nz, desc="slice - wise processing") if visualize else range(nz)
    for idx_z in iter_bar:
        log_module.debug(f"Processing slice: {idx_z+1} / {nz}")
        # perform svd
        batch = in_comp_data[:, :, idx_z].to(device)
        u, s, vh = torch.linalg.svd(torch.movedim(batch, -1, -2), full_matrices=False)

        # take first num_comp_c rows of uH as initial compression matrices
        a_gcc_0 = u.mH[..., :num_compressed_channels, :]

        # align a matrices along read direction
        a_x = a_gcc_0.clone()
        for i in range(a_x.shape[0] - 1):
            c = torch.matmul(a_x[i+1], a_x[i].mH)
            uc, sc, vhc = torch.linalg.svd(c, full_matrices=False)
            p = torch.matmul(vhc.mH, uc.mH)
            a_x[i+1] = torch.matmul(p, a_x[i+1])

        # prep input k-space data - take slice
        uncomp_data = fft(input_k_space[:, :, idx_z].to(device), img_to_k=False, axes=(0,))
        # do the coil compression at each location of all slice data!
        out_comp_data[:, :, idx_z] = torch.einsum("xdc, xyce -> xyde", a_x, uncomp_data).cpu()

    # reshape data, have combined
    # out_comp_data = torch.reshape(out_comp_data, (ne, nz, nx, ny, num_compressed_channels))
    # out_comp_data = torch.movedim(out_comp_data, 0, 2)
    # out_comp_data = torch.movedim(out_comp_data, -2, -1)

    # reverse fft in x dim
    out_comp_data = fft(out_comp_data, img_to_k=True, axes=(0,))
    out_comp_data = torch.movedim(out_comp_data, 0, read_dir)
    torch.cuda.empty_cache()
    return out_comp_data.cpu()


def get_sampled_data(input_k_space: torch.Tensor, sampling_pattern: torch.Tensor, use_ac_data: bool = True):
    nx, ny, nz, nc, nt = input_k_space.shape
    log_module.info(f"extract sampled data location from sampling mask")
    if use_ac_data:
        log_module.info(f"\t\tUsing only AC region")
    # find ac data - we look for fully contained neighborhoods starting from middle position of the sampling mask
    # assuming ACs data is equal for all echoes (potentially except first)
    # Here we have a specific implementation for the sampling patterns used in jstmc mese and megesse sequences.
    # In the mese case, the second echo usually has higher SNR. In the megesse case,
    # the first echo is sampled with partial fourier in read direction.
    # Hence we skip the first echo.
    mid_x = int(nx / 2)
    mid_y = int(ny / 2)
    # make sure sampling pattern is bool
    sampling_mask = sampling_pattern.clone().to(torch.bool)
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
                    if sampling_mask[pos_x, pos_y, 0, 0, 0] and cont_lrbt[idx_dir]:
                        # sampling pattern true and still looking
                        lrbt[idx_dir] = pos
                    elif not sampling_mask[pos_x, pos_y, 0, 0, 0] and cont_lrbt[idx_dir]:
                        # sampling pattern false, toggle looking
                        cont_lrbt[idx_dir] = False
        # extract ac region
        data_mask = torch.zeros_like(sampling_mask[:, :, 0, 0, 0])
        data_mask[lrbt[0]:lrbt[1], lrbt[2]:lrbt[3]] = True
        # check detected region
        data_mask = data_mask[:, :, None, None, None].expand(-1, -1, nz, nc, nt)
    else:
        # use all available sampled data
        data_mask = sampling_mask.expand(-1, -1, nz, nc, -1)
    if torch.nonzero(data_mask).shape[0] < 100:
        err = f"\t\tNumber of available sampled data / AC region detected too small (<100 voxel). exiting"
        log_module.error(err)
        raise AttributeError(err)
    sampled_data = input_k_space.clone()
    sampled_data[~data_mask] = 0
    s_y = data_mask[mid_x, :, 0, 0, 0].to(torch.int).sum()
    s_x = data_mask[:, mid_y, 0, 0, 0].to(torch.int).sum()

    # find readout direction, assume fs (edges can be unsampled):
    if s_y < ny - 5 and s_x < nx - 5:
        err = f"\t\tneither read nor phase ACs region is fully sampled."
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
    sampled_data = torch.movedim(sampled_data, read_dim, 0)
    data_mask = torch.movedim(data_mask, read_dim, 0)

    # reduce non fs dimension
    sampled_data = sampled_data[:, data_mask[mid].expand(-1, -1, nc, -1)]
    sampled_data = torch.reshape(sampled_data, (n_read, -1, nz, nc, nt))
    return sampled_data


def compress_channels_arxv(input_k_space: torch.tensor, sampling_pattern: torch.tensor,
                      num_compressed_channels: int, use_ac_data: bool = True, use_gcc_along_read: bool = True
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
    sampling_mask = sampling_pattern.clone().to(torch.bool)
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
                    if sampling_mask[pos_x, pos_y, 0, 0, 1] and cont_lrbt[idx_dir]:
                        # sampling pattern true and still looking
                        lrbt[idx_dir] = pos
                    elif not sampling_mask[pos_x, pos_y, 0, 0, 1] and cont_lrbt[idx_dir]:
                        # sampling pattern false, toggle looking
                        cont_lrbt[idx_dir] = False
        # extract ac region
        ac_mask = torch.zeros_like(sampling_mask[:, :, 0, 0, 0])
        ac_mask[lrbt[0]:lrbt[1], lrbt[2]:lrbt[3]] = True
        # check detected region
        ac_mask = ac_mask[:, :, None, None, None].expand(-1, -1, nz, nch, nt)
    else:
        # use all available sampled data
        ac_mask = sampling_mask.expand(-1, -1, nz, nch, -1)
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
    # skip last third of echoes.
    dim_t = int(input_k_space.shape[-1] / 3)
    if dim_t > 0:
        ac_data = ac_data[:, :, :, :, :-dim_t]
    if use_gcc_along_read:
        # we want to compute coil compression along fs read dim, create hybrid data with slice
        # and read in img domain
        in_comp_data = fft(in_comp_data, img_to_k=False, axes=(0,))
        ac_data = fft(ac_data, img_to_k=False, axes=(0,))
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
    compressed_data[~sampling_mask.expand(-1, -1, nz, num_compressed_channels, -1)] = 0

    # if opts.visualize and opts.debug:
    #     sli, ch, t = (torch.tensor([*compressed_data.shape[2:]]) / 2).to(torch.int)
    #     plot_k = compressed_data[:, :, sli, ch, 0]
    #     plotting.plot_img(img_tensor=plot_k.clone().detach().cpu(), log_mag=True,
    #                       out_path=fig_path, name=f"fs_k_space_compressed")
    #     fs_img_recon = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(plot_k)))
    #     plotting.plot_img(img_tensor=fs_img_recon.clone().detach().cpu(),
    #                       out_path=fig_path, name="fs_recon_compressed")
    return compressed_data
