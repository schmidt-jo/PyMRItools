import logging
import pathlib as plib

import torch
import numpy as np
import tqdm
import plotly.graph_objects as go

from pymritools.config import setup_program_logging, setup_parser
from pymritools.config.processing import DenoiseSettingsKLC

from pymritools.utils import torch_load, torch_save, nifti_save, fft, root_sum_of_squares
from pymritools.utils.matrix_indexing import get_linear_indices

log_module = logging.getLogger(__name__)


def weighting_fn(input_tensor: torch.Tensor, cutoff: float, percent_range: float = 5.0):
    """
    Computes the weighting function value for a PyTorch tensor based on the cutoff and the percent range.

    Parameters:
        input_tensor (torch.Tensor): The tensor of input values to evaluate.
        cutoff (float): The cutoff value.
        percent_range (float): The percentage range around the cutoff for smoothing (default is 5%).

    Returns:
        torch.Tensor: A tensor of the computed weighting values. Values will be 0 below the lower bound,
                      1 above the upper bound, and vary smoothly between them.
    """
    # Calculate the lower and upper bounds for the transition region
    lower_bound = cutoff * (1 - percent_range / 100)
    upper_bound = cutoff * (1 + percent_range / 100)

    # Mask for values below the lower bound
    below_lower = input_tensor <= lower_bound
    # Mask for values above the upper bound
    above_upper = input_tensor >= upper_bound

    # Compute the normalized value for smoothstep
    t = (input_tensor - lower_bound) / (upper_bound - lower_bound)
    # Clamp `t` to be between 0 and 1 (for safety, though not strictly necessary)
    t = torch.clamp(t, 0, 1)

    # Apply the smoothstep formula: t^2 * (3 - 2 * t)
    smooth = t * t * (3 - 2 * t)

    # Combine the results:
    # - Assign 0 where input is below the lower bound
    # - Assign 1 where input is above the upper bound
    # - Use smoothstep in between
    weighting = torch.where(
        condition=below_lower, input=torch.zeros_like(smooth),
        other=torch.where(
            condition=above_upper, input=torch.ones_like(smooth),
            other=smooth
        )
    )

    return weighting


def denoise(k_space: torch.Tensor, noise_scans: torch.Tensor,
            line_patch_size: int = 20, batch_size: int = 100,
            noise_dist_area_threshold: float = 0.9, ev_cutoff_percent_range: float = 15.0,
            visualization_path: plib.Path | str = None,
            device: torch.device = torch.get_default_device(), lr_svd: bool = False):
    log_module.info("deduce noise threshold from noise scans")
    log_module.info(f"assume noise scan dims (num_noise_scans, n_channels, n_samples) :: got {noise_scans.shape}")
    # we combine first the num_noise scans and samples as it should be uncorrelated iid noise
    noise_scans = torch.movedim(noise_scans, 1, 2).contiguous()
    noise_scans = noise_scans.view(-1, noise_scans.shape[-1])
    ns, nc = noise_scans.shape
    # we now want to get this to a batch dimension and a read as big as the k_space data
    nb = ns // k_space.shape[0]
    # if this is not an integer we make it one
    noise_scans = noise_scans[:nb * k_space.shape[0]]
    # get into correct shape
    noise_scans = noise_scans.view(nb, k_space.shape[0], nc)
    m = min(nc, line_patch_size)
    # input assumes (nr, npe, ns, nc, ne) - take batch num to be echoes
    noise_scans = torch.movedim(noise_scans, 0, -1)
    log_module.info(f"\t\tget singular values")
    _, noise_s_vals = denoise_data(
        k_space=noise_scans[:, None, None, :, :], device=device, batch_size=batch_size,
        line_patch_size=line_patch_size, lr_svd=lr_svd
    )

    log_module.info(f"\t\tdeduce noise s-val distribution")
    # get histogram
    hist, bins = torch.histogram(
        (noise_s_vals ** 2 / m).view(-1),
        bins=200
    )
    bins = bins[:-1] + torch.diff(bins) / 2

    hist[0] = 0
    hist /= hist.max()
    cs = torch.cumsum(hist, dim=0) / torch.sum(hist)
    cutoff_ind = torch.nonzero(cs < noise_dist_area_threshold)[-1].item()
    cutoff_val = bins[cutoff_ind].item()
    # for smooth_iter in range(3):
    #     w_ind = torch.nonzero(torch.cumsum(hist, dim=0) / torch.sum(hist) < noise_dist_area_threshold)[-1]
    #     hist[:w_ind] = 1
    #     hist[-hist.shape[0] // 30:] = 0
    #     hist = torch.from_numpy(
    #         gaussian_filter(hist.numpy(), sigma=5)
    #     )
    #     hist /= hist.max()
    #     hist[-hist.shape[0] // 30:] = 0
    #
    # weighting_function = 1 - hist

    if visualization_path is not None:
        visualization_path = plib.Path(visualization_path).absolute()
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=bins, y=hist, name="histogram")
        )
        fig.add_trace(
            go.Scatter(x=bins, y=weighting_fn(bins, cutoff_val, ev_cutoff_percent_range), name="weighting function")
        )
        fig.add_trace(
            go.Scatter(
                x=[cutoff_val, cutoff_val], y=[0, 1], mode="lines", name="cutoff val"
            )
        )
        fn = visualization_path.joinpath(f"noise-s-vals-histogram").with_suffix(".html")
        logging.info(f"write file: {fn}")
        fig.write_html(fn)

    log_module.info("Denoise k-space data using noise deduced pca threshold.")

    denoised_data, _ = denoise_data(
        k_space=k_space, line_patch_size=line_patch_size,
        ev_cutoff=cutoff_val, ev_cutoff_percent_range=ev_cutoff_percent_range,
        batch_size=batch_size, device=device, lr_svd=lr_svd
    )

    noise = k_space - denoised_data
    return denoised_data, noise


def denoise_data(k_space: torch.Tensor, line_patch_size: int = 0,
            ev_cutoff: float = None, ev_cutoff_percent_range: float = 15,
            batch_size: int = 100, device: torch.device = torch.get_default_device(),
            lr_svd: bool = False):

    log_module.info(f"Denoising k-space lines via patches across channels (nc).")
    log_module.info(f"Assume 5D data (nr, npe, ns, nc, ne). Found input shape: {k_space.shape}.")
    log_module.info(f"Assume read dimension is first dimension.")
    # assuming k-space dims [x, y, z, ch, t], cast if not the case
    while k_space.shape.__len__() < 5:
        k_space = k_space.unsqueeze(-1)
        log_module.info(f"\t\tExpanding shape to {k_space.shape}.")

    nr, npe, ns, nc, ne = k_space.shape
    if line_patch_size < 1:
        log_module.debug(f"Setting line patch size to channel dimension to maximize pca matrix. ({nc})")
        line_patch_size = nc

    log_module.debug(f"Assume sampling scheme equal across slices and channels. Usual 2D sequence behaviour.")
    mask = torch.abs(k_space[nr // 2, :, 0, 0]) > 1e-9

    # get number of sampled lines
    n_lines = torch.tensor([torch.count_nonzero(mask[:, e]) for e in range(ne)])
    if n_lines.unique().__len__() != 1:
        err = "found different number of sampled lines per echo. Not yet implemented."
        raise AttributeError(err)

    n_lines = n_lines[0].item()
    mask = mask.unsqueeze(0).unsqueeze(2).unsqueeze(2).expand((nr, npe, ns, nc, ne))
    sampled_k_lines = k_space[mask]
    # we possibly reduced the phase encode direction, but the masking gives 1D, hence move it back
    sampled_k_lines = sampled_k_lines.view(nr, n_lines, ns, nc, ne)

    log_module.info(f"denoising")
    # move batch dims to front and channels to back (nr, npe, ns, nc, ne) -> (npe, ns, ne, nr, nc)
    sampled_k_lines = torch.movedim(sampled_k_lines, -2, -1)
    sampled_k_lines = torch.movedim(sampled_k_lines, 0, -2).contiguous().view(-1, nr, nc)
    b, _, _ = sampled_k_lines.shape

    log_module.debug(f"\t\tbuild patch indices - ignore batch dim")
    indices, matrix_shape = get_linear_indices(
        k_space_shape=(nr, nc),
        patch_shape=(line_patch_size, -1), sample_directions=(1, 0)
    )
    # want to build matrix neighborhood in 2D
    nv = matrix_shape[1] // nc
    matrix_shape = (matrix_shape[0], nv, nc)
    if lr_svd:
        m = 10
    else:
        m = min(nv, nc)

    log_module.debug(f"\t\tset count matrix for recombination of patches.")
    count_matrix = torch.bincount(indices)
    count_matrix[count_matrix == 0] = 1
    count_matrix = count_matrix.view(nr, nc)[:, None, None, :, None]

    log_module.info(f"Set device: {device}")
    num_batches = int(np.ceil(b / batch_size))
    # allocate
    s_vals = torch.zeros((b, matrix_shape[0], m))
    denoised_lines = torch.zeros_like(sampled_k_lines).view(sampled_k_lines.shape[0], -1)
    torch.cuda.empty_cache()
    for i in tqdm.trange(num_batches, desc=f"Batch processing :: batch size {batch_size}"):
        batch_start = i * batch_size
        batch_end = min(b, (i + 1) * batch_size)
        bs = batch_end - batch_start
        # pick batch and index neighborhoods
        batch_k_lines = sampled_k_lines[batch_start:batch_end].view(
            bs, -1
        )[:, indices].view(
            bs * matrix_shape[0], *matrix_shape[1:]
        ).to(device)

        # do svd
        if lr_svd:
            u, s, v = torch.svd_lowrank(A=batch_k_lines, q=m, niter=2)
            v = v.mH
        else:
            u, s, v = torch.linalg.svd(batch_k_lines, full_matrices=False)

        # we assume we have a weighting function that weights singular values based on a weighting function
        # ev_weighting_function, defined at the eigenvalues (svals**2) ev_noise.

        # and we want to compute an interpolated weighting for the singular values we just calculated
        # to know which ones might belong to noise
        # if the weighting is not given we just keep everything and return only the s-vals
        # ToDo: build weighting function based on function definition for easier calculation.
        # No need to interpolate the weighting if we have an easy soft cutoff threshold function definition.
        s_wfn = weighting_fn(
            input_tensor=s**2 / m,
            cutoff=2*s.max()**2 / m if ev_cutoff is None else ev_cutoff,
            percent_range=ev_cutoff_percent_range
        )
        # recon with weighted singular values
        d = torch.matmul(torch.einsum("ilm, im -> ilm", u, (s*s_wfn).to(u.dtype)), v)
        s_vals[batch_start:batch_end] = s.cpu().view(bs, -1, m)

        d_lines = d.view(bs, *matrix_shape)
        denoised_lines[batch_start:batch_end] = denoised_lines[batch_start:batch_end].index_add(
            1, indices, d_lines.view(bs, -1).cpu()
        )
        torch.cuda.empty_cache()

    # move back dims, started here (ny, nz, ne, nx, nc)
    denoised_lines = denoised_lines.view(n_lines, ns, ne, nr, nc)
    denoised_lines = torch.movedim(denoised_lines, -2, 0)
    denoised_lines = torch.movedim(denoised_lines, -1, -2)
    denoised_lines /= count_matrix

    # build denoised data
    denoised_data = torch.zeros_like(k_space)
    denoised_data[mask] = denoised_lines.contiguous().view(-1)

    # move back dims svals (ny, nz, ne, num_patches, m)
    s_vals = s_vals.view(n_lines, ns, ne, -1, m)
    s_vals = torch.movedim(s_vals, -2, 0)
    s_vals = torch.movedim(s_vals, -1, -2).contiguous()
    return denoised_data, s_vals


def denoise_from_cli(settings: DenoiseSettingsKLC):
    # check path
    path_out = plib.Path(settings.out_path).absolute()
    log_module.info(f"Setting up output path {path_out}")
    path_out.mkdir(exist_ok=True, parents=True)

    # figures
    if settings.visualize:
        path_figs = path_out.joinpath("figs/")
        log_module.info(f"Setting up visualization path {path_figs}")
        path_figs.mkdir(exist_ok=True, parents=True)
    else:
        path_figs = None

    if settings.use_gpu:
        device = torch.device(f"cuda:{settings.gpu_device}")
    else:
        device = torch.device("cpu")

    # load data
    in_path = plib.Path(settings.in_k_space).absolute()
    k_space = torch_load(in_path)
    k_space = torch.movedim(k_space, settings.line_dir, 0)
    noise_scans = torch_load(settings.in_noise_scans)
    affine = torch_load(settings.in_affine)

    denoised_k, noise = denoise(
        k_space=k_space, noise_scans=noise_scans,
        line_patch_size=settings.line_patch_size, batch_size=settings.batch_size,
        noise_dist_area_threshold=settings.noise_dist_area_threshold,
        ev_cutoff_percent_range=settings.ev_cutoff_percent_range,
        visualization_path=path_figs,
        device=device
    )
    denoised_k = torch.movedim(denoised_k, 0, settings.line_dir)
    noise = torch.movedim(noise, 0, settings.line_dir)

    # save
    if settings.visualize:
        # do quick naive fft rsos recon
        img = fft(denoised_k, img_to_k=False, axes=(0, 1))
        img = root_sum_of_squares(img, dim_channel=-2)
        nifti_save(img, img_aff=affine, path_to_dir=settings.out_path, file_name=f"klc_naive_rsos_recon")

    torch_save(denoised_k, path_to_file=path_out, file_name=f'd_{in_path.stem}')
    torch_save(noise, path_to_file=path_out, file_name=f'd_noise_{in_path.stem}')


def main():
    # setup logging
    setup_program_logging(name="k-space line filter denoising based on SVT", level=logging.INFO)
    # setup parser
    parser, args = setup_parser(
        prog_name="k-space line channels SVT denoising",
        dict_config_dataclasses={"settings": DenoiseSettingsKLC}
    )
    # get config
    settings = DenoiseSettingsKLC.from_cli(args=args.settings, parser=parser)
    settings.display()

    try:
        denoise_from_cli(settings=settings)
    except Exception as e:
        parser.print_help()
        logging.exception(e)
        exit(-1)


if __name__ == '__main__':
    main()
