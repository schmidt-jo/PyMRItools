import logging
import pathlib as plib

import torch
import numpy as np
import tqdm
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

from pymritools.config import setup_program_logging, setup_parser
from pymritools.config.processing import DenoiseSettingsKLC

from pymritools.utils import torch_load, torch_save, nifti_save, fft, root_sum_of_squares
from pymritools.utils.matrix_indexing import get_linear_indices
from pymritools.utils.functions import interpolate

log_module = logging.getLogger(__name__)


def denoise(k_space: torch.Tensor, noise_scans: torch.Tensor,
            line_patch_size: int = 0, batch_size: int = 100,
            noise_dist_area_threshold: float = 0.85, visualization_path: plib.Path | str = None,
            device: torch.device = torch.get_default_device()):
    log_module.info("deduce noise threshold from noise scans")
    log_module.info(f"assume noise scan dims (num_noise_scans, n_channels, n_samples) :: got {noise_scans.shape}")
    # we combine first the num_noise scans and samples as ist shold be uncorrelated iid noise
    noise_scans = torch.movedim(noise_scans, 1, 2).contiguous()
    noise_scans = noise_scans.view(-1, noise_scans.shape[-1])
    ns, nc = noise_scans.shape
    # we now want to get this to a batch dimension and a read as big as the k_space data
    nb = ns // k_space.shape[0]
    # if this is not an integer we make it one
    noise_scans = noise_scans[:nb * k_space.shape[0]]
    # get into correct shape
    noise_scans = noise_scans.view(nb, k_space.shape[0], nc)
    # input assumes (nr, npe, ns, nc, ne) - take batch num to be echoes
    noise_scans = torch.movedim(noise_scans, 0, -1)
    log_module.info(f"\t\tget singular values")
    _, noise_s_vals = denoise_data(
        k_space=noise_scans[:, None, None, :, :], device=device, batch_size=batch_size,
        line_patch_size=line_patch_size
    )

    log_module.info(f"\t\tdeduce noise s-val distribution")
    # get histogram
    hist, bins = torch.histogram(
        (noise_s_vals ** 2 / nc).view(-1),
        bins=200
    )
    bins = bins[:-1] + torch.diff(bins) / 2

    hist[0] = 0
    hist /= hist.max()
    for smooth_iter in range(3):
        w_ind = torch.nonzero(torch.cumsum(hist, dim=0) / torch.sum(hist) < noise_dist_area_threshold)[-1]
        hist[:w_ind] = 1
        hist[-hist.shape[0] // 30:] = 0
        hist = torch.from_numpy(
            gaussian_filter(hist.numpy(), sigma=5)
        )
        hist /= hist.max()
        hist[-hist.shape[0] // 30:] = 0

    weighting_function = 1 - hist

    if visualization_path is not None:
        visualization_path = plib.Path(visualization_path).absolute()
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=bins, y=hist, name="histogram")
        )
        fig.add_trace(
            go.Bar(x=bins, y=weighting_function, name="weighting function")
        )
        fig.add_trace(
            go.Scatter(x=bins, y=weighting_function, mode='markers', name="cutoff")
        )
        fn = visualization_path.joinpath(f"noise-s-vals-histogram").with_suffix(".html")
        logging.info(f"write file: {fn}")
        fig.write_html(fn)

    log_module.info("Denoise k-space data using noise deduced pca threshold.")

    denoised_data, _ = denoise_data(
        k_space=k_space, line_patch_size=line_patch_size,
        ev_noise=bins, ev_weighting_fn=weighting_function,
        batch_size=batch_size, device=device
    )

    noise = k_space - denoised_data
    return denoised_data, noise


def denoise_data(k_space: torch.Tensor, line_patch_size: int = 0,
            ev_noise: torch.Tensor = None, ev_weighting_fn: torch.Tensor = None,
            batch_size: int = 100, device: torch.device = torch.get_default_device()):

    log_module.info(f"Denoising k-space lines via patches across channels (nc).")
    log_module.info(f"Assume 5D data (nr, npe, ns, nc, ne). Found input shape: {k_space.shape}.")
    log_module.info(f"Assume read dimension is first dimension.")
    # assuming k-space dims [x, y, z, ch, t], cast if not the case
    while k_space.shape.__len__() < 5:
        k_space.unsqueeze(-1)
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

    # check weighting func
    if ev_noise is None and ev_weighting_fn is not None:
        log_module.info(f"Given eigenvalue weighting function but no noise eigenvalues. resetting to no weighting.")
        ev_weighting_fn = None
    if ev_noise is not None and ev_weighting_fn is None:
        log_module.info(f"Given eigenvalues but no weighting function. resetting to no weighting.")
        ev_noise = None
    if ev_noise is None and ev_weighting_fn is None:
        ev_noise = torch.linspace(0, 1, 10, device=device)
        ev_weighting_fn = torch.ones_like(ev_noise, device=device)
    else:
        ev_noise = ev_noise.to(device)
        ev_weighting_fn = ev_weighting_fn.to(device)

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
        u, s, v = torch.linalg.svd(batch_k_lines, full_matrices=False)

        # we assume we have a weighting function that weights singular values based on a weighting function
        # ev_weighting_function, defined at the eigenvalues (svals**2) ev_noise.

        # and we want to compute an interpolated weighting for the singular values we just calculated
        # to know which ones might belong to noise
        # if the weighting is not given we just keep everything and return only the s-vals
        s_wfn = torch.squeeze(
            interpolate(
                x=(s ** 2 / m).unsqueeze(1), xp=ev_noise, fp=ev_weighting_fn.unsqueeze(0)
            )
        )
        # recon with weighted singular values
        d = torch.matmul(torch.einsum("ilm, im -> ilm", u, (s*s_wfn).to(u.dtype)), v)
        s_vals[batch_start:batch_end] = s.cpu().view(bs, -1, m)

        d_lines = d.view(bs, *matrix_shape)
        denoised_lines[batch_start:batch_end] = denoised_lines[batch_start:batch_end].index_add(
            1, indices, d_lines.view(bs, -1).cpu()
        )

    # move back dims, started here (ny, nz, ne, nx, nc)
    denoised_lines = denoised_lines.view(n_lines, ns, ne, nr, nc)
    denoised_lines = torch.movedim(denoised_lines, -2, 0)
    denoised_lines = torch.movedim(denoised_lines, -1, -2)
    denoised_lines /= count_matrix

    # build denoised data
    denoised_data = torch.zeros_like(k_space)
    denoised_data[mask] = denoised_lines.contiguous().view(-1)
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
    noise_scans = torch_load(settings.in_noise_scans)
    affine = torch_load(settings.in_affine)

    denoised_k, noise = denoise(
        k_space=k_space, noise_scans=noise_scans,
        line_patch_size=0, batch_size=settings.batch_size,
        noise_dist_area_threshold=settings.noise_dist_area_threshold,
        visualization_path=path_figs,
        device=device
    )

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
    setup_program_logging(name="MP distribution k-space filter denoising", level=logging.INFO)
    # setup parser
    parser, args = setup_parser(
        prog_name="MP distribution k-space filter denoising",
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
