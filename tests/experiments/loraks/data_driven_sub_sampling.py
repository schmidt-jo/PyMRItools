"""
Sketch for optimizing the sampling scheme from torch autograd gradient loraks reconstruction of fully sampled data

"""
import logging
import pathlib as plib
from typing import Tuple

import torch
from pymritools.recon.loraks.loraks import Loraks, OperatorType
from pymritools.recon.loraks.utils import (
    prepare_k_space_to_batches, pad_input, unprepare_batches_to_k_space, unpad_output,
    check_channel_batch_size_and_batch_channels
)
from pymritools.recon.loraks.ac_loraks import AcLoraksOptions, SolverType, m_op_base
from pymritools.utils.functions import fft_to_img
from pymritools.utils import torch_load, Phantom, torch_save
from tests.utils import get_test_result_output_dir, create_phantom, ResultMode

import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc

logger = logging.getLogger(__name__)


class AdaptiveSampler:
    def __init__(self, p: torch.Tensor):
        """
        Parameters:
        - p: 2D importance distribution
        - n1, n2: dimensions of the sampling space
        - n_sub: number of samples to draw per dimension
        """
        self.p = p / p.sum()  # Normalize distribution
        self.n1, self.n2 = p.shape

    def sample(self, n_sub: int, repulsion_radius: int = 2):
        """
        Sample with importance and spatial diversity

        Args:
        - repulsion_radius: Minimum distance between selected points
        """
        # Initial sampling based on importance
        samples = []

        # Flattened importance distribution
        flat_dist = self.p.ravel()

        for e in range(self.n2):
            # Candidate points for this dimension
            e_candidates = torch.where(self.p[:, e] > 0)[0]

            # Weighted sampling of candidates
            e_weights = self.p[e_candidates, e]
            e_weights /= e_weights.sum()

            # Track selected points for this dimension
            dim_samples = []

            while len(dim_samples) < n_sub:
                # Sample candidate with probability proportional to importance
                # candidate = np.random.choice(e_candidates, p=e_weights)
                candidate_idx = e_weights.multinomial(num_samples=1).item()
                candidate = e_candidates[candidate_idx]

                # Check repulsion condition
                if not self._is_too_close(candidate, dim_samples, repulsion_radius):
                    dim_samples.append(candidate)

            samples.append(dim_samples)

        return torch.tensor(samples)

    @staticmethod
    def _is_too_close(point, existing_points, radius):
        """
        Check if point is too close to existing points
        """
        return any(abs(point - ex) < radius for ex in existing_points)


def prep_k_space(k: torch.Tensor, batch_size_channels: int = -1):
    # we need to prepare the k-space
    batch_channel_idx = check_channel_batch_size_and_batch_channels(
        k_space_rpsct=k, batch_size_channels=batch_size_channels
    )
    prep_k, in_shape = prepare_k_space_to_batches(
        k_space_rpsct=k, batch_channel_indices=batch_channel_idx
    )
    prep_k, padding = pad_input(prep_k, sampling_dims=(-2, -1))
    return prep_k, in_shape, padding, batch_channel_idx


def unprep_k_space(k: torch.Tensor, padding: Tuple[int, int], batch_idx: torch.tensor, input_shape: Tuple):
    k = unpad_output(k_space=k, padding=padding)

    return unprepare_batches_to_k_space(
        k_batched=k, batch_channel_indices=batch_idx, original_shape=input_shape
    )


def create_phantom_data_fs_ac():
    nx, ny, nc, ne = (156, 140, 4, 2)
    phantom = create_phantom(shape_xyct=(nx, ny, nc, ne), acc=1).unsqueeze(2)
    phantom_ac_reg = create_phantom(shape_xyct=(156, 140, 4, 2), acc=30).unsqueeze(2)
    return phantom, phantom_ac_reg


def autograd_subsampling_optimization(data: torch.Tensor, data_ac: torch.Tensor, rank: int, batch_size_channels: int = -1,):
    options = AcLoraksOptions(
        solver_type=SolverType.AUTOGRAD, regularization_lambda=0.0, max_num_iter=500,
        loraks_matrix_type=OperatorType.S, batch_size_channels=batch_size_channels
    )
    options.rank.value = rank
    loraks = Loraks.create(options=options)

    # set device
    device = torch.device("cuda:0")

    # we need to prepare the k-space
    prep_k_ac, in_shape, padding, batch_channel_idx = prep_k_space(k=data_ac, batch_size_channels=batch_size_channels)

    loraks._initialize(k_space=prep_k_ac)

    # we want to prepare the batch to extract the AC region driven matrices
    batch_ac = prep_k_ac[0].to(device)
    # find the ac mask in k-space
    mask = batch_ac.abs() > 1e-10
    ac_mask = mask.sum(dim=0) == mask.shape[0]
    ac_mask = ac_mask.cpu()

    vc, vs = loraks._prepare_batch(batch=batch_ac)

    # we want to do the rest of the optimization on the fully sampled data
    prep_k, in_shape, padding, batch_channel_idx = prep_k_space(data, batch_size_channels=batch_size_channels)

    grad_pb = torch.zeros_like(prep_k)
    # send into one iteration
    for b, batch_k in enumerate(prep_k):
        k = batch_k.clone().to(device).requires_grad_()

        # compute only LR loss
        mv = m_op_base(
            x=k, v_c=vc, v_s=vs, nb_size=options.loraks_neighborhood_size, shape_batch=prep_k.shape[1:]
        )

        loss = torch.linalg.norm(mv)
        loss.backward()

        # get gradient
        grad_pb[b] = k.grad.clone().cpu()

    prep_k = unprep_k_space(prep_k, padding=padding, batch_idx=batch_channel_idx, input_shape=in_shape)
    grad_pb = unprep_k_space(grad_pb, padding=padding, batch_idx=batch_channel_idx, input_shape=in_shape)
    ac_mask = unpad_output(ac_mask, padding=padding[:2*ac_mask.ndim])
    return grad_pb, prep_k, ac_mask


def autograd_subsampling_optimization_sl():
    path = plib.Path(
        get_test_result_output_dir(autograd_subsampling_optimization_sl, mode=ResultMode.EXPERIMENT)
    )
    data, data_ac = create_phantom_data_fs_ac()

    nx, ny, ns, nc, ne = data.shape
    grad, k, ac_mask = autograd_subsampling_optimization(data=data, data_ac=data_ac, rank=20)

    fig = psub.make_subplots(
        rows=3, cols=nc * ne,
        row_titles=["Img", "K-space", "Grad"],
        vertical_spacing=0.02, horizontal_spacing=0.02
    )
    for i, d in enumerate([k.squeeze(), k.squeeze(), grad]):
        d = d.detach().cpu()
        if i == 0:
            d = fft_to_img(d, dims=(-1, -2)).abs()
        elif i == 1:
            d = torch.log(d.abs())
        else:
            d = d.abs()
            d = d > 0.9 * d.mean()
            d = torch.logical_or(d, ac_mask.unsqueeze(0).expand(d.shape))
            d = d.to(torch.int)
        zmax = d.abs().max().item() * 0.5
        for h, e in enumerate(d.cpu()):
            fig.add_trace(
                go.Heatmap(
                    z=e, transpose=True, showlegend=False, showscale=i == 2 and h == 0,
                    colorscale="Inferno",
                    # zmin=0, zmax=zmax if i == 2 else None
                ),
                row=1 + i, col=1 + h
            )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fn = path.joinpath("plot").with_suffix(".html")
    print(f"Write file: {fn}")
    fig.write_html(fn)


def autograd_subsampling_optimization_iv():
    # load input data fully sampled
    path = plib.Path(
        get_test_result_output_dir(subsampling_optimization, mode=ResultMode.EXPERIMENT)
    )

    k = torch_load(path.joinpath("fs_data_slice.pt"))

    # create sub-sampling schemes
    nx, ny, nc, ne = k.shape
    phantom = Phantom.get_shepp_logan(shape=(ny, nx), num_coils=nc, num_echoes=ne)
    phantom_ac = phantom.sub_sample_ac_random_lines(acceleration=6, ac_lines=36).permute(1, 0, 2, 3)
    mask = phantom_ac.abs() > 1e-10

    k_us = k.clone()
    k_us[~mask] = 0

    grad, k, ac_mask = autograd_subsampling_optimization(
        data=k.unsqueeze_(2), data_ac=k_us.unsqueeze_(2), rank=150, batch_size_channels=8
    )

    # we now have the gradients for all channels and echoes.
    # we want to compute a sampling scheme per echo in the following way
    # take the AC region per default
    # per echo we sum the absolute value of the gradients within all non ac lines per line
    # we sort the indices descending for highest gradient usage and declare those points "more important"
    # we take the mean across channels
    grad_am_cm = grad.abs().mean(dim=-2).squeeze()
    # we sum across the readout direction - and get rid here of the singular slice.
    # we can take this to a per slice sampling scheme if we wanted to
    grad_am = torch.sum(grad_am_cm, dim=1)
    # now we have an "importance" score per phase encode point, per echo.

    # remove ac indices
    indices_ac = torch.where(torch.sum(ac_mask.to(torch.int), dim=0) == ac_mask.shape[0])[0]

    grad_mac = grad_am.clone()
    grad_mac[indices_ac] = 0
    sampling_density = grad_mac / grad_mac.sum(dim=0)
    cmap = plc.sample_colorscale("Inferno", ne, 0.1, 0.9)

    fig = psub.make_subplots(
        rows=2, cols=1,
        row_titles=["Grad. Density", "Sampling Distribution"],
        y_title="Magnitude [a.u.]",
        x_title="Phase Encode Direction"
    )
    p_min = torch.where(ac_mask[0])[0][0].item()
    p_max = torch.where(ac_mask[0])[0][-1].item()
    fig.add_trace(
        go.Scatter(
            x=[p_min, p_max, p_max, p_min, p_min],
            y=[0, 0, 1.1*torch.log(grad_am).max().item(), 1.1*torch.log(grad_am).max().item(), 0],
            mode="lines", fill="toself", line=dict(width=0), name="AC Region",
            showlegend=False, marker=dict(color="#B6E880"), opacity=0.8
        ),
        row=1, col=1
    )
    fig.add_annotation(
        text="AC Region", x=p_min, y=0, xref="x", yref="y",
        xanchor="left", yanchor="bottom", showarrow=False,
    )
    for i in range(ne):
        # fig.add_trace(
        #     go.Heatmap(
        #         z=torch.log(grad_am_cm[..., i]), showlegend=False, colorscale="Inferno", showscale=False
        #     ),
        #     row=1, col=1 + i
        # )
        # fig.update_xaxes(visible=False, row=1, col=1 + i)
        # fig.update_yaxes(visible=False, row=1, col=1 + i)

        for h, gg in enumerate([torch.log(grad_am), sampling_density]):
            fig.add_trace(
                go.Scatter(
                    y=gg[..., i], showlegend=False, marker=dict(color=cmap[i]),
                    name=f"Echo {i + 1}", mode="lines", opacity=0.7
                ),
                row=1+h, col=1
            )
    # add colorbar
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            marker=dict(
                size=ne / 100,
                color=[1, ne],
                colorscale="Inferno",
                colorbar=dict(title="Echo", thickness=10),
                showscale=True
            ),
            showlegend=False
        )
    )

    fig.update_layout(
        width=800,
        height=350,
        margin=dict(t=15, b=55, l=65, r=5)
    )

    path = plib.Path(get_test_result_output_dir(autograd_subsampling_optimization_iv, mode=ResultMode.EXPERIMENT))
    fn = path.joinpath("sampling_density").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)
    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_image(fn)
    # calculate how many we want

    acceleration = 6
    n_sub = (nx - len(indices_ac)) // acceleration

    # we now want to draw from the indexes to keep, using the gradient function as sampling distribution per echo.
    # additionally we want to maximize the distribution to neighboring points across phase encodes and echoes.AvE!!ow92_pHd33
    adaptive_sampler = AdaptiveSampler(p=grad_mac)
    us_idxs = adaptive_sampler.sample(n_sub, repulsion_radius=acceleration // 2)
    indices_sub = torch.concatenate((indices_ac[None, ]. expand(ne, -1), us_idxs), dim=1)

    # reset size
    mask_sub = torch.zeros(nx, ne)
    for i in range(ne):
        mask_sub[indices_sub[i], i] = 1
    mask_sub = mask_sub[:, None, None, None, :].expand(k.shape).to(torch.int)
    return mask_sub


def subsampling_optimization():
    # load input data fully sampled
    path = plib.Path(
        get_test_result_output_dir(subsampling_optimization, mode=ResultMode.EXPERIMENT)
    )

    k = torch_load(path.joinpath("fs_data_slice.pt"))

    # create sub-sampling schemes
    nx, ny, nc, ne = k.shape
    phantom = Phantom.get_shepp_logan(shape=(ny, nx), num_coils=nc, num_echoes=ne)
    masks = []
    names = ["Fully Sampled", "Pseudo Rand.", "Weighted Rand.", "Skip (Grappa)", "Skip Interl.", "Optimized"]
    # fnames = ["fs", "pr", "wr", "s", "si", "opt"]
    for i, f in enumerate([
        phantom.sub_sample_ac_random_lines,
        phantom.sub_sample_ac_random_lines, phantom.sub_sample_ac_weighted_lines,
        phantom.sub_sample_ac_grappa, phantom.sub_sample_ac_skip_lines]):
        k_us = f(acceleration=6 if i > 0 else 1, ac_lines=36).permute(1, 0, 2, 3)

        masks.append((k_us.abs() > 1e-10).to(torch.int))

    # compute optimized sampling scheme
    m_opt = autograd_subsampling_optimization_iv(data=k.clone())
    masks.append(m_opt.squeeze())

    # plot and process
    ac_opts = AcLoraksOptions(
        loraks_matrix_type=OperatorType.S, regularization_lambda=0.0, batch_size_channels=8, max_num_iter=50,
        solver_type=SolverType.LEASTSQUARES
    )
    ac_opts.rank.value = 150
    ac_loraks = Loraks.create(ac_opts)

    # plot sampling schemes
    fig = psub.make_subplots(
        rows=1, cols=len(names),
        column_titles=names,
        vertical_spacing=0.02,
        y_title="Phase Encode Direction",
        shared_yaxes=True,
        x_title="Echo Number"
    )

    sample_column_width = 70
    # for each sub-sampled dataset
    for i, d in enumerate(masks):
        logger.info(f"__ plot sampling: {names[i]} __")
        # plot mask
        m = torch.zeros((nx, ne * sample_column_width))
        for e in range(ne):
            m[
            :, e * sample_column_width:(e + 1) * sample_column_width
            ] = (e + 3) * d[:, d.shape[1] // 2, 0, e, None].expand(nx, sample_column_width)

        fig.add_trace(
            go.Heatmap(
                z=m, showscale=False, transpose=False, showlegend=False, colorscale="Inferno",
            ),
            col=1+i, row=1
        )
        fig.update_xaxes(
            tickmode="array", ticktext=torch.arange(1, 1 + ne).to(torch.int).numpy(),
            tickvals=(0.5 + torch.arange(ne)) * sample_column_width, row=1, col=1 + i,
            # title="Echo Number"
        )

    fig.update_layout(
        width=1000,
        height=350,
        margin=dict(t=25, b=55, l=65, r=5)
    )
    fn = path.joinpath(f"sampling_patterns").with_suffix(".html")
    logger.info(f"write file: {fn}")
    fig.write_html(fn)
    for suff in [".pdf", ".png"]:
        fn = fn.with_suffix(suff)
        logger.info(f"write file: {fn}")
        fig.write_image(fn)

    # create figure
    fig = psub.make_subplots(
        rows=5, cols=len(names),
        column_titles=names,
        row_titles=["Input K", "Input FFT", "Recon. FFT", "RSOS", "RSOS - GT"],
        vertical_spacing=0.015, horizontal_spacing=0.015,
        x_title="Phase Encode Direction", y_title="Readout Direction", shared_xaxes=True, shared_yaxes=True,
    )
    gt = 0
    norms = []
    rmse = []
    plot_err_percentage = 15

    # for each sub-sampled dataset
    for i, d in enumerate(masks):
        # save undersampled masked input
        # p = path.joinpath(f"k_slice_sub-{fnames[i]}").with_suffix(".pt")
        # if not p.is_file():
        km = k.clone()
        km[~d.to(torch.bool)] = 0
        #     torch_save(data=km, path_to_file=p.parent, file_name=p.name)
        # else:
        #     km = torch_load(p)

        # km = km[:, :, :8].clone()
        im = fft_to_img(km, dims=(0, 1))

        if i > 0:
            # do recon
            prep_k, in_shape, padding, batch_idx = prep_k_space(k=km.unsqueeze(2), batch_size_channels=8)

            recon_k = ac_loraks.reconstruct(k_space=prep_k)

            recon_k = unprep_k_space(recon_k, padding=padding, batch_idx=batch_idx, input_shape=in_shape).squeeze()
        else:
            recon_k = km.clone()
        recon_im = fft_to_img(recon_k, dims=(0, 1))
        rsos_rim = torch.sqrt(torch.sum(torch.square(recon_im.abs()), dim=-2))
        if i == 0:
            gt = rsos_rim.clone()
        delta = rsos_rim - gt
        norms.append(torch.mean(delta.abs()).item())
        rmse.append(torch.sqrt(torch.mean(delta.abs() ** 2)))
        delta = torch.nan_to_num(delta.abs() / gt, nan=0.0, posinf=0.0, neginf=0.0) * 100

        # for plotting
        channel_no = 2

        for ki, kk in enumerate([torch.log(km.abs()), im.abs(), recon_im.abs(), rsos_rim, delta]):
            if ki < 3:
                kk = kk[:, :, channel_no]
            pkm = kk[..., 0].clone()

            zmin = [None, 0.0, 0.0, 0, 0]
            zmax = [None, pkm.max().item() * 0.5, pkm.max().item() * 0.5, pkm.max().item() * 0.5, plot_err_percentage]
            row = 1 + ki
            fig.add_trace(
                go.Heatmap(
                    z=pkm, showscale=True if ki == 4 and i == 2 else False,
                    showlegend=False, transpose=True, colorscale="Inferno",
                    zmin=zmin[ki], zmax=zmax[ki]
                ),
                row=1+ki, col=1+i
            )

    fig.update_layout(
        width=1000,
        height=800,
        margin=dict(t=25, b=55, l=65, r=5)
    )
    fn = path.joinpath("recon_vs_sampling").with_suffix(".html")
    print(f"Write file: {fn}")
    fig.write_html(fn)
    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        print(f"Write file: {fn}")
        fig.write_image(fn)

    logger.info("Plot Norms")
    print(norms)
    print(rmse)
    # cmap = plc.sample_colorscale("Inferno", len(norms), 0.2, 0.9)
    # fig = go.Figure()
    # fig.add_trace(
    #     go.Scatter(
    #         y=[None, *norms[1:]], x=names, showlegend=False,
    #         line=dict(color=cmap[0]),
    #     ),
    # )
    #
    # fig.update_layout(
    #     yaxis=dict(title="|| Recon - Ground Truth || [a.u.]"),
    #     width=800,
    #     height=250,
    #     margin=dict(t=5, b=5, l=5, r=5)
    # )
    # fn = path.joinpath("recon_norm").with_suffix(".html")
    # print(f"Write file: {fn}")
    # fig.write_html(fn)
    # for suff in [".png", ".pdf"]:
    #     fn = fn.with_suffix(suff)
    #     print(f"Write file: {fn}")
    #     fig.write_image(fn)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    autograd_subsampling_optimization_iv()
