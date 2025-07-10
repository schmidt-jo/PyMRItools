"""
Sketch for optimizing the sampling scheme from torch autograd gradient loraks reconstruction of fully sampled data

"""
import pickle
import sys
import logging
import pathlib as plib
from enum import Enum, auto
import json

import torch
from pymritools.recon.loraks.loraks import Loraks, OperatorType, RankReduction, RankReductionMethod
from pymritools.recon.loraks.p_loraks import PLoraksOptions, LowRankAlgorithmType
from pymritools.recon.loraks.utils import unpad_output

from pymritools.recon.loraks.ac_loraks import AcLoraksOptions, SolverType, m_op_base
from pymritools.utils.functions import fft_to_img
from pymritools.utils import Phantom

p_tests = plib.Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(p_tests.as_posix())
from tests.utils import get_test_result_output_dir, ResultMode
from tests.experiments.loraks.utils import prep_k_space, unprep_k_space, DataType, load_data

import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc

logger = logging.getLogger(__name__)


class TorchEncoder(json.JSONEncoder):
    def default(self, obj):
        if torch.is_tensor(obj):
            return obj.tolist()
        return super().default(obj)


class LoraksType(Enum):
    P = auto()
    AC = auto()


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


#__ variant specific optimization functions
# def autograd_optimization_p(k: torch.Tensor,
#         rank: int, batch_size_channels: int,
#         max_num_iter=500, regularization_lambda=0.0, operator_type=OperatorType.S,
#         device: torch.device = torch.get_default_device()):
#     # P specific things
#     rank_reduction = RankReduction(method=RankReductionMethod.HARD_CUTOFF, value=rank)
#     options = PLoraksOptions(
#         loraks_matrix_type=operator_type,
#         rank=rank_reduction,
#         regularization_lambda=regularization_lambda, batch_size_channels=batch_size_channels,
#         max_num_iter=max_num_iter, patch_shape=(-1, 5, 5), sample_directions=(0, 1, 1),
#         device=device,
#     )
#     loraks = Loraks.create(options)
#     loraks = (loraks.
#               with_rank_reduction(rank_reduction).
#               with_sv_hard_cutoff(q=None, rank=rank_reduction.value).
#               with_lowrank_algorithm(algorithm_type=LowRankAlgorithmType.TORCH_LOWRANK_SVD, q=rank_reduction.value+2)
#               )
#     # we want to do optimization on the fully sampled data
#     prep_k, in_shape, padding, batch_channel_idx = prep_k_space(
#         k.unsqueeze_(2), batch_size_channels=batch_size_channels
#     )
#     grad_pb = torch.zeros_like(prep_k)
#     loraks._initialize(prep_k)
#     # send into one iteration
#     for b, batch_k in enumerate(prep_k):
#         k = batch_k.clone().to(device).requires_grad_()
#
#         # compute only LR loss
#         loss, _, _ = loraks.loss_func(
#             k_space_candidate=k,
#             indices=loraks.indices, matrix_shape=loraks.matrix_shape,
#             k_sampled_points=torch.zeros_like(k), sampling_mask=torch.zeros_like(k),
#             lam_s=0.0
#         )
#         loss.backward()
#
#         # get gradient
#         grad_pb[b] = k.grad.clone().cpu()
#
#     prep_k = unprep_k_space(prep_k, padding=padding, batch_idx=batch_channel_idx, input_shape=in_shape)
#     grad_pb = unprep_k_space(grad_pb, padding=padding, batch_idx=batch_channel_idx, input_shape=in_shape)
#     ac_mask = unpad_output(ac_mask, padding=padding[:2 * ac_mask.ndim]).mT
#     return grad_pb, prep_k, ac_mask, loraks


def autograd_optimization_ac(
        k: torch.Tensor,
        rank: int, batch_size_channels: int, num_ac_lines: int = 36,
        max_num_iter=500, regularization_lambda=0.0, operator_type=OperatorType.S,
        device: torch.device = torch.get_default_device()):
    # AC specific things
    # create AC options
    options = AcLoraksOptions(
        solver_type=SolverType.AUTOGRAD, regularization_lambda=regularization_lambda, max_num_iter=max_num_iter,
        loraks_matrix_type=operator_type, batch_size_channels=batch_size_channels
    )
    options.rank.value = rank
    loraks = Loraks.create(options=options)

    # create subsampling to let the algorithm work on the AC region
    phantom = Phantom.get_shepp_logan(shape=k.shape[:2], num_coils=k.shape[-2], num_echoes=k.shape[-1])
    tmp = phantom.sub_sample_ac_random_lines(acceleration=6, ac_lines=num_ac_lines)
    mask = tmp.abs() > 1e-11
    del tmp

    k_us = k.clone()
    k_us[~mask] = 0

    # we need to prepare the k-space
    prep_k_ac, in_shape, padding, batch_channel_idx = prep_k_space(
        k=k_us.unsqueeze_(2), batch_size_channels=batch_size_channels
    )

    loraks._initialize(k_space=prep_k_ac)

    # we want to prepare the batch to extract the AC region driven matrices
    batch_ac = prep_k_ac[0].to(device)
    # find the ac mask in k-space
    mask = batch_ac.abs() > 1e-10
    ac_mask = mask.sum(dim=0) == mask.shape[0]
    ac_mask = ac_mask.cpu()

    vc, vs = loraks._prepare_batch(batch=batch_ac)

    # we want to do the rest of the optimization on the fully sampled data
    prep_k, in_shape, padding, batch_channel_idx = prep_k_space(
        k.unsqueeze_(2), batch_size_channels=batch_size_channels
    )

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
    ac_mask = unpad_output(ac_mask, padding=padding[:2 * ac_mask.ndim]).mT
    return grad_pb, prep_k, ac_mask, loraks


def extract_sampling_density(grad: torch.Tensor, ac_mask: torch.Tensor):
    # we now have the gradients for all channels and echoes.
    # we want to compute a sampling scheme per echo in the following way
    # take the AC region per default
    # per echo we sum the absolute value of the gradients within all non ac lines per line
    # we sort the indices descending for highest gradient usage and declare those points "more important"
    # we take the mean across channels
    grad_am_cm = grad.abs().mean(dim=-2).squeeze()
    # we sum across the readout direction - and get rid here of the singular slice.
    # we can take this to a per slice sampling scheme if we wanted to
    grad_am = torch.sum(grad_am_cm, dim=0)
    # now we have an "importance" score per phase encode point, per echo.

    # remove ac indices
    indices_ac = torch.where(torch.sum(ac_mask.to(torch.int), dim=0) == ac_mask.shape[0])[0]

    grad_mac = grad_am.clone()
    grad_mac[indices_ac] = 0
    sampling_density = grad_mac / grad_mac.sum(dim=0)
    return sampling_density, grad_mac, indices_ac


def built_optimized_sampling_mask(acceleration: int, grad_mac: torch.Tensor, indices_ac: torch.Tensor, shape: tuple):
    n_phase = grad_mac.shape[0]
    n_echoes = grad_mac.shape[-1]
    # calculate how many we want
    n_sub = (n_phase - len(indices_ac)) // acceleration

    # we now want to draw from the indexes to keep, using the gradient function as sampling distribution per echo.
    # additionally we want to maximize the distribution to neighboring points across phase encodes and echoes.
    adaptive_sampler = AdaptiveSampler(p=grad_mac)
    us_idxs = adaptive_sampler.sample(n_sub, repulsion_radius=acceleration // 2)
    indices_sub = torch.concatenate((indices_ac[None,].expand(n_echoes, -1), us_idxs), dim=1)

    # reset size
    mask_sub = torch.zeros(n_phase, n_echoes)
    for i in range(n_echoes):
        mask_sub[indices_sub[i], i] = 1
    mask_sub = mask_sub[None, :, None, :].expand(shape).to(torch.int)
    return mask_sub


def create_subsampling_masks(shape):
    # create sub-sampling schemes
    nx, ny, nc, ne = shape
    phantom = Phantom.get_shepp_logan(shape=(nx, ny), num_coils=nc, num_echoes=ne)
    masks = []
    names = ["Fully Sampled", "Pseudo Rand.", "Weighted Rand.", "Skip (Grappa)", "Skip Interl.", "Optimized"]
    # fnames = ["fs", "pr", "wr", "s", "si", "opt"]
    for i, f in enumerate([
        phantom.sub_sample_ac_random_lines,
        phantom.sub_sample_ac_random_lines, phantom.sub_sample_ac_weighted_lines,
        phantom.sub_sample_ac_grappa, phantom.sub_sample_ac_skip_lines]):
        k_us = f(acceleration=6 if i > 0 else 1, ac_lines=36)

        masks.append({"name": names[i], "data": (k_us.abs() > 1e-10).to(torch.int)})
    return masks


def recon_sampling_patterns(k: torch.Tensor, bet: torch.Tensor, masks: list, loraks):
    options = loraks.options
    options.solver_type = SolverType.LEASTSQUARES
    options.max_num_iter = 40
    loraks.configure(options)
    loraks._initialize(k)
    # process sampling schemes
    results = []
    gt = 0
    bet = bet.to(torch.bool)

    # for each sub-sampled dataset
    for i, m in enumerate(masks):
        d = m["data"]
        tmp_dict = {"name": m["name"]}

        # save undersampled masked input
        # p = path.joinpath(f"k_slice_sub-{fnames[i]}").with_suffix(".pt")
        # if not p.is_file():
        km = k.clone().squeeze()
        km[~d.to(torch.bool)] = 0
        #     torch_save(data=km, path_to_file=p.parent, file_name=p.name)
        # else:
        #     km = torch_load(p)

        # km = km[:, :, :8].clone()
        im = fft_to_img(km, dims=(0, 1))

        if i > 0:
            # do recon
            prep_k, in_shape, padding, batch_idx = prep_k_space(k=km.unsqueeze(2), batch_size_channels=8)

            recon_k = loraks.reconstruct(k_space=prep_k)

            recon_k = unprep_k_space(recon_k, padding=padding, batch_idx=batch_idx, input_shape=in_shape).squeeze()
        else:
            recon_k = km.clone()
        recon_im = fft_to_img(recon_k, dims=(0, 1))
        rsos_rim = torch.sqrt(torch.sum(torch.square(recon_im.abs()), dim=-2))
        if i == 0:
            gt = rsos_rim.clone()
        delta = rsos_rim - gt
        delta[~bet[:, :, 0]] = 0
        tmp_dict["img_us"] = im
        tmp_dict["norm"] = torch.mean(delta.abs()).item()
        tmp_dict["rmse"] = torch.sqrt(torch.mean(delta.abs() ** 2))
        tmp_dict["delta"] = torch.nan_to_num(delta.abs() / gt, nan=0.0, posinf=0.0, neginf=0.0) * 100
        tmp_dict["k_us"] = km
        tmp_dict["k_recon"] = recon_k
        tmp_dict["img_recon"] = recon_im
        tmp_dict["img_recon_rsos"] = rsos_rim
        results.append(tmp_dict)
    return results


# __plotting
def plot_sampling_masks(masks: list, path: plib.Path):
    # plot sampling schemes
    fig = psub.make_subplots(
        rows=1, cols=len(masks),
        column_titles=[n["name"] for n in masks],
        vertical_spacing=0.02,
        y_title="Phase Encode Direction",
        shared_yaxes=True,
        x_title="Echo Number"
    )

    sample_column_width = 70
    # for each sub-sampled dataset
    for i, m in enumerate(masks):
        d = m["data"]
        n = m["name"]
        logger.info(f"__ plot sampling: {n} __")
        nx, ny, nc, ne = d.shape
        # plot mask
        m = torch.zeros((ny, ne * sample_column_width))
        for e in range(ne):
            m[
                :, e * sample_column_width:(e + 1) * sample_column_width
            ] = (e + 3) * d[d.shape[0] // 2, :, 0, e, None].expand(ny, sample_column_width)

        fig.add_trace(
            go.Heatmap(
                z=m, showscale=False, transpose=False, showlegend=False, colorscale="Inferno",
                zmin=0.0, zmax=ne + 2,
            ),
            col=1 + i, row=1
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


def plot_sampling_density(
        grad: torch.Tensor, sampling_density: torch.Tensor, ac_mask: torch.Tensor,
        path: plib.Path):
    ne = grad.shape[-1]
    # ensure real valued and magnitude data
    grad = grad.abs()
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
            y=[0, 0, 1.1 * torch.log(grad).max().item(), 1.1 * torch.log(grad).max().item(), 0],
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
        #         z=torch.log(grad_cm[..., i]), showlegend=False, colorscale="Inferno", showscale=False
        #     ),
        #     row=1, col=1 + i
        # )
        # fig.update_xaxes(visible=False, row=1, col=1 + i)
        # fig.update_yaxes(visible=False, row=1, col=1 + i)

        for h, gg in enumerate([torch.log(grad), sampling_density]):
            fig.add_trace(
                go.Scatter(
                    y=gg[..., i], showlegend=False, marker=dict(color=cmap[i]),
                    name=f"Echo {i + 1}", mode="lines", opacity=0.7
                ),
                row=1 + h, col=1
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

    fn = path.joinpath("sampling_density").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)
    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_image(fn)


def plot_results(results: list, path: plib.Path):
    # create figure
    fig = psub.make_subplots(
        rows=4, cols=len(results),
        column_titles=[n["name"] for n in results],
        row_titles=["Input FFT C.", "Recon. FFT C.", "RSOS", "RSOS - GT"],
        vertical_spacing=0.015, horizontal_spacing=0.015,
        x_title="Phase Encode Direction", y_title="Readout Direction", shared_xaxes=True, shared_yaxes=True,
    )

    # for plotting
    channel_no = 2
    plot_err_percentage = 15
    cmin = 0
    cmax = -1
    # for each sub-sampled dataset
    for i, r in enumerate(results):
        km = r["k_us"]
        im = r["img_us"]
        recon_im = r["img_recon"]
        rsos_rim = r["img_recon_rsos"]
        delta = r["delta"]


        for ki, kk in enumerate([im.abs(), recon_im.abs(), rsos_rim, delta]):
            coloraxis = "coloraxis" if ki == 3 else None
            if ki < 2:
                kk = kk[:, :, channel_no]
            pkm = kk[..., 0].clone()

            zmin = [0.0, 0.0, 0, 0]
            zmax = [pkm.max().item() * 0.5, pkm.max().item() * 0.5, pkm.max().item() * 0.5, plot_err_percentage]
            if max(zmax[1:]) > cmax:
                cmax = max(zmax[1:])
            fig.add_trace(
                go.Heatmap(
                    z=pkm, showscale=True if ki == 3 and i == 2 else False,
                    showlegend=False, transpose=False, colorscale="Inferno",
                    zmin=zmin[ki], zmax=zmax[ki], coloraxis=coloraxis,
                ),
                row=1 + ki, col=1 + i
            )

    fig.update_layout(
        width=1000,
        height=700,
        coloraxis=dict(
            colorscale="Inferno",
            cmin=0.0, cmax=plot_err_percentage,
            colorbar=dict(
                x=1.01, y=-0.01, len=0.25, thickness=14,
                title=dict(text="Error vs Fully Sampled [%]", side="right"),
                xanchor="left", yanchor="bottom",
            )
        ),
        margin=dict(t=25, b=55, l=65, r=5)
    )
    fn = path.joinpath("recon_vs_sampling").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)
    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_image(fn)

def plot_rmse(results: list, path: plib.Path):
    # create figure
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[n["name"] for n in results], y=[n["rmse"] for n in results],

        )
    )

    fig.update_layout(
        width=800,
        height=200,
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis=dict(title="RMSE [a.u.]")
    )

    fn = path.joinpath("recon_vs_sampling_rmse").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)
    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_image(fn)


#__ main function call
def subsampling_optimization_loraks(loraks_type: LoraksType, data_type: DataType, device: torch.device):
    # get path
    path_out = plib.Path(
        get_test_result_output_dir(
            f"autograd_subsampling_optimization",
            mode=ResultMode.EXPERIMENT)
    )
    path_out = path_out.joinpath(f"{loraks_type.name}".lower()).joinpath(f"{data_type.name}".lower())
    logger.info(f"Set Output Path: {path_out}")
    # set path for data output
    path_out_data = path_out.joinpath("data")
    path_out_data.mkdir(exist_ok=True, parents=True)

    logger.info(f"Set Data Output Path: {path_out_data}")

    # load data
    k, _, bet = load_data(data_type=data_type)

    # Extract sampling density
    path_tmp = path_out_data.joinpath("sampling_density_data").with_suffix(".pkl")
    if path_tmp.is_file():
        logger.info(f"Load data: {path_tmp}")
        with open(path_tmp, "rb") as f:
            sd = pickle.load(f)
        sampling_density = torch.tensor(sd["sampling_density"])
        grad_mac = torch.tensor(sd["grad_mac"])
        indices_ac = torch.tensor(sd["indices_ac"])
        mask = torch.tensor(sd["mask"])
        loraks = None
    else:
        # compute optimized sampling scheme
        if loraks_type == LoraksType.AC:
            grad, prep_k, mask, loraks = autograd_optimization_ac(
                k=k, device=device,
                batch_size_channels=8, rank=150
            )
        else:
            # grad, prep_k, mask, loraks = autograd_optimization_p(
            #     k=k, device=device,
            #     batch_size_channels=8, rank=150
            # )
            raise NotImplementedError(f"Loraks type {loraks_type} not implemented")
        sampling_density, grad_mac, indices_ac = extract_sampling_density(grad=grad, ac_mask=mask)
        logger.info(f"Save data: {path_tmp}")
        sd = {"sampling_density": sampling_density, "grad_mac": grad_mac, "indices_ac": indices_ac, "mask": mask}
        with open(path_tmp, "wb") as f:
            pickle.dump(sd, f)
        plot_sampling_density(grad=grad_mac, sampling_density=sampling_density, ac_mask=mask, path=path_out)

    # create sub-sampling masks for visuals
    path_tmp = path_out_data.joinpath("masks").with_suffix(".pkl")
    if path_tmp.is_file():
        with open(path_tmp.as_posix(), "rb") as  j_file:
            masks = pickle.load(j_file)
    else:
        masks = create_subsampling_masks(k.squeeze().shape)
        m_opt = built_optimized_sampling_mask(
            acceleration=6, grad_mac=grad_mac, indices_ac=indices_ac, shape=k.shape
        )
        masks.append({"name": "Optimized", "data": m_opt.squeeze()})
        with open(path_tmp.as_posix(), "wb") as j_file:
            pickle.dump(masks, j_file)
        plot_sampling_masks(masks, path=path_out)

    if loraks_type == LoraksType.AC:
        path_tmp = path_out_data.joinpath("ac_recon_results").with_suffix(".pkl")
        if path_tmp.is_file():
            with open(path_tmp.as_posix(), "rb") as  j_file:
                results = pickle.load(j_file)
        else:
            # compare sampling patterns
            options = AcLoraksOptions(
                solver_type=SolverType.AUTOGRAD, regularization_lambda=0.0, max_num_iter=500,
                loraks_matrix_type=OperatorType.S, batch_size_channels=8
            )
            options.rank.value = 150
            loraks = Loraks.create(options=options)
            results = recon_sampling_patterns(k=k, masks=masks, bet=bet, loraks=loraks)
            with open(path_tmp.as_posix(), "wb") as j_file:
                pickle.dump(results, j_file)
            plot_results(results, path=path_out)
        plot_rmse(results, path=path_out)

        logger.info(f"RMSE: {[r['rmse'] for r in results]}")
        logger.info(f"Norm: {[r['norm'] for r in results]}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    subsampling_optimization_loraks(
        loraks_type=LoraksType.AC, data_type=DataType.INVIVO,
        device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
    )
