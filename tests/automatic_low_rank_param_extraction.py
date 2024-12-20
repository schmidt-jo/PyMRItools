import os

import torch
import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc

from pymritools.utils.phantom import SheppLogan
from pymritools.recon.loraks_dev.operators import c_operator, s_operator
from pymritools.recon.loraks_dev.matrix_indexing import get_all_idx_nd_square_patches_in_nd_shape
from tests.utils import get_test_result_output_dir

def test_automatic_low_rank_param_extraction():
    shape = (256, 256, 6, 3)
    # for plot

    fig = psub.make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{}, {}]
        ]
    )

    num_acc = 15
    cmap = plc.sample_colorscale("Turbo", torch.linspace(0.1, 0.9, 4 * num_acc).tolist())
    c_max_val = -1
    s_max_val = -1

    accs = torch.arange(1,num_acc + 1)
    c_ths = []
    c_ratios = []
    s_ths = []
    s_ratios = []
    for ai, acc in enumerate(accs.tolist()):
        # get phantom
        sl_us = SheppLogan().get_sub_sampled_k_space(
            shape=shape[:2], acceleration=acc, ac_lines=30, mode="skip", as_torch_tensor=True,
            num_echoes=shape[-1], num_coils=shape[-2]
        )

        # get indices
        loraks_patch_mapping, reshape = get_all_idx_nd_square_patches_in_nd_shape(
            k_space_shape=shape, size=5, patch_direction=(1, 1, 0, 0), combination_direction=(0, 0, 1, 1)
        )
        sl_us = torch.reshape(sl_us, reshape)

        # take first batch (should be only one batch)
        sl_us = sl_us[0]
        loraks_patch_mapping = loraks_patch_mapping[0]

        # create c matrix
        c_matrix = c_operator(k_space=sl_us, indices=loraks_patch_mapping)
        # count nonzero
        c_nz = torch.count_nonzero(c_matrix)
        c_size = torch.prod(torch.tensor(c_matrix.shape))
        c_ratios.append(c_nz / c_size)

        # create s_matrix
        s_matrix = s_operator(k_space=sl_us, indices=loraks_patch_mapping)
        s_nz = torch.count_nonzero(s_matrix)
        s_size = torch.prod(torch.tensor(s_matrix.shape))
        s_ratios.append(s_nz / s_size)

        # get svd vals
        c_s = torch.linalg.svdvals(c_matrix)
        s_s = torch.linalg.svdvals(s_matrix)

        # scale to max 1
        c_s /= torch.max(c_s)
        s_s /= torch.max(s_s)

        # calculate area under cummulatively
        c_total_area = torch.sum(c_s)
        c_cum_area = torch.cumsum(c_s, dim=0)
        # find threshold
        c_max = torch.max(c_s)
        # estimate 95 % of area under curve
        c_th = torch.where(c_cum_area > 0.9 * c_total_area)[0][0]

        c_ths.append(c_th)

        c_ax = torch.arange(c_s.shape[0]) + 1

        s_total_area = torch.sum(s_s)
        s_cum_area = torch.cumsum(s_s, dim=0)
        # find threshold
        s_max = torch.max(s_s)
        s_th = torch.where(s_cum_area > 0.9 * s_total_area)[0][0]

        s_ths.append(s_th)

        s_ax = torch.arange(s_s.shape[0]) + 1

        if c_max > c_max_val:
            c_max_val = c_max
        if s_max > s_max_val:
            s_max_val = s_max
        for i, s in enumerate([c_s, s_s]):
            showlegend = True if i == 0 else False
            ax = [c_ax, s_ax][i]
            th = [c_th, s_th][i]
            max_val = [c_max_val, s_max_val][i]
            fig.add_trace(
                go.Scattergl(
                    x=ax, y=s, name=f"Acc.: {acc}", marker=dict(color=cmap[4*ai]), showlegend=False,
                    legendgroup=ai
                ),
                row=i+1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[th, th], y=[0, max_val], name=f"Acc. {acc}, Th. (95 %) C: {c_th:d}, S: {s_th:d}", mode="lines",
                    line=dict(color=cmap[4*ai + 1]), legendgroup=ai, showlegend=showlegend,
                ),
                row=i+1, col=1,
            )

    # simple approximate rank extraction
    # we take rank_0 to be the rank of the non-accelerated case and want the others to match.
    # We have the additional information of 0 filling of the matrix given the amount of acceleration (the ratio)
    # We have the computed threshold at 90% of singular value spectrum
    # try simple linear dependence: rank = th + alpha * (ratio - 1).
    # extract alpha
    c_ratios = torch.tensor(c_ratios)
    s_ratios = torch.tensor(s_ratios)
    c_ths = torch.tensor(c_ths).to(c_ratios.dtype)
    s_ths = torch.tensor(s_ths).to(s_ratios.dtype)
    b =  (c_ths[0] - c_ths[1:])
    a = (c_ratios[1:] - 1)
    alpha_c = torch.dot(a, b) / torch.dot(a.T, a)
    b =  (s_ths[0] - s_ths[1:])
    a = (s_ratios[1:] - 1)
    alpha_s = torch.dot(a, b) / torch.dot(a, a)

    for i, th in enumerate([c_ths, s_ths]):
        th = torch.tensor(th)
        r = torch.tensor([c_ratios, s_ratios][i])
        a = [alpha_c, alpha_s][i]

        fig.add_trace(
            go.Scatter(
                x=accs, y=th, name=["C th", "S th"][i]
            ), row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=accs, y=th + (r - 1) * a , name=["C rank extr.", "S rank extr."][i]
            ), row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=accs, y=[c_ratios, s_ratios][i], name=["C ratio", "S ratio"][i]
            ), row=3, col=2
        )
    output_dir = get_test_result_output_dir(test_automatic_low_rank_param_extraction)
    fn = f"SVD_Spectrum_with_us"
    fig.write_html(os.path.join(output_dir, f"{fn}.html"))


