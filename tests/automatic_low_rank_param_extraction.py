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
    shape = (256, 256, 8, 4)
    # for plot

    fig = psub.make_subplots(rows=2, cols=1)

    num_acc = 15
    cmap = plc.sample_colorscale("Turbo", torch.linspace(0.1, 0.9, 4 * num_acc).tolist())
    c_max_val = -1
    s_max_val = -1

    for ai, acc in enumerate(torch.arange(1,num_acc + 1).tolist()):
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
        # create s_matrix
        s_matrix = s_operator(k_space=sl_us, indices=loraks_patch_mapping)

        # get svd vals
        c_s = torch.linalg.svdvals(c_matrix)
        s_s = torch.linalg.svdvals(s_matrix)

        # calculate area under cummulatively
        c_total_area = torch.sum(c_s)
        c_cum_area = torch.cumsum(c_s, dim=0)
        # find threshold
        c_th = torch.where(c_cum_area > 0.95 * c_total_area)[0][0]

        c_ax = torch.arange(c_s.shape[0]) + 1

        s_total_area = torch.sum(s_s)
        s_cum_area = torch.cumsum(s_s, dim=0)
        # find threshold
        s_th = torch.where(s_cum_area > 0.95 * s_total_area)[0][0]

        s_ax = torch.arange(s_s.shape[0]) + 1

        if torch.max(c_s) > c_max_val:
            c_max_val = torch.max(c_s)
        if torch.max(s_s) > s_max_val:
            s_max_val = torch.max(s_s)

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

    output_dir = get_test_result_output_dir(test_automatic_low_rank_param_extraction)
    fn = f"SVD_Spectrum_with_us"
    fig.write_html(os.path.join(output_dir, f"{fn}.html"))

