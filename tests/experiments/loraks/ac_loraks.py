import logging
from tests.experiments.utils import get_output_dir, create_phantom_data
from pymritools.recon.loraks_dev_cleanup.loraks import Loraks, OperatorType
from pymritools.recon.loraks_dev_cleanup.ac_loraks import AcLoraksOptions, SolverType
from pymritools.recon.loraks_dev_cleanup.utils import (
    prepare_k_space_to_batches, unprepare_batches_to_k_space, check_channel_batch_size_and_batch_channels,
    pad_input, unpad_output
)
from pymritools.utils import fft_to_img

import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc

log_module = logging.getLogger(__name__)


def main():
    log_module.info(f"Create phantom")
    k = create_phantom_data((140, 128, 4, 3)).unsqueeze(2)

    log_module.info(f"create ac loraks object")
    ac_opts = AcLoraksOptions(
        loraks_matrix_type=OperatorType.S, regularization_lambda=0.0,
        solver_type=SolverType.AUTOGRAD, max_num_iter=250
    )

    ac_loraks = Loraks.create(ac_opts)

    log_module.info("prepare k - space")
    batch_size_channels = 2

    batch_channel_indices = check_channel_batch_size_and_batch_channels(
        k_space_rpsct=k, batch_size_channels=batch_size_channels
    )

    k_batched, input_shape = prepare_k_space_to_batches(
        k_space_rpsct=k, batch_channel_indices=batch_channel_indices
    )

    k_batched, padding = pad_input(k_batched)

    log_module.info("reconstruction")
    k_recon = ac_loraks.reconstruct(k_batched)

    log_module.info("unprepare")

    k_recon = unpad_output(k_space=k_recon, padding=padding)

    k_recon = unprepare_batches_to_k_space(
        k_batched=k_recon, batch_channel_indices=batch_channel_indices, original_shape=input_shape
    )

    img_in = fft_to_img(k[:, :, 0, 0, 0], dims=(0, 1))
    img_recon = fft_to_img(k_recon[:, :, 0, 0, 0], dims=(0, 1))
    path = get_output_dir("ac_loraks")
    fig = psub.make_subplots(
        rows=1, cols=2
    )
    for i, d in enumerate([img_in, img_recon]):
        fig.add_trace(
            go.Heatmap(z=d.abs()),
            row=1, col=1 + i
        )
    fn = path.joinpath("recon_test").with_suffix(".html")
    log_module.info(f"Write fie: {fn}")
    fig.write_html(fn)

    if ac_loraks.solver_type == SolverType.AUTOGRAD:
        fig = psub.make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_titles=["loss", "learning rate"]
        )
        losses, lrs = ac_loraks.get_autograd_stats()
        cmap = plc.sample_colorscale("Sunset", losses.shape[0], 0.1, 0.9)
        for i, d in enumerate([losses, lrs]):
            for h, b in enumerate(d):
                fig.add_trace(
                    go.Scatter(
                        y=b.abs(), showlegend=True if i==0 else False, marker=dict(color=cmap[h]), name=f"Batch {h+1}",
                        legendgroup=h
                    ),
                    row=i+1, col=1
                )
        fn = path.joinpath("recon_test_autograd_stats").with_suffix(".html")
        log_module.info(f"Write fie: {fn}")
        fig.write_html(fn)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
