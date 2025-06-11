import sys
import pathlib as plib
path = plib.Path(__name__).absolute().parent
sys.path.append(path.as_posix())

import json
import logging

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.colors as plc

from tests.utils import get_test_result_output_dir, ResultMode
from pymritools.config.emc import EmcSimSettings, EmcParameters
from pymritools.simulation.emc.sequence.mese import MESE
from pymritools.utils.algorithms import DE

logging.getLogger('pymritools.simulation.emc.core').setLevel(logging.WARNING)
logging.getLogger('pymritools.simulation.emc.sequence.mese').setLevel(logging.WARNING)
logging.getLogger('pymritools.simulation.emc.sequence.base_sequence').setLevel(logging.WARNING)


def main():
    # hardcode some of the path and parameters
    path = plib.Path(__name__).absolute().parent
    path_out = plib.Path(get_test_result_output_dir("vfa_mese_de", mode=ResultMode.OPTIMIZATION)).joinpath("optim_run")
    path_out.mkdir(exist_ok=True, parents=True)
    # setup logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO, filename=path_out.joinpath("log.txt").as_posix(),
        filemode="w"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    settings_path = path.joinpath("optimization/optim_emc_settings.json")
    settings = EmcSimSettings.load(settings_path.as_posix())

    # we need to fix some path business in the example file
    for key, val in settings.__dict__.items():
        if key.endswith("file"):
            p = path.joinpath(val)
            settings.__setattr__(key, p.as_posix())

    settings.display()
    # testing: set set to simulate low
    # settings.t2_list = [35, 45, 55, 65]
    # settings.b1_list = [0.5, 1.0]

    params = EmcParameters.load(settings.emc_params_file)

    # setup fas
    bounds = torch.tensor([85, 140], device=device)
    lam_snr = 0.98
    de = DE(
        param_dim=8,
        data_dim=1,
        population_size=12,
        p_crossover=0.9,
        differential_weight=0.8,
        max_num_iter=100,
        conv_tol=1e-6,
        device=device
    )

    def loss_function(x):
        shape = x.shape
        # flatten x
        x = x.view(-1, shape[-1])
        # allocate
        losses = torch.zeros(x.shape[0])
        # get fas within bounds
        fas = bounds[0] + torch.diff(bounds) * x
        # for each fa compute
        for idx_fa, fa in enumerate(fas):
            logging.info(f"computing for population: {idx_fa+1} / {de.population_size}")
            params.refocus_angle = fa.tolist()

            # build sim object
            mese = MESE(params=params, settings=settings)
            mese.simulate()

            # compute losses
            sar = torch.sqrt(torch.sum((torch.tensor(fas) / 180 * torch.pi) ** 2))
            snr = torch.linalg.norm(mese.data.signal_mag, dim=-1).flatten().mean()
            # minimize sar, maximize snr, with a minimizing total loss

            losses[idx_fa] = (1.0 - lam_snr) * sar - lam_snr * snr
        losses = torch.reshape(losses, shape[:-1])
        return losses

    de.set_fitness_function(loss_function)

    optim_fa, progress = de.optimize()
    optim_fa = bounds[0] + torch.diff(bounds) * optim_fa

    fig = go.Figure()
    colors = plc.sample_colorscale("viridis", np.linspace(0, 1, len(progress)))
    for idx_f, f in enumerate(progress):
        f = bounds[0] + torch.diff(bounds) * f
        # result has batch dim, just plot first one
        fig.add_trace(
            go.Scattergl(y=f[0].cpu().numpy(), line=dict(width=1), marker=dict(color=colors[idx_f]))
        )
    fig_path = path_out.joinpath("de_progress").with_suffix(".html")
    logging.info(f"Save file: {fig_path}")
    fig.write_html(fig_path.as_posix())

    file_path = path_out.joinpath("de_optim_fa").with_suffix(".json")
    logging.info(f"optimized fa: {optim_fa}")
    logging.info(f"Save file: {file_path}")
    with open(file_path.as_posix(), "w") as f:
        json.dump(optim_fa.cpu().tolist(), f)


if __name__ == '__main__':
    main()

#
# def main():
#     logging.basicConfig(
#         format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
#         datefmt='%I:%M:%S', level=logging.INFO
#     )
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f"using device : {device}")
#     bounds = torch.tensor([80, 200], device=device)
#     batch_size = 1000
#     de = DE(
#         param_dim=6,
#         data_dim=1,
#         population_size=10,
#         p_crossover=0.9,
#         differential_weight=0.8,
#         max_num_iter=10,
#         conv_tol=1e-6,
#         device=device
#     )
#     def loss_func(fa_s):
#
#         # assume dims [batch, population, params]
#         shape = x_dim_param.shape
#         x_in = torch.reshape(x_dim_param, (-1, shape[-1])) * bounds[None]
#         x_dim_etl = emc.forward(x_in)
#         x_dim_etl = torch.reshape(x_dim_etl, (*shape[:-1], -1))
#         return torch.linalg.norm(b_curves[:, None] - x_dim_etl, dim=-1)
#
#
#
#     emc = EMC(etl=8, device=device)
#     # create some params
#     num_params = 10000
#     x = torch.rand((num_params, 2), device=device) * bounds[None]
#     # create some curves
#     curves = emc.forward(x)
#
#     # allocate
#     fit_params = torch.zeros_like(x)
#     # batch processing
#     num_batches = int(np.ceil(num_params / batch_size))
#     for idx_b in range(num_batches):
#         log_module.info(f"Process batch: {idx_b + 1} / {num_batches}")
#         start = idx_b * batch_size
#         end = np.min([(idx_b + 1) * batch_size, num_params])
#         batch_size = end - start
#
#         # set loss function dependent on the parameter batch
#         b_curves = curves[start:end]
#         def loss_func(x_dim_param):
#             # assume dims [batch, population, params]
#             shape = x_dim_param.shape
#             x_in = torch.reshape(x_dim_param, (-1, shape[-1])) * bounds[None]
#             x_dim_etl = emc.forward(x_in)
#             x_dim_etl = torch.reshape(x_dim_etl, (*shape[:-1], -1))
#             return torch.linalg.norm(b_curves[:, None] - x_dim_etl, dim=-1)
#
#         de.set_fitness_function(loss_func)
#         fit_params[start:end] = de.optimize()
#     # put into correct scale
#     fit_params = fit_params * bounds[None]
#     selection = torch.randint(low=0, high=num_params, size=(20,))
#     gt = x[selection].cpu()
#     fit = fit_params[selection].cpu()
#     log_module.info(f"gold standard params:\n {gt}")
#     log_module.info(f"fit params:\n {fit}")
#
#     fig = psub.make_subplots(rows=2, cols=1)
#     fig.add_trace(
#         go.Scattergl(x=x[:, 0].cpu(), y=fit_params[:, 0].cpu(), mode='markers', name='gold standard'),
#         row=1, col=1
#     )
#     fig.add_trace(
#         go.Scattergl(x=x[:, 1].cpu(), y=fit_params[:, 1].cpu(), mode='markers', name='fit params'),
#         row=2, col=1
#     )
#     fig_path = plib.Path(__name__).absolute().parent.joinpath("de_result").with_suffix(".html")
#     fig.write_html(fig_path.as_posix())
#
