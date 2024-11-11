import json
import logging
import pathlib as plib
import torch

from pymritools.config.emc import EmcSimSettings, EmcParameters
from pymritools.simulation.emc.sequence.mese import MESE
from pymritools.utils.algorithms import DE

logging.getLogger('pymritools.simulation.emc.core').setLevel(logging.WARNING)
logging.getLogger('pymritools.simulation.emc.sequence.mese').setLevel(logging.WARNING)
logging.getLogger('pymritools.simulation.emc.sequence.base_sequence').setLevel(logging.WARNING)

def main():
    # setup logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO
    )
    # hardcode some of the path and parameters
    path = plib.Path("./optimization/").absolute()
    path.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set some R2s
    r2 = 3 * torch.exp(0.15 * torch.arange(1, 31))
    t2 = 1e3 / r2

    # set some params
    sim_settings = EmcSimSettings(
        out_path=path.joinpath("optim_sim").as_posix(),
        t2_list=t2.tolist(),
        b1_list=[[0.3, 1.7, 0.05]],
        visualize=False
    )
    # sim_settings.display()

    params = EmcParameters(
        etl=6, esp=7.37, bw=350, gradient_excitation=-32.21,
        duration_excitation=2000, gradient_excitation_rephase=-23.324,
        duration_excitation_rephase=380,
        gradient_refocus=-17.179, duration_refocus=2500,
        gradient_crush=-42.274, duration_crush=1000,
        sample_number=1000
    )

    # setup fas
    bounds = torch.tensor([60, 140], device=device)
    de = DE(
        param_dim=6,
        data_dim=1,
        population_size=2,
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
            mese = MESE(params=params, settings=sim_settings)
            mese.simulate()

            # compute losses
            sar = torch.sqrt(torch.sum((fa / 180 * torch.pi)**2))
            snr = torch.linalg.norm(mese.data.signal_mag, dim=-1).flatten().mean()
            # minimize sar, maximize snr, with a minimizing total loss
            losses[idx_fa] = sar - 5 * snr
        losses = torch.reshape(losses, shape[:-1])
        return losses

    de.set_fitness_function(loss_function)

    optim_fa = de.optimize()
    optim_fa = bounds[0] + torch.diff(bounds) * optim_fa

    logging.info(f"optimized fa: {optim_fa}")
    path = plib.Path(__name__).absolute().parent.joinpath("de_optim_fa").with_suffix(".json")
    logging.info(f"Save file: {path}")
    with open(path, "w") as f:
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
