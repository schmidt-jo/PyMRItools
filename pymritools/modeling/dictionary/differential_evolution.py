"""
Wanting to build a CVAE architecture for the dictionary fitting of mese / megesse data.
potentially extendable to multi-compartmental of mixed gaussian modelling.

The g
ist is: we use a fairly simple encoder structure and try to run the torch emc simulation as decoder.
As first step we exclude noise processing. Could be additional processing step.
"""
import logging
import pathlib as plib

import tqdm
import numpy as np
import torch

import plotly.graph_objects as go
import plotly.subplots as psub

log_module = logging.getLogger(__name__)


class EMC:
    """
    Build a small emc simulation just pushing mono-exponential models to create simple signal patterns.

    """
    def __init__(self, etl: int = 8, esp_ms: float = 9.0, device: torch.device = torch.get_default_device()):
        # fix some echo times given esp and etl
        self.tes_s : torch.Tensor = 1e-3 * esp_ms * torch.arange(1, etl + 1).to(device)

    def forward(self, x):
        # assume input to be of dims [b, 2 (r2, s0)]
        # create fake signal curves for now just using exponential
        sig_r2 = x[:, 1, None] * torch.exp(-self.tes_s[None, :] * x[:, 0, None])
        return sig_r2


class DE:
    def __init__(self, param_dim: int, data_dim: int, population_size: int = 10,
                 p_crossover: float = 0.9, differential_weight: float = 0.8,
                 max_num_iter: int = 1000, conv_tol: float = 1e-4,
                 device: torch.device = torch.get_default_device()):
        # set parameter dimensions and bounds
        self.param_dim: int = param_dim
        self.data_dim: int = data_dim

        # algorithm vars
        self.device: torch.device = device

        self.population_size: int = population_size
        self.p_crossover: float = p_crossover
        self.differential_weight: float = differential_weight

        self.max_num_iter: int = max_num_iter
        self.conv_tol: float = conv_tol

        # functions
        self.func = NotImplemented
        self.func_forward = None

    def set_fitness_function(self, func):
        """ set function to calculate fitness of agents."""
        self.func = func

    def optimize(self):
        """ find minimum of fittness function. """
        # initialize agents -> parents und 3 random selected per parent a, b and c
        # batch everything, assume dims [b, num_params]
        agents = torch.rand(
            (self.data_dim, self.population_size, self.param_dim),
            device=self.device
        )

        # calculate the fitness / loss per agent -> agents should find optimal params
        fitness_agents = self.func(agents)

        # get best agent within population to calculate convergence later
        agents_min = torch.min(fitness_agents, dim=-1)
        last_best_agent = agents[torch.arange(self.data_dim), agents_min.indices]

        # start iteration per batch
        conv_counter = 0
        last_conv_idx = 0

        # get batch, push to device
        bar = tqdm.trange(self.max_num_iter, desc="DE optimization")
        update_bar = int(self.max_num_iter / 20)
        for idx in bar:
            a, b, c = torch.rand(
                (3, self.data_dim, self.population_size, self.param_dim),
                device=self.device
            )

            # create random numbers and indices for each dims
            r_p = torch.rand(
                (self.data_dim, self.population_size, self.param_dim),
                device=self.device
            )
            r_index = torch.randint(
                low=0, high=self.data_dim, size=(self.data_dim, self.population_size),
                device=self.device
            ).unsqueeze(-1)

            # select components to mutate
            mask_crossover = r_p < self.p_crossover
            mask_indices = torch.arange(self.param_dim, device=self.device)[None, None] == r_index
            mutation_condition = mask_crossover | mask_indices
            # calculate new candidates for the condition
            y = torch.where(
                condition=mutation_condition,
                input=a + self.differential_weight * (b - c),
                other=agents
            )

            # calculate fitness of new candidates
            fitness_y = self.func(y)

            # check for improvement and update
            better_fitness = fitness_y < fitness_agents
            # update agents
            agents = torch.where(
                condition=better_fitness.unsqueeze(-1),
                input=y,
                other=agents
            )
            # update fitness
            fitness_agents = torch.where(
                condition=better_fitness,
                input=fitness_y,
                other=fitness_agents
            )

            # get best agents within population
            agents_min = torch.min(fitness_agents, dim=-1)
            best_agent = agents[torch.arange(self.data_dim), agents_min.indices]
            # calculate convergence as max difference between best agent to last iteration.
            convergence = torch.max(torch.linalg.norm(best_agent - last_best_agent, dim=-1))
            last_best_agent = best_agent
            # ToDo: think about reducing the number of agents to process based on convergence criterion.
            # i.e. exclude converged agents from future iterations

            if convergence < self.conv_tol:
                if conv_counter > 10 and last_conv_idx == idx - 1:
                    bar.postfix = f"converged at iteration: {idx} :: conv: {convergence:.5f}"
                    break
                last_conv_idx = idx
                conv_counter += 1
            if idx % update_bar == 0:
                bar.postfix = f"convergence: {convergence:.5f}"
        return best_agent.cpu()


def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"using device : {device}")
    bounds = torch.tensor([80, 200], device=device)
    batch_size = 1000
    de = DE(
        param_dim=2,
        data_dim=batch_size,
        population_size=10,
        p_crossover=0.9,
        differential_weight=0.8,
        max_num_iter=1000,
        conv_tol=1e-6,
        device=device
    )
    emc = EMC(etl=8, device=device)
    # create some params
    num_params = 10000
    x = torch.rand((num_params, 2), device=device) * bounds[None]
    # create some curves
    curves = emc.forward(x)

    # allocate
    fit_params = torch.zeros_like(x)
    # batch processing
    num_batches = int(np.ceil(num_params / batch_size))
    for idx_b in range(num_batches):
        log_module.info(f"Process batch: {idx_b + 1} / {num_batches}")
        start = idx_b * batch_size
        end = np.min([(idx_b + 1) * batch_size, num_params])
        batch_size = end - start

        # set loss function dependent on the parameter batch
        b_curves = curves[start:end]
        def loss_func(x_dim_param):
            # assume dims [batch, population, params]
            shape = x_dim_param.shape
            x_in = torch.reshape(x_dim_param, (-1, shape[-1])) * bounds[None]
            x_dim_etl = emc.forward(x_in)
            x_dim_etl = torch.reshape(x_dim_etl, (*shape[:-1], -1))
            return torch.linalg.norm(b_curves[:, None] - x_dim_etl, dim=-1)

        de.set_fitness_function(loss_func)
        fit_params[start:end] = de.optimize()
    # put into correct scale
    fit_params = fit_params * bounds[None]
    selection = torch.randint(low=0, high=num_params, size=(20,))
    gt = x[selection].cpu()
    fit = fit_params[selection].cpu()
    log_module.info(f"gold standard params:\n {gt}")
    log_module.info(f"fit params:\n {fit}")

    fig = psub.make_subplots(rows=2, cols=1)
    fig.add_trace(
        go.Scattergl(x=x[:, 0].cpu(), y=fit_params[:, 0].cpu(), mode='markers', name='gold standard'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scattergl(x=x[:, 1].cpu(), y=fit_params[:, 1].cpu(), mode='markers', name='fit params'),
        row=2, col=1
    )
    fig_path = plib.Path(__name__).absolute().parent.joinpath("de_result").with_suffix(".html")
    fig.write_html(fig_path.as_posix())


if __name__ == '__main__':
    main()
