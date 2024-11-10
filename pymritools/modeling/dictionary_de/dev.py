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
import torch.nn as nn
from scipy.fft import ifftshift
from torch.optim import Adam, SGD
from torch.distributions import Normal

import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc


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
    def __init__(self, param_dim: int, param_bounds: torch.Tensor, population_size: int = 10,
                 p_crossover: float = 0.9, differential_weight: float = 0.8,
                 device: torch.device = torch.get_default_device()):
        # set parameter dimensions and bounds
        self.dim: int = param_dim
        self.bounds: torch.Tensor = param_bounds

        self.device: torch.device = device
        # set vars
        self.population_size: int = population_size
        self.p_crossover: float = p_crossover
        self.differential_weight: float = differential_weight
        # initialize agents
        self.agents = torch.rand((self.population_size, self.dim), device=device) * self.bounds[None]
        self.func = NotImplemented

    def set_fitness_function(self, func):
        self.func = func

    def optimize(self, y):
        # initialize agents - parents and 3 random picks per parent a, b and c
        shape = self.agents.shape
        curves_agents = emc.forward(torch.reshape(agents, (-1, shape[-1])))
        curves_agents = torch.reshape(curves_agents, (*shape[:-1], -1))
        fitness_agents = torch.linalg.norm(curves_agents - curves, dim=-1)

        for idx in tqdm.trange(1000):
            y = torch.zeros_like(agents)
            a = torch.rand((population_size, 2), device=device) * r2_s0_b
            b = torch.rand((population_size, 2), device=device) * r2_s0_b
            c = torch.rand((population_size, 2), device=device) * r2_s0_b
            # for each dim pick randomly uniform distributed number and index
            r_p = torch.rand((population_size, 2), device=device)
            r_index = torch.randint(low=0, high=2, size=(population_size,), device=device)
            for idx_p in range(population_size):
                # check per dim
                for idx_n in range(2):
                    if r_index[idx_p] == idx_n or r_p[idx_p, idx_n] < p_crossover:
                        y[idx_p, idx_n] = a[idx_p, idx_n] + differential_weight * (b[idx_p, idx_n] - c[idx_p, idx_n])
                    else:
                        y[idx_p, idx_n] = agents[idx_p, idx_n]
            curves_y = emc.forward(y)
            fitness_y = torch.linalg.norm(curves_y - curves, dim=-1)

            for idx_p in range(population_size):
                if fitness_y[idx_p] < fitness_agents[idx_p]:
                    agents[idx_p] = y[idx_p]
                    fitness_agents[idx_p] = fitness_y[idx_p]
            # print(f"loss agents: {torch.mean(fitness_agents).item():.3f}")
            idx_min = torch.min(fitness_agents, dim=-1).indices
            best_agent = agents[idx_min]
            loss = torch.mean(torch.abs(best_agent - x))
            if loss.item() < 0.3:
                print(f"loss to start param: {loss.item():.5f} found candidate")
                break


def ai_vectorized_suggestion():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    etl = 8
    emc = EMC(etl=etl, device=device)
    # Parameter des Algorithmus
    population_size = 10
    p_crossover = 0.9
    differential_weight = 0.8
    num_dimensions = 2

    bounds_r2_s0 = torch.tensor([80, 200], device=device)
    batch_dim = 1000
    x = torch.rand((batch_dim, num_dimensions), device=device) * bounds_r2_s0[None]
    curves = emc.forward(x)
    # for now we have a batch dimension and for x the num parameters and for curves the number of echoes
    # dims: x [b, 2], curves [b, etl]

    # initialize agents -> parents und 3 random selected per parent a, b and c
    # batch everything
    agents = torch.rand((batch_dim, population_size, num_dimensions), device=device) * bounds_r2_s0[None, None]
    shape_agents = agents.shape
    # calculate the fitness / loss per agent -> agents should find optimal params,
    # hence we need to calculate the curves for forward model, need to put in correct shape and back
    curves_agents = torch.reshape(
        emc.forward(torch.reshape(agents, (-1, num_dimensions))),
        (batch_dim, population_size, etl)
    )
    fitness_agents = torch.linalg.norm(curves_agents - curves[:, None, :], dim=-1)

    num_iter = 3000
    fig = psub.make_subplots(
        rows=6, cols=2, shared_xaxes=True, shared_yaxes=True,
    )
    plot_count = 0
    for idx in tqdm.trange(num_iter):
        a, b, c = torch.rand((3, batch_dim, population_size, num_dimensions), device=device) * bounds_r2_s0[None, None]

        # Zufällig verteilte Zahlen und Indizes pro Dimension und Agent
        r_p = torch.rand((batch_dim, population_size, num_dimensions), device=device)
        r_index = torch.randint(low=0, high=num_dimensions, size=(batch_dim, population_size,), device=device).unsqueeze(-1)

        # Vektorisierte Auswahl der zu mutierenden Komponenten
        mask_crossover = r_p < p_crossover
        mask_indices = torch.arange(num_dimensions, device=device)[None, None] == r_index
        mutation_condition = mask_crossover | mask_indices
        y = torch.where(mutation_condition, a + differential_weight * (b - c), agents)

        curves_y = torch.reshape(
            emc.forward(torch.reshape(y, (-1, num_dimensions))),
            (batch_dim, population_size, etl)
        )
        fitness_y = torch.linalg.norm(curves_y - curves[:, None, :], dim=-1)

        # Aktualisiere Agenten und Fitnesswerte bei Verbesserung
        better_fitness = fitness_y < fitness_agents
        agents = torch.where(better_fitness.unsqueeze(-1), y, agents)
        fitness_agents = torch.where(better_fitness, fitness_y, fitness_agents)

        # Überprüfen des besten Agenten und gegebenenfalls Abbruch
        agents_min = torch.min(fitness_agents, dim=-1)
        best_agent = agents[torch.arange(batch_dim), agents_min.indices]
        loss = torch.max(torch.mean(torch.abs(best_agent - x), dim=-1))

        if loss.item() < 0.3:
            print(f"Loss to start param: {loss.item():.5f} found candidate")
            break

        if idx in [0, 100, 200, 500, 1000, 2000]:
            plot_count +=1
            for idx_c in range(2):
                fig.add_trace(
                    go.Scatter(
                        x=x[:, idx_c].cpu().numpy(), y=best_agent[:, idx_c].cpu().numpy(),
                        mode="markers"
                    ),
                    row=plot_count, col=1+idx_c
                )
    print(f"Best candidate: \n{best_agent.tolist()}")
    print(f"vs x: \n{torch.squeeze(x).tolist()}")
    fig_name = plib.Path(__name__).absolute().parent.joinpath("de_iterations").with_suffix(".html")
    fig.write_html(fig_name.as_posix())


def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"using device : {device}")
    ai_vectorized_suggestion()

    # etl = 8
    # # check curve creation
    # r2_bounds = 50.0
    # s0_bounds = 100.0
    # emc = EMC(etl=etl, esp_ms=9.0, device=device)
    #
    # # create a bunch of curves
    # num_curves = 1
    #
    # r2_s0_b = torch.tensor([r2_bounds, s0_bounds]).to(device)
    # x = torch.rand((num_curves, 2), device=device) * r2_s0_b
    # curves = emc.forward(x)
    #
    # # de algorithm
    # population_size = 10
    # p_crossover = 0.9
    # differential_weight = 0.8
    #
    # # initialize agents - parents and 3 random picks per parent a, b and c
    # agents = torch.rand((population_size, 2), device=device) * r2_s0_b[None]
    # shape = agents.shape
    # curves_agents = emc.forward(torch.reshape(agents, (-1, shape[-1])))
    # curves_agents = torch.reshape(curves_agents, (*shape[:-1], -1))
    # fitness_agents = torch.linalg.norm(curves_agents - curves, dim=-1)
    #
    # for idx in tqdm.trange(1000):
    #     y = torch.zeros_like(agents)
    #     a = torch.rand((population_size, 2), device=device) * r2_s0_b
    #     b = torch.rand((population_size, 2), device=device) * r2_s0_b
    #     c = torch.rand((population_size, 2), device=device) * r2_s0_b
    #     # for each dim pick randomly uniform distributed number and index
    #     r_p = torch.rand((population_size, 2), device=device)
    #     r_index = torch.randint(low=0, high=2, size=(population_size,), device=device)
    #     for idx_p in range(population_size):
    #         # check per dim
    #         for idx_n in range(2):
    #             if r_index[idx_p] == idx_n or r_p[idx_p, idx_n] < p_crossover:
    #                 y[idx_p, idx_n] = a[idx_p, idx_n] + differential_weight * (b[idx_p, idx_n] - c[idx_p, idx_n])
    #             else:
    #                 y[idx_p, idx_n] = agents[idx_p, idx_n]
    #     curves_y = emc.forward(y)
    #     fitness_y = torch.linalg.norm(curves_y - curves, dim=-1)
    #
    #     for idx_p in range(population_size):
    #         if fitness_y[idx_p] < fitness_agents[idx_p]:
    #             agents[idx_p] = y[idx_p]
    #             fitness_agents[idx_p] = fitness_y[idx_p]
    #     # print(f"loss agents: {torch.mean(fitness_agents).item():.3f}")
    #     idx_min = torch.min(fitness_agents, dim=-1).indices
    #     best_agent = agents[idx_min]
    #     loss = torch.mean(torch.abs(best_agent - x))
    #     if loss.item() < 0.3:
    #         print(f"loss to start param: {loss.item():.5f} found candidate")
    #         break
    #
    # print(f"best candidate: {best_agent.tolist()}, vs x: {torch.squeeze(x).tolist()}")
    #
    # # fig_path = plib.Path(__name__).parent.absolute()
    # # file_name = fig_path.joinpath("test_predict").with_suffix(".html")
    # # print(f"write file: {file_name}")
    # # fig.write_html(file_name.as_posix())

if __name__ == '__main__':
    main()