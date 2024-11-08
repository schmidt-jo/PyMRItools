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
    def __init__(self, param_dim: int, population_size: int = 10,
                 p_crossover: float = 0.9, differential_weight: float = 0.8,
                 device: torch.device = torch.get_default_device()):
        self.dim: int = param_dim
        self.device: torch.device = device

    def optimize(self, y):
        pass


def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"using device : {device}")

    etl = 8
    # check curve creation
    r2_bounds = 50.0
    s0_bounds = 100.0
    emc = EMC(etl=etl, esp_ms=9.0, device=device)

    # create a bunch of curves
    num_curves = 1

    r2_s0_b = torch.tensor([r2_bounds, s0_bounds]).to(device)
    x = torch.rand((num_curves, 2), device=device) * r2_s0_b
    curves = emc.forward(x)

    # de algorithm
    population_size = 10
    p_crossover = 0.9
    differential_weight = 0.8

    # initialize agents - parents and 3 random picks per parent a, b and c
    agents = torch.rand((population_size, 2), device=device) * r2_s0_b[None]
    shape = agents.shape
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

    print(f"best candidate: {best_agent.tolist()}, vs x: {torch.squeeze(x).tolist()}")

    # fig_path = plib.Path(__name__).parent.absolute()
    # file_name = fig_path.joinpath("test_predict").with_suffix(".html")
    # print(f"write file: {file_name}")
    # fig.write_html(file_name.as_posix())

if __name__ == '__main__':
    main()