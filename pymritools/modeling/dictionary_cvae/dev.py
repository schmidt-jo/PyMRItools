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
from torch.optim import Adam
from torch.distributions import Normal

import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc


class EMC:
    """
    Build a small emc simulation just pushing mono-exponential models to create simple signal patterns.

    """
    def __init__(self, etl: int = 8):
        # init vector
        self.m_init: torch.Tensor = torch.tensor([0, 0, 0, 1])
        # excitation pulse
        self.tes_s : torch.Tensor = 0.009 * torch.arange(1, etl + 1)
        # just introduce some b1 attenuation factors to fake b1 imprint
        self.b1_att: torch.Tensor = torch.ones(etl) + torch.exp(
            -torch.linspace(0, 1, etl)
        ) * torch.sin(torch.linspace(0, torch.pi * 0.8 * etl, etl))
        self.b1_att /= torch.max(self.b1_att)
        # introduce some gre echo times to fake r2p attenutation
        self.r2p_gre_times: torch.Tensor = torch.zeros(etl)
        self.r2p_gre_times[::2] = 0.003
        self.r2p_gre_times[2::3] = 0.006


    def forward(self, x):
        # assume input to be of dims [b, 3 (r2, r2p, b1)]
        # create fake signal curves for now
        sig_r2 = torch.exp(-self.tes_s[None, :] * x[:, 0, None])
        att_r2p = torch.exp(-self.tes_s[None, :] * x[:, 1, None])
        att_b1 = self.b1_att[None, :] * x[:, 2, None]
        sig = sig_r2 * att_r2p * att_b1
        return sig


class Encoder(nn.Module):
    def __init__(self, etl: int, num_params: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=etl*8, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=2*num_params),
            nn.ReLU()
        )
        self.mu = nn.Linear(in_features=2*num_params, out_features=num_params)
        self.log_var = nn.Linear(in_features=2*num_params, out_features=num_params)
        self.bounds_r2_r2p_b1: torch.Tensor = torch.tensor([50, 30, 1.6])

    def forward(self, x):
        # insert channels
        x = x.unsqueeze(1)
        inp_x = self.encoder(x)
        mu = nn.Sigmoid()(self.mu(inp_x))   # mean bigger 0 up to 1
        log_var = self.log_var(inp_x)
        # we want to hit the upper bounds
        mu = mu * self.bounds_r2_r2p_b1[None, :]
        sigma = (torch.exp(0.5 * log_var) + 1e-9) * self.bounds_r2_r2p_b1[None, :]
        return mu, sigma


class CVAE(nn.Module):
    def __init__(self, etl: int = 8):
        super().__init__()
        self.encoder = Encoder(etl=etl)
        self.decoder = EMC(etl=etl)

    def forward(self, x):
        params_mu, params_sigma = self.encoder.forward(x)
        # sample from normal
        # params_draw = Normal(loc=params_mu, scale=params_sigma).rsample()
        curves = self.decoder.forward(params_mu)
        return curves


def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO
    )
    etl = 8
    # check curve creation
    emc = EMC(etl=etl)
    # x = torch.zeros((5, 3))
    # x[:, 0] = torch.linspace(20,50, 5)
    # x[:, 1] = torch.linspace(10, 30, 5)
    # x[:, 2] = torch.linspace(0.4,1.5,5)
    #
    # sig = emc.forward(x)
    #
    # fig = psub.make_subplots(rows=5, cols=1, shared_xaxes=True)
    # for i in range(5):
    #     fig.add_trace(
    #         go.Scattergl(y=sig[i]),
    #         row=i+1, col=1
    #     )
    # fig_path = plib.Path(__name__).parent.absolute()
    # fig_path.mkdir(exist_ok=True, parents=True)
    # file_name = fig_path.joinpath("test_generation").with_suffix(".html")
    # print(f"write file: {file_name}")
    # fig.write_html(file_name.as_posix())
    #
    # decoder = Decoder(etl=etl)
    # params = decoder.forward(sig)
    # print(params)
    logging.info("train")

    cvae = CVAE(etl=etl)
    optim = Adam(params=cvae.parameters(), lr=1e-2)
    batch_size = 50000
    max_num_iter = 500
    losses = []
    bar = tqdm.trange(max_num_iter)
    for i in bar:
        # create a bunch of curves
        x = torch.zeros((batch_size, 3))
        x[:, 0] = torch.randint(low=20, high=50, size=(batch_size,))
        x[:, 1] = torch.randint(low=10, high=30, size=(batch_size,))
        x[:, 2] = torch.randint(low=30, high=150, size=(batch_size,)) / 100
        curves = emc.forward(x)
        curves /= torch.linalg.norm(curves, dim=-1, keepdim=True)

        predicted_curves = cvae.forward(curves)
        normed_curves = predicted_curves / torch.linalg.norm(predicted_curves, dim=-1, keepdim=True)
        loss = nn.MSELoss()(curves, normed_curves)
        loss.backward()

        optim.step()
        optim.zero_grad()

        if i % 20 == 0:
            bar.postfix(f"loss: {loss.item():.5f}  [{i:d}/{max_num_iter}]")
        losses.append(loss.item())
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(y=losses)
    )
    fig_path = plib.Path(__name__).parent.absolute()
    file_name = fig_path.joinpath("test_losses").with_suffix(".html")
    print(f"write file: {file_name}")
    fig.write_html(file_name.as_posix())

    logging.info("predict")
    # create some parameter sets
    num_sets = 5
    x = torch.zeros((num_sets, 3))
    x[:, 0] = torch.randint(low=20, high=50, size=(num_sets,))
    x[:, 1] = torch.randint(low=10, high=30, size=(num_sets,))
    x[:, 2] = torch.randint(low=30, high=150, size=(num_sets,)) / 100
    # create curves
    curves = emc.forward(x).detach()
    # normalize
    curves /= torch.linalg.norm(curves, dim=-1, keepdim=True)

    # the prediction is only done by encoder of cvae
    mu, sigma = cvae.encoder.forward(curves)
    mu = mu.detach()
    sigma = sigma.detach()

    # plot
    names = ["r2", "r2p", "b1"]
    fig = psub.make_subplots(
        rows=num_sets, cols=3,
        shared_xaxes=True, shared_yaxes=True,
        column_titles=names
    )
    colors = plc.sample_colorscale('viridis', np.linspace(0, 1, 3))

    for idx_s in range(num_sets):
        for idx_p in range(3):
            fig.add_trace(
                go.Scattergl(
                    y=[0, 1], x=[x[idx_s, idx_p], x[idx_s, idx_p]], name=names[idx_p], showlegend=False,
                    line=dict(color=colors[0])
                ),
                row=idx_s+1, col=1 + idx_p
            )
            fig.add_trace(
                go.Scattergl(
                    y=[0, 1], x=[mu[idx_s, idx_p], mu[idx_s, idx_p]], name=names[idx_p], showlegend=False,
                    line=dict(color=colors[1])
                ),
                row=idx_s+1, col=1 + idx_p
            )
            # dist = Normal(loc=mu[idx_s, idx_p], scale=sigma[idx_s, idx_p]).log_prob(torch.linspace(0, 50, 200))
            # dist = torch.exp(dist)
            # dist /= torch.max(dist)
            #
            # fig.add_trace(
            #     go.Scattergl(y=dist, name=f"{names[idx_p]} estimation", line=dict(color=colors[1]), showlegend=False),
            #     row=idx_s+1, col=1+idx_p
            # )
    fig_path = plib.Path(__name__).parent.absolute()
    file_name = fig_path.joinpath("test_predict").with_suffix(".html")
    print(f"write file: {file_name}")
    fig.write_html(file_name.as_posix())

if __name__ == '__main__':
    main()