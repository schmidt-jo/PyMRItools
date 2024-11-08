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
    def __init__(self, etl: int = 8, device: torch.device = torch.get_default_device()):
        # init vector
        self.m_init: torch.Tensor = torch.tensor([0, 0, 0, 1])
        # excitation pulse
        self.tes_s : torch.Tensor = 0.009 * torch.arange(1, etl + 1).to(device)
        # just introduce some b1 attenuation factors to fake b1 imprint
        # self.b1_att: torch.Tensor = torch.ones(etl) + torch.exp(
        #     -torch.linspace(0, 1, etl)
        # ) * torch.sin(torch.linspace(0, torch.pi * 0.8 * etl, etl))
        # self.b1_att /= torch.max(self.b1_att)
        # introduce some gre echo times to fake r2p attenutation
        # self.r2p_gre_times: torch.Tensor = torch.zeros(etl)
        # self.r2p_gre_times[::2] = 0.003
        # self.r2p_gre_times[2::3] = 0.006


    def forward(self, x):
        # assume input to be of dims [b, 3 (r2, s0)]
        # create fake signal curves for now
        sig_r2 = x[:, 1, None] * torch.exp(-self.tes_s[None, :] * x[:, 0, None])
        # att_r2p = torch.exp(-self.tes_s[None, :] * x[:, 1, None])
        # att_b1 = self.b1_att[None, :] * x[:, 2, None]
        # sig = sig_r2 * att_r2p * att_b1
        return sig_r2


class Encoder(nn.Module):
    def __init__(self, etl: int, output_dim: int = 3, device: torch.device = torch.get_default_device()):
        super().__init__()
        self.conv_1 = nn.Conv1d(1, 16, kernel_size=3, padding=1, device=device)
        self.conv_2 = nn.Conv1d(16, 16, kernel_size=3, padding=1, device=device)
        self.leaky = nn.LeakyReLU(0.1)
        self.flat = nn.Flatten()
        self.linear = nn.Linear(in_features=etl * 16, out_features=2*output_dim, device=device)
        self.mu = nn.Linear(in_features=2*output_dim, out_features=output_dim).to(device)
        self.log_var = nn.Linear(in_features=2*output_dim, out_features=output_dim).to(device)

    def forward(self, x):
        # insert channels
        x = x.unsqueeze(1)
        x = self.conv_1(x)
        x = self.leaky(x)
        x = self.conv_2(x)
        x = self.leaky(x)
        x = self.flat(x)
        x = self.linear(x)
        mu =self.mu(x)   # mean bigger 0 up to 1
        log_var = self.log_var(x)
        return mu, log_var


class CVAE(nn.Module):
    def __init__(
            self, etl: int = 8, num_params: int = 3, device: torch.device = torch.get_default_device()):
        super().__init__()
        self.encoder = Encoder(etl=etl, output_dim=num_params, device=device)
        self.decoder = EMC(etl=etl, device=device)

    @staticmethod
    def reparametrize(mu, log_var):
        std = torch.exp(0.5 * log_var) + 1e-9
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        params_mu, params_log_var = self.encoder.forward(x)
        # sample from normal
        params_draw = self.reparametrize(params_mu, params_log_var)
        # params_draw = Normal(loc=params_mu, scale=params_sigma).rsample()
        curves = self.decoder.forward(params_mu)
        return curves, params_mu, params_log_var

    def predict(self, x):
        params_mu, params_log_var = self.encoder.forward(x)
        # params_scaled = params_mu * self.bounds_r2_r2p_b1[None, :]
        # params_sig = torch.exp(0.5 * params_log_var) * self.bounds_r2_r2p_b1[None, :]
        return params_mu,  params_log_var

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
    emc = EMC(etl=etl, device=device)

    # create a bunch of curves
    batch_size = 100000

    r2_s0_b = torch.tensor([r2_bounds, s0_bounds]).to(device)[None, :]
    x = torch.rand((batch_size, 2), device=device) * r2_s0_b
    curves = emc.forward(x)

    # decoder = Decoder(etl=etl)
    # params = decoder.forward(sig)
    # print(params)

    logging.info("train cvae")

    cvae = CVAE(etl=etl, num_params=2, device=device)
    # enc = Encoder(etl=8, num_params=2, device=device)
    max_num_iter = 5000
    losses = []
    bar = tqdm.trange(max_num_iter)
    #
    # logging.info(cvae)
    # optim = Adam(params=cvae.parameters(), lr=1e-3, weight_decay=1e-5)
    #
    # for i in bar:
    #     predicted_curves, predicted_params, predicted_params_log_var = cvae.forward(curves)
    #
    #     loss_curves = torch.linalg.norm(curves - predicted_curves, dim=-1)
    #     loss_curves = torch.mean(loss_curves)
    #
    #     # loss_params = nn.MSELoss()(predicted_params, x)
    #     # loss_params = torch.clamp(
    #     #     torch.tensor([i], device=device) / 1000, 0, 1
    #     loss_dist = nn.GaussianNLLLoss(eps=1e-9)(predicted_params, x, torch.exp(predicted_params_log_var))
    #     # loss_kl = - 0.5 * torch.mean(1 + enc_log_var - torch.square(enc_mu) - torch.exp(enc_log_var))
    #     loss = loss_curves + loss_dist
    #     loss.backward()
    #
    #     optim.step()
    #     optim.zero_grad()
    #     # f = torch.linspace(20,0.05, max_num_iter)
    #     # with torch.no_grad():
    #     #     r2_s0.sub_(100*r2_s0.grad)
    #
    #     # r2_s0.grad.zero_()
    #
    #     if i > 50 and i % 20 == 0:
    #         bar.postfix = (
    #             f"loss: {loss.item():.5f} :: loss curves: {loss_curves.item():.5f} :: "
    #             f"loss params: {loss_dist.item():.5f} :: [{i:d}/{max_num_iter}]"
    #         )
    #     losses.append(loss.detach().item())
    #     if i > 500 and loss.item() > 1e5 or torch.isnan(loss):
    #         logging.error("training fault (nans or exploding loss).")
    #         raise ValueError("training fault (nans or exploding loss).")
    # fig = go.Figure()
    # fig.add_trace(
    #     go.Scattergl(y=losses)
    # )
    # fig_path = plib.Path(__name__).parent.absolute()
    # file_name = fig_path.joinpath("test_losses").with_suffix(".html")
    # print(f"write file: {file_name}")
    # fig.write_html(file_name.as_posix())
    #
    # logging.info("predict")
    # # create some parameter sets
    num_sets = 5
    x = torch.zeros((num_sets, 2), device=device)
    x[:, 0] = torch.rand(size=(num_sets,), device=device) * r2_bounds
    x[:, 1] = torch.rand(size=(num_sets,), device=device) * s0_bounds
    # # create curves
    curves = emc.forward(x)

    # the prediction is only done by encoder of cvae
    # params, log_var = cvae.predict(curves)

    logging.info("train emc")
    params_bp = torch.full_like(x, 0.5, device=device) * r2_s0_b
    params_bp.requires_grad_(True)
    num_iter = 1000
    factor = np.linspace(0.5, 0.01, num_iter)
    bar = tqdm.trange(num_iter)
    for i in bar:
        sig = emc.forward(params_bp)
        loss = torch.linalg.norm(sig - curves, dim=-1)
        loss = torch.sum(loss)

        loss.backward()

        with torch.no_grad():
            params_bp.sub_(params_bp.grad * 0.1)

        params_bp.grad.zero_()

        logging.info(f"loss: {loss.item():.5f} :: [{i:d}/{num_iter}]")

    rmse = torch.sqrt(torch.mean((params_bp - x) ** 2))
    logging.info(f"rmse: {rmse.item():.5f}")

    # plot
    names = ["r2", "s0", "b1"]
    fig = psub.make_subplots(
        rows=num_sets, cols=2,
        shared_xaxes=True, shared_yaxes=True,
        column_titles=names
    )
    colors = plc.sample_colorscale('viridis', np.linspace(0, 1, 3))

    x = x.detach().cpu()
    # mu = mu.detach().cpu()
    # sigma = torch.exp(log_var).detach().cpu()
    # r2_s0 = params.detach().cpu()
    for idx_s in range(num_sets):
        for idx_p in range(2):
            val = x[idx_s, idx_p].item()
            fig.add_trace(
                go.Bar(
                    y=[0, 1, 0], x=[val-2, val, val+2], name='set', showlegend=False,
                    marker=dict(color=colors[0]), width=1
                ),
                row=idx_s+1, col=1 + idx_p
            )
            val = params_bp[idx_s, idx_p].item()
            fig.add_trace(
                go.Bar(
                    y=[0, 1, 0], x=[val-2, val, val+2], name='bp', showlegend=False,
                    marker=dict(color=colors[2]), width=1
                ),
                row=idx_s+1, col=1 + idx_p
            )
            # val = r2_s0[idx_s, idx_p].item()
            # fig.add_trace(
            #     go.Bar(
            #         y=[0, 1, 0], x=[val-2, val, val+2], name="predict", showlegend=False,
            #         marker=dict(color=colors[1]), width=1
            #     ),
            #     row=idx_s+1, col=1 + idx_p
            # )
            # ax = torch.linspace(0, 100, 200)
            # dist = Normal(loc=val, scale=sigma[idx_s, idx_p]).log_prob(ax)
            # dist = torch.exp(dist)
            # dist /= torch.max(dist)
            #
            # fig.add_trace(
            #     go.Scattergl(x=ax, y=dist, name=f"{names[idx_p]} estimation", line=dict(color=colors[1]), showlegend=False),
            #     row=idx_s+1, col=1+idx_p
            # )
    fig_path = plib.Path(__name__).parent.absolute()
    file_name = fig_path.joinpath("test_predict").with_suffix(".html")
    print(f"write file: {file_name}")
    fig.write_html(file_name.as_posix())

if __name__ == '__main__':
    main()