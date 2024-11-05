"""
Wanting to build a CVAE architecture for the dictionary fitting of mese / megesse data.
potentially extendable to multi-compartmental of mixed gaussian modelling.

The g
ist is: we use a fairly simple encoder structure and try to run the torch emc simulation as decoder.
As first step we exclude noise processing. Could be additional processing step.
"""
import logging

import torch
import torch.nn as nn


class EMC:
    """
    Build a small emc simulation just pushing mono-exponential models to create simple signal patterns.

    """
    def __init__(self):
        # init vector
        self.m_init: torch.Tensor = torch.tensor([0, 0, 0, 1])
        # excitation pulse
        num_tes = 14
        self.tes_s : torch.Tensor = 0.009 * torch.arange(1, num_tes + 1)
        # just introduce some b1 attenuation factors to fake b1 imprint
        self.b1_att: torch.Tensor = torch.ones(num_tes) + torch.exp(
            -torch.linspace(0, 1, num_tes)
        ) * torch.sin(torch.linspace(0, torch.pi * 0.8 * num_tes, num_tes))
        self.b1_att /= torch.max(self.b1_att)
        # introduce some gre echo times to fake r2p attenutation
        self.r2p_gre_times: torch.Tensor = torch.zeros(num_tes)
        self.r2p_gre_times[::2] = 0.003
        self.r2p_gre_times[2::3] = 0.006

    def forward(self, x):
        # assume input to be of dims [b, 3 (r2, r2p, b1)]
        pass


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = None

    def train(self):
        pass


def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO
    )


if __name__ == '__main__':
    main()