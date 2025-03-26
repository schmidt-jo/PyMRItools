"""
This script is intended to optimize the mese (or megesse) pulse trains to produce dictionary curves with
minimal SAR while offering maximal SNR.
"""
import logging
import pathlib as plib
import sys

path_to_pymritools = plib.Path(__name__).parent.parent.absolute()
sys.path.append(path_to_pymritools.as_posix())

import torch
import wandb

from pymritools.config.emc import EmcSimSettings, EmcParameters
from pymritools.simulation.emc.sequence.mese import MESE


def main():
    run = wandb.init()

    # setup logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO
    )
    # hardcode some of the path and parameters
    path_out = plib.Path(
        "/data/pt_np-jschmidt/data/03_sequence_dev/mese_pulse_train_optimization/optimization"
    ).absolute()
    path_out.mkdir(exist_ok=True, parents=True)

    path_in = plib.Path(
        "/data/pt_np-jschmidt/data/03_sequence_dev/build_sequences/2025-03-20_mese_phantom/emc_settings.json"
    ).absolute()
    # set some params
    sim_settings = EmcSimSettings.load(path_in)
    sim_settings.visualize = False
    # sim_settings.display()

    params = EmcParameters.load(sim_settings.emc_params_file)

    # here we need to plug in the actual FAs
    fas = [
        wandb.config.fa_1, wandb.config.fa_2, wandb.config.fa_3,
        wandb.config.fa_4, wandb.config.fa_5, wandb.config.fa_6,
        wandb.config.fa_7, wandb.config.fa_8
    ]
    # fas = torch.randint(low=50, high=140, size=(6,))
    params.refocus_angle = fas

    # build sim object
    mese = MESE(params=params, settings=sim_settings)
    mese.simulate()

    # compute losses
    sar = torch.sqrt(torch.sum((torch.tensor(fas) / 180 * torch.pi)**2))
    snr = torch.linalg.norm(mese.data.signal_mag, dim=-1).flatten().mean()
    # minimize sar, maximize snr, with a minimizing total loss
    loss = (1.0 - wandb.config.lam_snr) * sar - wandb.config.lam_snr * snr
    wandb.log({"loss": loss, "sar": sar, "snr": snr, "fas": fas})


if __name__ == '__main__':
    main()
