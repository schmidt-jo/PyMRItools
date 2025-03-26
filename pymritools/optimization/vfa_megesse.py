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
from pymritools.simulation.emc.sequence.megesse_asym import MEGESSE


def main():
    run = wandb.init()

    # setup logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO
    )
    # hardcode some of the path and parameters
    path_out = plib.Path(
        "/data/pt_np-jschmidt/data/03_sequence_dev/megesse_pulseq_train_optimization/"
    ).absolute()
    path_out.mkdir(exist_ok=True, parents=True)

    path_in = plib.Path(
        "/data/pt_np-jschmidt/data/03_sequence_dev/build_sequences/2025-03-03_megesse_phantom_vfa/emc/emc_settings.json"
    ).absolute()
    # set some params
    sim_settings = EmcSimSettings.load(path_in)
    sim_settings.visualize = False
    device = torch.device("cuda" if torch.cuda.is_available() and sim_settings.use_gpu else "cpu")
    # sim_settings.display()

    params = EmcParameters.load(sim_settings.emc_params_file)

    # here we need to plug in the actual FAs
    fas = [
        wandb.config.fa_1, wandb.config.fa_2, wandb.config.fa_3,
        wandb.config.fa_4
    ]
    # fas = torch.randint(low=50, high=140, size=(6,))
    params.refocus_angle = fas

    # build sim object
    megesse = MEGESSE(params=params, settings=sim_settings)
    megesse.simulate()
    db_cplx = megesse.data.signal_mag * torch.exp(1j * megesse.data.signal_phase)

    # compute losses
    sar = torch.sqrt(torch.sum((torch.tensor(fas) / 180 * torch.pi)**2))
    snr = torch.linalg.norm(db_cplx, dim=-1).flatten().mean()
    corr = torch.abs(
        torch.mean(
            torch.tril(
                torch.corrcoef(
                    torch.reshape(db_cplx, (-1, db_cplx.shape[-1])).to(device)
                ),
                diagonal=-1
            )
        )
    ).cpu()

    # minimize sar, maximize snr, with a minimizing total loss
    loss = (1.0 - wandb.config.lam_snr) * sar - wandb.config.lam_snr * snr + corr
    wandb.log({"loss": loss, "sar": sar, "snr": snr, "corr": corr, "fas": fas})


if __name__ == '__main__':
    main()
