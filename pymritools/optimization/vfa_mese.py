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
    path = plib.Path("./optimization/").absolute()
    path.mkdir(exist_ok=True, parents=True)
    # set some params
    sim_settings = EmcSimSettings(
        out_path=path.joinpath("optim_sim").as_posix(),
        # kernel_file=path.joinpath("mese_v1p0_acc-3p0_res-0p70-0p70-0p70_kernels").with_suffix(".pkl").as_posix(),
        # te_file=path.joinpath("mese_v1p0_acc-3p0_res-0p70-0p70-0p70_te").with_suffix(".json").as_posix(),
        # pulse_file="/data/pt_np-jschmidt/data/03_sequence_dev/build_sequences/gauss.json",
        t2_list=[
            [1, 50, 10], [50, 200, 25], [200, 500, 100]
        ],
        b1_list=[[0.3, 1.7, 0.1]],
        gpu_device=wandb.config.gpud,
        visualize=False
    )
    # sim_settings.display()

    params = EmcParameters(
        etl=7, esp=7.37, bw=350, gradient_excitation=-32.21,
        duration_excitation=2000, gradient_excitation_rephase=-23.324,
        duration_excitation_rephase=380,
        gradient_refocus=-17.179, duration_refocus=2500,
        gradient_crush=-42.274, duration_crush=1000,
    )

    # here we need to plug in the actual FAs
    # fas = [
    #     wandb.config.fa_1, wandb.config.fa_2, wandb.config.fa_3,
    #     wandb.config.fa_4, wandb.config.fa_5, wandb.config.fa_6,
    #     wandb.config.fa_7
    # ]
    fas = torch.randint(low=60, high=140, size=(7,))
    params.refocus_angle = fas

    # build sim object
    mese = MESE(params=params, settings=sim_settings)
    mese.simulate()

    # compute losses
    sar = torch.sqrt(torch.sum((torch.tensor(fas) /180*torch.pi)**2))
    snr = torch.linalg.norm(mese.data.signal_mag, dim=-1).flatten().mean()
    # minimize sar, maximize snr, with a minimizing total loss
    loss = sar - snr
    wandb.log({"loss": loss, "sar": sar, "snr": snr, "fas": fas})


if __name__ == '__main__':
    main()
