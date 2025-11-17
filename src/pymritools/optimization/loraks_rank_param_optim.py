import sys
import pathlib as plib
import torch
import wandb

from pymritools.recon.loraks.ac_loraks import AcLoraksOptions
from pymritools.recon.loraks.loraks import RankReduction, RankReductionMethod, Loraks
from pymritools.recon.loraks.operators import OperatorType
from pymritools.utils import fft_to_img, Phantom, calc_nmse, calc_psnr, root_sum_of_squares, \
    calc_ssim

p_tests = plib.Path(__file__).absolute().parent.parent.parent
sys.path.append(p_tests.as_posix())
from tests.experiments.loraks.utils import prep_k_space, DataType, load_data, unprep_k_space


def wandb_rank_param_optim(debug: bool = False):
    # need to call this to synchronise to wandb
    run = wandb.init() if not debug else None
    # choose GPU if available on the compute
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # load data, you need to input your specific datapath here
    # (since its synchronised by wandb, you could hardcode this)
    k, _, bet = load_data(data_type=DataType.INVIVO)
    bet = bet[:, :, 0].to(torch.int)

    # i used a fully sampled dataset as ground truth and subsampled, hence the following should be exchanged by your specific data.
    # i.e., load your ground truth and load your undersampled data separately
    shape = k.shape
    gt = fft_to_img(k, dims=(0, 1))
    gt = root_sum_of_squares(gt, dim_channel=-2)
    gt *= bet

    # set number of echoes to be used
    # i wanted to show that using more echoes results in better reconstructions when jointly reconstructing.
    # hence you dont need this and could just use all echoes
    ne = wandb.config.ne if not debug else 1
    k = k[..., :ne].clone()
    bet = bet[..., :ne].clone()
    # my specific undersampling mechanism
    phantom = Phantom.get_shepp_logan(shape=shape[:2], num_coils=shape[-2], num_echoes=shape[-1])

    sub = wandb.config.sub if not debug else "random_lines"
    match sub:
        case "random":
            k_us = phantom.sub_sample_random(acceleration=4, ac_central_radius=20)
        case "random_lines":
            k_us = phantom.sub_sample_ac_random_lines(acceleration=4, ac_lines=36)
        case "grappa":
            k_us = phantom.sub_sample_ac_grappa(acceleration=4, ac_lines=36)
        case _:
            k_us = phantom.sub_sample_ac_skip_lines(acceleration=4, ac_lines=36)

    # squeezing slice dim if present, but also echo if only 1
    k_in = k.clone().squeeze()
    if ne == 1:
        k_in.unsqueeze_(-1)
        k_us.unsqueeze_(-1)
    mask = k_us.abs() > 1e-10
    k_in[~mask] = 0
    # here k_in is my undersampled data. at this point you should have your data ready

    # get number of channel batches - i also varied the number of batch channels, you can fix this
    nc = wandb.config.nc if not debug else 8
    data_in, in_shape, padding, batch_channel_idx = prep_k_space(k_in.unsqueeze(2), batch_size_channels=nc)

    # loraks
    # set rank parameter to be optimised - this is the jist.
    # you want this parameter to be pulled from your wandb sweep config
    rank = wandb.config.rank if not debug else 200
    # you might just set lambda = 0.0, again, i wanted to show that using higher values gives worse recon
    lam = wandb.config.lam

    if nc > 30 and ne > 4:
        wandb.log({
            "loss": 1, "nmse": 1, "psnr": 0, "ssim": 0, "rank": rank, "lambda": lam,
            "nc": nc, "ne": ne, "mce": 25 * nc * ne
        })
        return 0

    # this is as you might know it. just setting the options
    ac_opts = AcLoraksOptions(
        loraks_neighborhood_size=5, loraks_matrix_type=OperatorType.S,
        rank=RankReduction(method=RankReductionMethod.HARD_CUTOFF, value=rank), regularization_lambda=lam,
        max_num_iter=30, device=device
    )
    # creating the algorithm instance
    loraks = Loraks.create(ac_opts)

    # reconstruction
    k_recon = loraks.reconstruct(data_in)

    # unprepping the k-space
    k_recon = unprep_k_space(k_recon, padding=padding, batch_idx=batch_channel_idx, input_shape=in_shape)
    # squeezing slice dim if present, but also echo if 1 - this is a bit specific to my input,
    # you might need to take a look at your data dims, i just wanted to make sure my channel combination
    # is not accidentially done wrong
    img_recon = fft_to_img(k_recon, dims=(0, 1)).squeeze()
    if ne == 1:
        img_recon.unsqueeze_(-1)

    # i also loaded a bet mask extracting only the brain,
    # thus i calculated SSIM, NMSE and PSNR only within the brain and not outside.
    rsos = root_sum_of_squares(img_recon, dim_channel=-2) * bet

    # calculate metrics
    nmse = calc_nmse(gt, rsos)
    psnr = calc_psnr(gt, rsos)
    ssim = calc_ssim(gt.permute(2, 1, 0), rsos.permute(2, 1, 0))

    # set loss - i optimised a combined loss metric. you could just use SSIM it might is enough for your use case.
    # i just wanted to make sure i optimise with respect to all 3 with some scaling to put them in the same order.
    loss = nmse - 1e-2 * psnr - ssim
    if not debug:
        wandb.log({
            "loss": loss, "nmse": nmse, "psnr": psnr, "ssim": ssim, "rank": rank, "lambda": lam,
            "nc": nc, "ne": ne, "mce": 25 * nc * ne
        })
    return 0

if __name__ == '__main__':
    wandb_rank_param_optim(debug=False)
