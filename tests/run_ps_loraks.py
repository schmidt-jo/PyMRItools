import torch

from pymritools.recon.loraks_dev.ps_loraks import Loraks, LowRankAlgorithmType
from pymritools.utils import Phantom, fft


def run_ps_loraks():
    # set log level to info
    import logging
    logging.basicConfig(level=logging.INFO)
    loraks = (Loraks(max_num_iter=200)
              .with_device(torch.device("cpu"))
              .with_s_matrix()
              .with_torch_lowrank_algorithm(q=30, niter=2)
              .with_sv_auto_soft_cutoff()
              # .with_sv_hard_cutoff(45, 15)
              .with_patch_shape((5, 5))
              .with_sample_directions((1, 1))
              .with_linear_learning_rate(0.0001, 0.002)
              )

    acceleration = 3
    k_shape = (512, 512)
    jupiter = Phantom.get_jupiter(k_shape)
    jupiter_k_space = jupiter.sub_sample_random(acceleration)
    jupiter_img = jupiter.get_2d_image()

    jupiter_k_recon = loraks.reconstruct(jupiter_k_space[None], (torch.abs(jupiter_k_space) > 1e-7)[None])
    jupiter_img_recon = fft(jupiter_k_recon[0], axes=(0, 1))


if __name__ == '__main__':
    run_ps_loraks()
