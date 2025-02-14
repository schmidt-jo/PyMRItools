import torch

from pymritools.recon.loraks_dev.ps_loraks import Loraks, LowRankAlgorithmType
from pymritools.utils import Phantom, fft

def test_run_ps_loraks():
    loraks = (Loraks()
              .with_device(torch.device("cpu"))
              .with_s_matrix()
              .with_torch_lowrank_algorithm(q=20, niter=2)
              .with_sv_hard_cutoff(20, 15)
              .with_patch_shape((5, 5))
              .with_sample_directions((1, 1))
              .with_constant_learning_rate(0.001)
              )

    acceleration = 3
    k_shape = (512, 512)
    jupiter = Phantom.get_jupiter(k_shape)
    jupiter_k_space = jupiter.sub_sample_random(acceleration)
    jupiter_img = jupiter.get_2d_image()

    jupiter_k_recon = loraks.reconstruct(jupiter_k_space[None], (torch.abs(jupiter_k_space) > 1e-7)[None])
    jupiter_img_recon = fft(jupiter_k_recon[0], axes=(0, 1))
