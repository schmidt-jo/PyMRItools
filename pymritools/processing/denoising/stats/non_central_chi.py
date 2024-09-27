import torch
import logging

log_module = logging.getLogger(__name__)


def from_noise_voxels(noise_voxel_data: torch.tensor) -> (float, int):
    sigma = get_sigma_from_noise_vox(noise_voxel_data)
    n = int(torch.round(get_n_from_noise_vox(noise_voxel_data, sigma)))
    return sigma, n


def get_n_from_noise_vox(noise_voxel_data: torch.tensor, sigma: float):
    return 1 / (2 * noise_voxel_data.shape[0] * sigma ** 2) * torch.sum(noise_voxel_data ** 2, dim=0)


def get_sigma_from_noise_vox(noise_voxel_data: torch.tensor) -> float:
    num_pts = noise_voxel_data.shape[0]
    a = torch.sqrt(torch.tensor([1 / 2]))
    b = torch.sum(noise_voxel_data ** 4, dim=0) / torch.sum(noise_voxel_data ** 2, dim=0)
    c = 1 / num_pts * torch.sum(noise_voxel_data ** 2, dim=0)
    d = torch.sqrt(b - c)
    return (a * d).item()


def noise_dist_jean(x: torch.Tensor, sigma: torch.Tensor | float, n: torch.Tensor | int):
    sigma = torch.as_tensor(sigma).to(torch.float64)
    t = x ** 2 / (2 * sigma ** 2)
    n = torch.round(torch.as_tensor(n)).to(torch.int)
    return 1 / torch.exp(torch.lgamma(n)) * torch.pow(t, n - 1) * torch.exp(-t)


def noise_dist_ncc(x: torch.tensor, sigma: torch.Tensor | float, n: torch.Tensor | int):
    sigma = torch.as_tensor(sigma).to(torch.float64)
    n = torch.round(torch.as_tensor(n)).to(torch.int)
    a = torch.pow(x, 2 * n - 1)
    b = torch.pow(2, n - 1) * torch.pow(sigma, 2 * n) * torch.exp(torch.lgamma(n))
    c = torch.exp((-x ** 2) / (2 * sigma ** 2))
    return a / b * c