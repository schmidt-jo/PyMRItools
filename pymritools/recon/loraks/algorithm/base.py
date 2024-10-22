"""
To build Loraks as efficiently as possible we dont want to recompute constant vectors and matrices during run
we implement a base class which computes and stores those objects upon init

Jochen Schmidt, 12.03.2024
"""
import abc
import logging
import pathlib as plib

import torch

from pymritools.recon.loraks.algorithm import operators

log_module = logging.getLogger(__name__)


class Base:
    def __init__(self, k_space_input: torch.Tensor, sampling_mask: torch.Tensor,
                 radius: int, max_num_iter: int, conv_tol: float,
                 rank_s: int = 250, rank_c: int = 150, lambda_c: float = 0.0, lambda_s: float = 0.0,
                 channel_batch_size: int = 4,
                 device: torch.device = torch.device("cpu"), fig_path: plib.Path = None, visualize: bool = True):
        log_module.info(f"config loraks flavor")
        # config
        self.device: torch.device = device
        # self.fig_path = plib.Path(opts.output_path).absolute().joinpath("plots")
        # self.fig_path.mkdir(parents=True, exist_ok=True)

        log_module.info(f"initialize | set input matrices | set operators")
        # input
        self.sampling_mask: torch.Tensor = sampling_mask
        # save dimensions - input assumed [x, y, z, ch, t]
        self.dim_read, self.dim_phase, self.dim_slice, self.dim_channels, self.dim_echoes = k_space_input.shape
        # combined xy dim
        self.dim_s = self.dim_read * self.dim_phase
        # batch channel dimension
        self.ch_batch_size = min(channel_batch_size, self.dim_channels)
        # channel batch idxs
        self.ch_batch_idxs = torch.split(torch.randperm(self.dim_channels), self.ch_batch_size)
        self.num_batches = len(self.ch_batch_idxs)

        # combined b-ch - t dim
        self.dim_t_ch = self.dim_echoes * self.ch_batch_size
        # loraks params
        self.radius: int = radius
        self.rank_s: int = rank_s
        self.lambda_s: float = lambda_s
        self.rank_c: int = rank_c
        self.lambda_c: float = lambda_c
        self.max_num_iter: int = max_num_iter
        self.conv_tol: float = conv_tol
        # per slice, hence squeeze slice dim if its leftover
        self.k_space_dims = torch.squeeze(k_space_input).shape
        # k_space_input = torch.moveaxis(k_space_input, 2, 0)
        # want to combine squeeze slice dir
        self.k_space_input: torch.Tensor = torch.squeeze(k_space_input)
        # build fHf
        # f maps k space data of length L to us data of length S,
        # fHf is the vector with 1 at positions of undersampled points and 0 otherwise,
        # basically mapping a vector of length L to us subspace S and back to L.
        # self.fhf = torch.zeros((self.dim_s, self.dim_echoes))
        # self.fhf[self.mask_indices_input[:, 0], self.mask_indices_input[:, 1]] = 1
        # we can use this to build us data from fs or just populating the us data (fHd),
        # hence fHd basically is the us input
        # store k-space in fhd vector - this would create us data from fs input data provided a mask
        # dims fhf [x, ŷ, ch, t], k-space-in [x, y, ch, t]
        # we assume here: sampling equal in channels, and readout dir = dim 0
        self.fhd: torch.Tensor = torch.reshape(
            self.k_space_input[
                torch.tile(self.sampling_mask[:, :, None, :], (1, 1, self.dim_channels, 1))
            ],
            (self.k_space_dims[0], -1, self.dim_channels, self.dim_echoes)
        )

        # set iterate - if we would initialize k space with 0 -> loraks terms would give all 0s,
        # and first iterations would just regress to fhd, hence we can init fhd from the get go
        # self.k_iter_current: torch.tensor = self.fhd.clone().detach()
        self.iter_residuals: torch.tensor = torch.zeros((self.dim_slice, self.max_num_iter))
        # get a dict with residuals and iteration
        self.stats: dict = {}
        # get operator
        self.op_s: operators.S = operators.S(k_space_dims_x_y_ch_t=self.k_space_dims, nb_radius=self.radius)
        self.op_c: operators.C = operators.C(k_space_dims_x_y_ch_t=self.k_space_dims, nb_radius=self.radius)
        # p*p is the same matrix irrespective of channel / time sampling information,
        # we can compute it for single slice, single channel, single echo data
        self.visualize: bool = visualize
        self.fig_path: plib.Path = fig_path

        # if self.visualize:
        # log_module.debug(f"Plotting P*P")
        # if self.lambda_c > 1e-6:
        #     plotting.plot_slice(
        #         torch.reshape(self.p_star_p_c, (self.dim_read, self.dim_phase)),
        #         name="p_star_p_c", outpath=self.fig_path,
        #     )
        # if self.lambda_s > 1e-6:
        #     plotting.plot_slice(
        #         torch.reshape(self.p_star_p_s, (self.dim_read, self.dim_phase)),
        #         name="p_star_p_s", outpath=self.fig_path,
        #     )
        # log_module.debug(f"Plotting fhf and fhd")
        # for idx_e in range(min(self.dim_echoes, 3)):
        #     plotting.plot_slice(
        #         torch.reshape(self.fhf[:, idx_e], (self.dim_read, self.dim_phase)),
        #         f"fhf_e-{idx_e+1}", outpath=self.fig_path
        #     )
        #     plotting.plot_slice(
        #         torch.reshape(self.fhd[0, :, 0, idx_e], (self.dim_read, self.dim_phase)),
        #         f"fhd_sli-0_ch-0_e-{idx_e+1}", outpath=self.fig_path
        #     )

    def reconstruct(self):
        log_module.info(f"start processing")
        self._recon()

    def get_k_space(self):
        # move slice dim back, dims [z, xy, ch, t]
        k_space_recon = torch.moveaxis(self.k_iter_current, 0, 1)
        # reshape to original input dims [xy, z, ch, t]
        k_space_recon = torch.reshape(k_space_recon, self.k_space_dims)

        return k_space_recon

    def get_residuals(self) -> (torch.tensor, dict):
        return self.iter_residuals, self.stats

    @abc.abstractmethod
    def _recon(self):
        """ to be set in the individual flavors, aka abstract method """
        return NotImplementedError
