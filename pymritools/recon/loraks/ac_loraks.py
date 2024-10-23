import logging
import pathlib as plib
import tqdm

import torch

from pymritools.recon.loraks.algorithm import Base
from pymritools.utils.algorithms import cgd

log_module = logging.getLogger(__name__)


class ACLoraks(Base):
    def __init__(
            self, k_space_input: torch.Tensor, sampling_mask: torch.Tensor,
            max_num_iter: int, conv_tol: float,
            radius: int = 3, rank_c: int = 150, lambda_c: float = 0.0, rank_s: int = 250, lambda_s: float = 0.0,
            fft_algorithm: bool = False, channel_batch_size: int = 4,
            device: torch.device = torch.device("cpu"), fig_path: plib.Path = None, visualize: bool = True):
        log_module.debug(f"Setup Base class")
        # initialize base constants
        super().__init__(
            k_space_input=k_space_input, sampling_mask=sampling_mask,
            radius=radius, rank_c=rank_c, lambda_c=lambda_c,
            rank_s=rank_s, lambda_s=lambda_s,
            max_num_iter=max_num_iter, conv_tol=conv_tol,
            channel_batch_size=channel_batch_size,
            device=device, fig_path=fig_path, visualize=visualize
        )
        log_module.debug(f"Setup AC LORAKS specifics")
        self.fft_algorithm: bool = fft_algorithm
        # compute aha - stretch along batched channels dim
        # fhf = self.fhf[:, None, :].expand((-1, self.ch_batch_size, -1))
        self.aha = (
                self.fhf + self.lambda_c * torch.reshape(self.op_c.p_star_p, self.k_space_dims) +
                self.lambda_s * torch.reshape(self.op_s.p_star_p, self.k_space_dims)
        ).to(self.device)
        # recon takes place per slice for combined echo and channel dimensions.
        # we can thus find the AC region and save it to not redo the computation each slice
        # we save the indices of mapped vectors to op matrix after finding which neighborhoods are
        # completely mapped into the operator matrix
        self.idxs_ac_s: torch.Tensor = self._find_ac_indices(mode="s")
        self.idxs_ac_c: torch.Tensor = self._find_ac_indices(mode="c")
        log_module.info(f"neighborhood/concat channel-time sizes: "
                        f"S - {self.dim_t_ch * self.op_s.neighborhood_size} ... rank S - {self.rank_s}; "
                        f"C - {self.dim_t_ch * self.op_c.neighborhood_size} ... rank C - {self.rank_c}")

    def _find_ac_indices(self, mode: str = "s"):
        # we can use the sampling mask, rebuild it in 2d, sampling pattern equal per coil but different per echo
        # momentarily the implementation is aiming at joint echo recon. we use all echoes,
        # but might need to batch the channel dimensions. in the nullspace approx we use all data
        # to check included neighborhoods we dont need to check the channel data, just pick first
        fhf = self.fhf[:, :, 0]
        if mode in ["s", "S"]:
            # get indices for centered origin k-space
            p_nb_nx_ny_idxs = self.op_s.neighborhood_indices
            # we want to find indices for which the whole neighborhood of point symmetric coordinates is contained
            a = fhf[p_nb_nx_ny_idxs[:, :, 0], p_nb_nx_ny_idxs[:, :, 1]]
            m_nb_nx_ny_idxs = self.op_s.neighborhood_indices_pt_sym
            b = fhf[m_nb_nx_ny_idxs[:, :, 0], m_nb_nx_ny_idxs[:, :, 1]]
            # lets look for the ACS which is constant across the joint dimension and the neighborhood
            # we check the mask for it, irrespective of the combined dimension, its usually sampled different
            # for different echoes, but equally for the different channels. We can deduce the dimensions from the mask.
            c = torch.sum(a, dim=(1, 2)) + torch.sum(b, dim=(1, 2))
            idxs_ac = torch.tile((c == a.shape[1] * a.shape[2] * 2), dims=(2,))
        elif mode in ["c", "C"]:
            # get indices for centered origin k-space
            p_nb_nx_ny_idxs = self.op_c.neighborhood_indices
            # we want to find indices for which the whole neighborhood of point symmetric coordinates is contained
            a = fhf[p_nb_nx_ny_idxs[:, :, 0], p_nb_nx_ny_idxs[:, :, 1]]
            c = torch.sum(a, dim=(1, 2))
            idxs_ac = (c == a.shape[1] * a.shape[2])
        else:
            err = f"mode {mode} not implemented for AC region search."
            log_module.error(err)
            raise ValueError(err)
        # plot extracted region if flag set
        # if self.visualize:
        #     # plot the found acs for reference
        #     acs = torch.zeros((idxs_ac.shape[0], self.dim_nb * self.dim_t_ch * df))
        #     acs[idxs_ac] = 1
        #     acs = op_x.operator_adjoint(acs)
        #     acs = torch.reshape(acs, (self.dim_read, self.dim_phase, -1))
        #     plotting.plot_slice(acs[:, :, 0], f"extracted_AC_region_{mode}", self.fig_path)
        # return the found indices
        return idxs_ac

    def get_acs_v(self, idx_batch: int, idx_slice: int, mode: str):
        """
                Compute the V matrix for solving the AC LORAKS formulation from autocalibration data.
                We need to find the ACS data first and then evaluate the nullspace subspace.
                """
        # we use data from current slice
        k_space_slice = self.fhd
        if mode in ["c", "C"]:
            # want sub-matrix of respective loraks matrix with fully populated rows
            # build X matrix from k-space current slice
            c_matrix = self.op_c.operator(k_space_x_y_ch_t=k_space_slice)
            # take only fully populated rows
            m_ac = c_matrix[self.idxs_ac_c]
            # we want to use the rank of the c mode
            rank = self.rank_c
        elif mode in ["s", "S"]:
            # now we build the s_matrix with the 0 filled data for the current slice
            s_matrix = self.op_s.operator(k_space_x_y_ch_t=k_space_slice)
            # and keep only the fully populated rows - we calculated which those are already for all
            # echo sampling patterns
            m_ac = s_matrix[self.idxs_ac_s]
            # we want to use the rank parameter of the s mode
            rank = self.rank_s
        else:
            err = f"ACS calibration not implemented (yet) for mode {mode}. Choose either c or s"
            log_module.error(err)
            raise ValueError(err)
        if m_ac.shape[0] < rank:
            err = f"detected AC region is too small for chosen rank"
            log_module.error(err)
            raise ValueError(err)
        # put on device if not there already
        m_ac = m_ac.to(self.device)
        # via eigh
        eig_vals, eig_vecs = torch.linalg.eigh(torch.matmul(m_ac.T, m_ac))
        m_ac_rank = eig_vals.shape[0]
        # get subspaces from svd of subspace matrix
        eig_vals, idxs = torch.sort(torch.abs(eig_vals), descending=True)
        # eig_vecs_r = eig_vecs[idxs]
        eig_vecs = eig_vecs[:, idxs]
        # v_sub_r = eig_vecs_r[:self.rank].to(self.device)
        v_sub = eig_vecs[:, :rank].to(self.device)

        if m_ac_rank < rank:
            err = f"loraks rank parameter is too large, cant be bigger than ac matrix dimensions."
            log_module.error(err)
            raise ValueError(err)

        # if idx_slice == 0 and idx_batch ==0 and self.visualize:
        #     plotting.plot_slice(v_sub, name=f"v_sub_{mode}", outpath=self.fig_path)

        return v_sub

    def _get_m_1_diag_vector(
            self, f: torch.tensor,
            v_s: torch.tensor = torch.zeros((1, 1)), v_c: torch.tensor = torch.zeros((1, 1))) -> torch.tensor:
        """
        We define the M_1 matrix from A^H A f and the loraks operator P_x(f) V V^H,
        after extracting V from ACS data and getting the P_x based on the Loraks mode used.
        """
        # if input is zero vector -> S and C matrices will be 0 matrices. hence all multiplications will return 0
        # can skip compute aka matrix generation
        if torch.max(torch.abs(f)) < 1e-7:
            return torch.zeros_like(f)
        m1_fhf = self.aha * f
        if self.lambda_c > 1e-6:
            m1_v_c = self.op_c.operator_adjoint(
                torch.matmul(
                    self.op_c.operator(f),
                    v_c
                )
            )
        else:
            m1_v_c = 0.0
        if self.lambda_s > 1e-6:
            m1_v_s = self.op_s.operator_adjoint(
                torch.matmul(
                    self.op_s.operator(f),
                    v_s
                )
            )
        else:
            m1_v_s = 0.0
        return m1_fhf - self.lambda_s * m1_v_s - self.lambda_c * m1_v_c

    def _solve_fft(self):
        """
        observing L_i(f) = P_x(f)v_i for a column v_i of V matrix. Because of convolutional structure of the
        Loraks Matrices, this operator can be computed via FFTs:
        L_i(f) = T F^-1 ( F(Z_2 v_i) * F(Z_1 f) )
        Where F are FFT operations, Z are zero padding ops, its a element wise multiplication and
        T extracts relevant samples of respective neighborhood.

        It follows we can construct M
        M = A^H A f + lam Z_1^H F^-1 (N * F(Z_1 f) )
        """
        pass

    def _cgd(self, b: torch.tensor, v_s: torch.tensor = torch.zeros((1, 1)), v_c: torch.tensor = torch.zeros((1, 1)),
             iter_bar: tqdm.tqdm = None):

        def func_op(x):
            return self._get_m_1_diag_vector(f=x, v_s=v_s, v_c=v_c)

        xmin, res_vec, results = cgd(
            func_operator=func_op,
            x=torch.zeros_like(b), b=b,
            iter_bar=iter_bar, max_num_iter=self.max_num_iter,
            conv_tol=self.conv_tol
        )

        # iter_dict = collections.OrderedDict([("iter", "_".rjust(self.max_num_iter, "_"))])
        # iter_bar.set_postfix(ordered_dict=iter_dict)
        # n2b = torch.linalg.norm(b)
        #
        # x = torch.zeros_like(b)
        # p = 1
        # xmin = x
        # iimin = 0
        # tolb = self.conv_tol * n2b
        #
        # r = b - self._get_m_1_diag_vector(f=x, v_s=v_s, v_c=v_c)
        #
        # normr = torch.linalg.norm(r)
        # normr_act = normr
        #
        # if normr < tolb:
        #     log_module.info("convergence before loop")
        #
        # res_vec = torch.zeros(self.max_num_iter)
        # normrmin = normr
        #
        # rho = 1
        #
        # for ii in range(self.max_num_iter):
        #     z = r
        #     rho1 = rho
        #     rho = torch.abs(torch.sum(r.conj() * r))
        #     if ii == 0:
        #         p = z
        #     else:
        #         beta = rho / rho1
        #         p = z + beta * p
        #     q = self._get_m_1_diag_vector(f=p, v_s=v_s, v_c=v_c)
        #     pq = torch.sum(p.conj() * q)
        #     alpha = rho / pq
        #
        #     x = x + alpha * p
        #     r = r - alpha * q
        #
        #     normr = torch.linalg.norm(r)
        #     normr_act = normr
        #     res_vec[ii] = normr
        #
        #     if normr <= tolb:
        #         r = b - self._get_m_1_diag_vector(f=x, v_s=v_s, v_c=v_c)
        #         normr_act = torch.linalg.norm(r)
        #         res_vec[ii] = normr_act
        #         log_module.info(f"reached convergence at step {ii + 1}")
        #         break
        #
        #     if normr_act < normrmin:
        #         normrmin = normr_act
        #         xmin = x
        #         iimin = ii
        #         log_module.debug(f"min residual {normrmin:.2f}, at {iimin + 1}")
        #     iter_dict["iter"] = f"{iter_dict['iter'][1:]}I"
        #     iter_bar.set_postfix(iter_dict)
        return xmin, res_vec, results

    def _recon(self):
        """
        AC LORAKS specific reconstruction method.
        """
        # loraks_recon = torch.zeros(self.k_space_dims, dtype=torch.complex128)
        # compute combined echo and channel information slice wise
        # essentially here we are minimizing data consistency for the sampled points against
        # low rank constraints of the LORAKS matrices
        for idx_slice in range(self.dim_slice):
            log_module.info(f"Reconstruction::Processing::Slice {1 + idx_slice} / {self.dim_slice}")
            iter_bar = tqdm.trange(self.num_batches, desc="Slice::Processing::Batch")
            for idx_batch in iter_bar:
                log_module.debug("estimate nullspace - C")
                if self.lambda_c > 1e-6:
                    v_sub_c = self.get_acs_v(idx_batch=idx_batch, idx_slice=idx_slice, mode="c")
                else:
                    v_sub_c = torch.zeros((1, 1))
                if self.lambda_s > 1e-6:
                    v_sub_s = self.get_acs_v(idx_batch=idx_batch, idx_slice=idx_slice, mode="s")
                else:
                    v_sub_s = torch.zeros((1, 1))

                vvs = torch.matmul(v_sub_s, v_sub_s.conj().T)
                vvc = torch.matmul(v_sub_c, v_sub_c.conj().T)

                log_module.debug("start pcg solve")

                # dim fhd [z, xy, ch, t]
                in_fhd = self.fhd.to(self.device)

                f_slice, residual_abs_sum, stats = self._cgd(b=in_fhd, v_s=vvs, v_c=vvc, iter_bar=iter_bar)
                self.k_iter_current[idx_slice, :, self.ch_batch_idxs[idx_batch], :] = torch.reshape(
                    f_slice,
                    (-1, self.ch_batch_size, self.dim_echoes)
                ).cpu()
                self.iter_residuals[idx_slice, :len(residual_abs_sum)] = residual_abs_sum.clone().cpu()
                self.stats = stats
            # ToDo implement different algorithmic choices here:
            #  multiplicative Majorize-Minimize Approach - FFT version, start with fft

