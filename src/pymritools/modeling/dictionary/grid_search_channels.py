import logging
import pathlib as plib
from enum import auto, Enum

import tqdm
import numpy as np
import torch
import json

from scipy.ndimage import gaussian_filter

from pymritools.recon.loraks.matrix_indexing import get_linear_indices

from pymritools.config import setup_program_logging, setup_parser
from pymritools.config.emc import EmcFitSettings
from pymritools.config.database import DB
from pymritools.utils import nifti_save, fft_to_img, ifft_to_k, root_sum_of_squares, torch_load, nifti_load

logger = logging.getLogger(__name__)

class InputType(Enum):
    B1 = auto()
    B0 = auto()
    B1ERR = auto()


def smooth_map(data: torch.Tensor, kernel_size: int = 5):
    # set kernel
    kernel = torch.zeros(data.shape[:2], dtype=data.dtype, device=data.device)
    kernel[
        (data.shape[0] - kernel_size) // 2:(data.shape[0] + kernel_size) // 2,
        (data.shape[1] - kernel_size) // 2:(data.shape[1] + kernel_size) // 2
    ] = 1
    while kernel.shape.__len__() < data.shape.__len__():
        kernel = kernel.unsqueeze(-1)
    # fft
    data_fft = fft_to_img(data, dims=(0, 1))
    # convolve
    data_fft *= kernel
    # fft back
    result = ifft_to_k(data_fft, dims=(0, 1))
    if not torch.is_complex(data):
        result = torch.abs(result)
        result = torch.clamp(result, min=data.min(), max=data.max())
    return result


def fit_megesse():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"set device: {device}")
    logger.info(f"Load data")
    data = torch_load(
        "/data/pt_np-jschmidt/data/00_phantom_scan_data/2025-03-06_megesse_tests/raw/megesse_cesp_axial/gnc/"
        "img_gnc_cplx_slice.pt"
    )
    # data = torch.squeeze(data)
    affine = torch_load(
        "/data/pt_np-jschmidt/data/00_phantom_scan_data/2025-03-06_megesse_tests/raw/megesse_cesp_axial/affine.pt"
    )
    logger.info(f"load database")
    db = DB.load(
        "/data/pt_np-jschmidt/data/03_sequence_dev/build_sequences/2025-03-03_megesse_phantom_vfa/emc/"
        "test_batch/db_megesse_cesp.pkl"
    )
    path = plib.Path(
        "/data/pt_np-jschmidt/data/00_phantom_scan_data/2025-03-06_megesse_tests/processing/megesse/ch_dot"
    )

    with open(
            "/data/pt_np-jschmidt/data/03_sequence_dev/build_sequences/2025-03-03_megesse_phantom_vfa/"
            "megesse_fa135/megesse_v1p0_acc-1p0_res-0p70-0p70-0p70_te.json",
            "r"
    ) as j_file:
        te = json.load(j_file)

    logger.info("Get Params")
    te = torch.tensor(te)
    gre_attenuation_times = torch.zeros_like(te)
    gre_attenuation_times[:5] = torch.abs(te[:5] - te[0])
    gre_attenuation_times[5:10] = torch.abs(te[5:10] - te[9])
    gre_attenuation_times[10:15] = torch.abs(te[10:15] - te[10])
    gre_attenuation_times[15:] = torch.abs(te[15:] - te[19])

    loss_residual_lower_limit = 0.5
    loss_residual_upper_limit = 1.0

    # get torch tensors
    db_torch_mag, db_torch_phase = db.get_torch_tensors_t1t2b1b0e()
    db_pattern = db_torch_mag * torch.exp(1j * db_torch_phase)
    db_pattern = db_pattern.to(data.dtype)

    db_shape = db_torch_mag.shape
    # get t2 and b1 values
    t1_vals, t2_vals, b1_vals, b0_vals = db.get_t1_t2_b1_b0_values()

    logger.info("normalize data")
    # data_rsos = root_sum_of_squares(input_data=data, dim_channel=-2)
    data_rsos = data

    logger.info("prep database")

    t1t2b1b0_vals = torch.tensor([
        [t1, t2, b1, b0] for t1 in t1_vals for t2 in t2_vals for b1 in b1_vals for b0 in b0_vals
    ], device=device)
    r2p_vals = torch.linspace(0.1, 20, 60)

    t1t2b0r2p_vals = torch.tensor([
        [t1, t2, b0, r2p] for t1 in t1_vals for t2 in t2_vals for b0 in b0_vals for r2p in r2p_vals
    ], device=device)

    nb_indices = torch.arange(-5, 6)
    t1t2r2p_vals = torch.tensor([
        [t1, t2, r2p] for t1 in t1_vals for t2 in t2_vals for r2p in r2p_vals
    ], device=device)

    # estimate b1 with SE only data
    se_ind = torch.squeeze(torch.argwhere(gre_attenuation_times < 1e-6))
    db_se = db_pattern[..., se_ind]
    data_se = data_rsos[..., se_ind]

    # normalize
    db_se = torch.nan_to_num(db_se / torch.linalg.norm(db_se, dim=-1, keepdim=True)).to(device)
    # data_normed_se = torch.abs(torch.nan_to_num(data_se / torch.linalg.norm(data_se, dim=-1, keepdim=True))[:, :, None])
    data_normed_se = torch.nan_to_num(data_se / torch.linalg.norm(data_se, dim=-1, keepdim=True))[:, :, None]
    # prep for b1 estimation
    db_se = torch.reshape(db_se, (-1, db_se.shape[-1]))

    logger.info("Allocate data")

    t2 = torch.zeros(data_normed_se.shape[:-1])
    r2p = torch.zeros(data_normed_se.shape[:-1])

    b1 = torch.zeros(data_normed_se.shape[:-1])
    b0 = torch.zeros(data_normed_se.shape[:-1])

    loss_residual = torch.zeros(data_normed_se.shape[:-1])
    phase_offset = torch.zeros(data_normed_se.shape[:-1])

    logger.info(f"rough estimate R2*")
    r2s = torch.squeeze(torch.zeros((*data_normed_se.shape[:-1], 2)))
    residual = torch.squeeze(torch.zeros((*data_normed_se.shape[:-1], 2)))
    weights = torch.tensor((3/4, 1/4))

    wavg = torch.linspace(1, 0.5, 5)
    wavg /= torch.sum(wavg)
    for i in range(2):
        gre_indices = torch.arange(5) + 10 * i
        data_gre = torch.abs(data_rsos[..., gre_indices])
        te_gre = gre_attenuation_times[gre_indices]
        # we assume that the data after the initial spin echo decays with some r2s
        log_sig = torch.zeros_like(data_gre)
        log_sig[data_gre > 0] = torch.log(data_gre[data_gre > 0])
        A = torch.ones((*data_gre.shape, 2))
        A[..., 0] = te_gre

        ATA = torch.linalg.matmul(A.mT, A)
        ATA_inv = torch.linalg.inv(ATA)
        ATY = torch.linalg.matmul(A.mT, log_sig[..., None])
        beta = torch.linalg.matmul(ATA_inv, ATY)
        r2s[..., i] = -torch.squeeze(beta)[..., 0]
        residual[..., i] = torch.mean((log_sig - torch.squeeze(torch.linalg.matmul(A, beta)))**2, dim=-1)
        # ToDo residual[..., i] =
    r2s = torch.sum(r2s * weights[None, None], dim=-1)
    residual = torch.sum(residual * weights[None, None], dim=-1)
    nifti_save(torch.clamp_min(r2s, 0.0), img_aff=affine, path_to_dir=path, file_name="r2s_rough_chw")

    weights = 1 - (torch.clamp(residual, 0.0, 0.6)) / 0.6
    weights = smooth_map(weights, kernel_size=8)
    r2s = torch.nan_to_num(
        torch.sum(weights * r2s, dim=-1) / torch.sum(weights, dim=-1)
    )
    nifti_save(torch.clamp_min(r2s, 0.0), img_aff=affine, path_to_dir=path, file_name="r2s_rough")
    nifti_save(weights, img_aff=affine, path_to_dir=path, file_name="r2s_weights")

    logger.info(f"rough estimate R2+")
    r2dag = torch.squeeze(torch.zeros((*data_normed_se.shape[:-1], 2)))
    residual = torch.squeeze(torch.zeros((*data_normed_se.shape[:-1], 2)))
    weights = torch.tensor((3/4, 1/4))

    wavg = torch.linspace(1, 0.5, 5)
    wavg /= torch.sum(wavg)
    for i in range(2):
        dag_indices = torch.arange(5, 10) + 10 * i
        data_dag = torch.abs(data_rsos[..., dag_indices])
        te_gre = gre_attenuation_times[dag_indices]
        # we assume that the data after the initial spin echo decays with some r2s
        log_sig = torch.zeros_like(data_dag)
        log_sig[data_dag > 0] = torch.log(data_dag[data_dag > 0])
        A = torch.ones((*data_dag.shape, 2))
        A[..., 0] = te_gre

        ATA = torch.linalg.matmul(A.mT, A)
        ATA_inv = torch.linalg.inv(ATA)
        ATY = torch.linalg.matmul(A.mT, log_sig[..., None])
        beta = torch.linalg.matmul(ATA_inv, ATY)
        r2dag[..., i] = torch.squeeze(beta)[..., 0]
        residual[..., i] = torch.mean((log_sig - torch.squeeze(torch.linalg.matmul(A, beta)))**2, dim=-1)
        # ToDo residual[..., i] =
    r2dag = torch.sum(r2dag * weights[None, None], dim=-1)
    residual = torch.sum(residual * weights[None, None], dim=-1)
    nifti_save(r2dag, img_aff=affine, path_to_dir=path, file_name="r2s_dagger_chw")

    weights = 1 - (torch.clamp(residual, 0.0, 0.6)) / 0.6
    weights = smooth_map(weights, kernel_size=8)
    r2dag = torch.nan_to_num(
        torch.sum(weights * r2dag, dim=-1) / torch.sum(weights, dim=-1)
    )
    nifti_save(torch.clamp_min(r2dag, 0.0), img_aff=affine, path_to_dir=path, file_name="r2dagger_rough")
    nifti_save(weights, img_aff=affine, path_to_dir=path, file_name="r2_dagger_weights")

    logger.info(f"rough estimate R2 & R2'")
    rough_r2 = 0.5 * (r2s + r2dag)
    rough_r2p = 0.5 * (r2s - r2dag)
    nifti_save(rough_r2, img_aff=affine, path_to_dir=path, file_name="r2_rough")
    nifti_save(rough_r2p, img_aff=affine, path_to_dir=path, file_name="r2p_rough")

    logger.info(f"Estimate rough B0 / B1")
    for idx_z in range(data_normed_se.shape[2]):
        logger.info(f"Process slice: {idx_z + 1} / {data_normed_se.shape[2]}")
        for idx_x in tqdm.trange(data_normed_se.shape[0]):
            data_batch = data_normed_se[idx_x, :, idx_z].to(device)

            # l2 = torch.linalg.norm(data_batch[None] - db_se[:, None], dim=-1)
            # vals, indices = torch.min(l2, dim=0)
            # vals, indices = torch.min(l2, dim=0)
            dot = torch.linalg.vecdot(data_batch[None], db_se[:, None], dim=-1)
            # want to maximize alignment but allow for phase offset
            loss = torch.abs(dot)
            # loss = dot.real + torch.abs(dot.imag)
            vals, indices = torch.max(loss, dim=0)

            batch_t1t2b1b0_vals = t1t2b1b0_vals[indices]

            b1[idx_x, :, idx_z] = batch_t1t2b1b0_vals[..., 2].cpu()
            b0[idx_x, :, idx_z] = batch_t1t2b1b0_vals[..., 3].cpu()
            loss_residual[idx_x, :, idx_z] = vals

    logger.info("B1 smoothing")
    b1_map = smooth_map(b1, kernel_size=min(b1.shape[:2]) // 32)

    nifti_save(b1, img_aff=affine, path_to_dir=path, file_name="reg_b1_estimate")
    nifti_save(b1_map, img_aff=affine, path_to_dir=path, file_name="reg_b1_smoothed")

    # if data_shape.__len__() == 5:
    #     b1_map = torch.mean(b1_map, dim=-1)
    # weights = (l2_residual_upper_limit - torch.clamp(l2_res, 0, l2_residual_upper_limit)) / l2_residual_upper_limit
    weights = (
            torch.clamp(loss_residual, loss_residual_lower_limit, loss_residual_upper_limit) - loss_residual_lower_limit
              ) / (loss_residual_upper_limit - loss_residual_lower_limit)
    weights = smooth_map(weights, kernel_size=5)
    b1_map = torch.nan_to_num(
        torch.sum(weights * b1_map, dim=-1) / torch.sum(weights, dim=-1)
    )

    nifti_save(weights, img_aff=affine, path_to_dir=path, file_name="reg_b1_weights")
    nifti_save(b1_map, img_aff=affine, path_to_dir=path, file_name="reg_b1_combined")

    logger.info("prep data")
    data_norm = torch.linalg.norm(data_rsos, dim=-1, keepdim=True)
    data_normed = torch.nan_to_num(data_rsos / data_norm)
    # data_normed = torch.abs(data_normed).unsqueeze(-2)
    data_normed = data_normed.unsqueeze(-2)
    b1_map = b1_map.unsqueeze(-1)
    # sample r2p
    r2p_att = torch.exp(-gre_attenuation_times[None] * r2p_vals[:, None]).to(device)
    logger.info("sample r2p db")
    db_r2p = torch.zeros((*db_shape[:-1], r2p_att.shape[0], db_shape[-1]), dtype=db_pattern.dtype)

    for idx_t2 in tqdm.trange(t2_vals.shape[0]):
        # for each t2 value we sample the database, this batching is only for computational and memory efficiency
        db_batch = db_pattern[:, idx_t2].to(device)
        db_batch = db_batch.unsqueeze(-2) * r2p_att[None, None, None]
        db_batch = torch.nan_to_num(db_batch / torch.linalg.norm(db_batch, dim=-1, keepdim=True))
        db_r2p[:, idx_t2] = db_batch.cpu()
    # now has dims [t1, t2, ny, b0, t]
    # r2p samples have dims [r2p, t] - > want [t1, t2, ny, b0, r2p, t]

    logger.info("B1 Regularized matching")
    batch_size = 100
    num_batches = int(np.ceil(data_normed.shape[1] / batch_size))

    for idx_z in range(data_normed.shape[2]):
        for idx_x in tqdm.trange(data_normed.shape[0]):
        # for idx_x in tqdm.trange(10):
            for idx_by in range(num_batches):
                start = idx_by * batch_size
                end = min((idx_by + 1) * batch_size, data_normed.shape[1])
                data_batch = data_normed[idx_x, start:end, idx_z].to(device)
                # now has dims [ny, t]
                b1_batch = b1_map[idx_x, start:end, idx_z]
                loss_b1 = torch.abs(b1_batch[None] - b1_vals[:, None])
                _, b1_ind = torch.min(loss_b1, dim=0)

                # b0_batch = b0_map[idx_x, start:end, idx_z]
                # loss_b0 = torch.abs(b0_batch[None] - b0_vals[:, None])
                # _, b0_ind = torch.min(loss_b0, dim=0)

                db_batch = db_r2p[:, :, b1_ind].to(device)
                # now has dims [t1, t2, nyb, b0, r2p, t)

                # move spatial ny dimension close to the time
                db_batch = torch.movedim(db_batch, 2, -2)
                db_batch = torch.reshape(db_batch, (-1, *db_batch.shape[-2:]))
                # match
                # l2 = torch.linalg.norm(data_batch[None] - db_batch, dim=-1)
                # vals, indices = torch.min(l2, dim=0)
                dot = torch.linalg.vecdot(data_batch[None], db_batch, dim=-1)
                # want to maximize alignment but allow for phase offset
                # loss = dot.real + torch.abs(dot.imag)
                loss = torch.abs(dot)
                vals, indices = torch.max(loss, dim=0)
                po = -torch.angle(dot[indices, torch.arange(dot.shape[1], device=dot.device)])

                batch_t1t2b0r2p = t1t2b0r2p_vals[indices]
                loss_residual[idx_x, start:end, idx_z] = vals
                t2[idx_x, start:end, idx_z] = batch_t1t2b0r2p[:, 1]
                b0[idx_x, start:end, idx_z] = batch_t1t2b0r2p[:, 2]
                r2p[idx_x, start:end, idx_z] = batch_t1t2b0r2p[:, 3]
                phase_offset[idx_x, start:end, idx_z] = po

    # logger.info("B0 smoothing")
    # b0_map = smooth_map(b0, kernel_size=min(b1.shape[:2]) // 32)
    # nifti_save(b0, img_aff=affine, path_to_dir=path, file_name="se_b0_estimate")
    # nifti_save(b0_map, img_aff=affine, path_to_dir=path, file_name="se_b0_smoothed")
    #
    # logger.info("B0 Regularized matching")
    # for idx_z in range(data_normed.shape[2]):
    #     for idx_x in tqdm.trange(data_normed.shape[0]):
    #     # for idx_x in tqdm.trange(20):
    #         for idx_by in range(num_batches):
    #             start = idx_by * batch_size
    #             end = min((idx_by + 1) * batch_size, data_normed.shape[1])
    #             data_batch = data_normed[idx_x, start:end, idx_z].to(device)
    #             # now has dims [ny, t]
    #
    #             # want some neighborhood rather than just the point
    #             b1_batch = b1_map[idx_x, start:end, idx_z]
    #             loss_b1 = torch.abs(b1_batch[None] - b1_vals[:, None])
    #             _, b1_ind = torch.min(loss_b1, dim=0)
    #
    #             b0_batch = b0_map[idx_x, start:end, idx_z]
    #             loss_b0 = torch.abs(b0_batch[None] - b0_vals[:, None])
    #             _, b0_ind = torch.min(loss_b0, dim=0)
    #
    #             db_batch = db_r2p[:, :, b1_ind, b0_ind].to(device)
    #             # now has dims [t1, t2, nyb, r2p, t)
    #
    #             # move spatial ny dimension close to the time
    #             db_batch = torch.movedim(db_batch, 2, -2)
    #             db_batch = torch.reshape(db_batch, (-1, *db_batch.shape[-2:]))
    #             # match
    #             # l2 = torch.linalg.norm(data_batch[None] - db_batch, dim=-1)
    #             # vals, indices = torch.min(l2, dim=0)
    #             dot = torch.linalg.vecdot(data_batch[None], db_batch, dim=-1)
    #             # want to maximize alignment but allow for phase offset
    #             # loss = dot.real + torch.abs(dot.imag)
    #             loss = torch.abs(dot)
    #             vals, indices = torch.max(loss, dim=0)
    #             po = -torch.angle(dot[indices, torch.arange(dot.shape[1], device=dot.device)])
    #
    #             batch_t1t2r2p = t1t2r2p_vals[indices]
    #
    #             loss_residual[idx_x, start:end, idx_z] = vals
    #             t2[idx_x, start:end, idx_z] = batch_t1t2r2p[:, 1]
    #             b0[idx_x, start:end, idx_z] = b0_vals[b0_ind]
    #             b1[idx_x, start:end, idx_z] = b1_vals[b1_ind]
    #             r2p[idx_x, start:end, idx_z] = batch_t1t2r2p[:, 2]
    #             phase_offset[idx_x, start:end, idx_z] = po

    r2 = torch.nan_to_num(1 / t2)
    t2 = 1e3 * t2

    logger.info(f"Combining channels")
    # weights = (l2_residual_upper_limit - torch.clamp(l2_res, 0, l2_residual_upper_limit)) / l2_residual_upper_limit
    weights = (
        torch.clamp(loss_residual, loss_residual_lower_limit, loss_residual_upper_limit) - loss_residual_lower_limit
              ) / (loss_residual_upper_limit - loss_residual_lower_limit)
    weights = smooth_map(weights, kernel_size=5)

    # weighted averaging
    r2_combined = torch.nan_to_num(
        torch.sum(weights * r2, dim=-1) / torch.sum(weights, dim=-1)
    )
    r2p_combined = torch.nan_to_num(
        torch.sum(weights * r2p, dim=-1) / torch.sum(weights, dim=-1)
    )
    r2s = r2_combined + r2p_combined
    b0 = torch.nan_to_num(
        torch.sum(weights * b0, dim=-1) / torch.sum(weights, dim=-1)
    )
    b1 = torch.nan_to_num(
        torch.sum(weights * b1, dim=-1) / torch.sum(weights, dim=-1)
    )

    logger.info(f"Saving channel fits")
    # reshape & save
    names = ["ch_optimize_residual", "ch_r2", "ch_t2", "weights", "r2", "r2p", "r2s", "b0", "b1", "phase_offset"]
    for i, r in enumerate([loss_residual, r2, t2, weights, r2_combined, r2p_combined, r2s, b0, b1, phase_offset]):
        nifti_save(
            r, img_aff=affine,
            path_to_dir=path, file_name=names[i]
        )


def fit_revisited():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"set device: {device}")
    logger.info(f"Load data")
    # data, img = nifti_load(
    #     "/data/pt_np-jschmidt/data/01_in_vivo_scan_data/paper_protocol_mese/7T/2023-12-08/processed/"
    #     "denoising/d_mppca_fixed-p-1_mod_semc_r0p7_fa140_z4000_pat2-36_sl31-200_esp9p5_nbc-manjon.nii"
    # )
    # data = torch.from_numpy(data).unsqueeze(-2)
    # data = data[:, :, data.shape[2] // 2, None]
    # affine = torch.from_numpy(img.affine)
    data = torch_load(
        "/data/pt_np-jschmidt/data/00_phantom_scan_data/2025-03-20_mese_fs/raw/mese_fs_vfa/img_rmos_slice.pt"
    )
    # data = torch.squeeze(data)
    affine = torch_load(
        "/data/pt_np-jschmidt/data/00_phantom_scan_data/2025-03-20_mese_fs/raw/mese_fs_vfa/affine.pt"
    )
    logger.info(f"load database")
    db = DB.load(
        "/data/pt_np-jschmidt/data/00_phantom_scan_data/2025-03-20_mese_fs/emc/mese_vfa/mese_vfa_kernels.pkl"
    )
    path = plib.Path(
        "/data/pt_np-jschmidt/data/00_phantom_scan_data/2025-03-20_mese_fs/processing/mese_fs_vfa/ch_l2_abs"
    )
    # get torch tensors
    db_torch_mag, db_torch_phase = db.get_torch_tensors_t1t2b1b0e()
    # get t2 and b1 values
    t1_vals, t2_vals, b1_vals, b0_vals = db.get_t1_t2_b1_b0_values()

    # normalize data
    # data_normed = root_sum_of_squares(input_data=data, dim_channel=-2)
    data_norm = torch.linalg.norm(data, dim=-1, keepdim=True)
    data_normed = torch.nan_to_num(data / data_norm)
    data_normed = torch.abs(data_normed)

    # set database
    db_pattern = db_torch_mag * torch.exp(1j * db_torch_phase)
    db_shape = db_pattern.shape
    # normalize database

    db_norm = torch.linalg.norm(db_pattern, dim=-1, keepdim=True)
    db_pattern_normed = torch.nan_to_num(db_pattern / db_norm)
    db_pattern_normed = torch.reshape(db_pattern_normed, (-1, db_pattern_normed.shape[-1]))
    # abs values
    db_pattern_normed = torch.abs(db_pattern_normed)
    db_pattern_normed = db_pattern_normed.to(dtype=data_normed.dtype, device=device)

    t1t2b1b0_vals = torch.tensor([
        [t1, t2, b1, b0] for t1 in t1_vals for t2 in t2_vals for b1 in b1_vals for b0 in b0_vals
    ], device=device)
    t1t2b0_vals = torch.tensor([
        [t1, t2, b0] for t1 in t1_vals for t2 in t2_vals for b0 in b0_vals
    ], device=device)

    t2 = torch.zeros(data_normed.shape[:-1])
    b1 = torch.zeros(data_normed.shape[:-1])
    l2_res = torch.zeros(data_normed.shape[:-1])

    logger.info(f"l2 fit - b1 estimate")
    batch_size = 1
    num_batches = int(np.ceil(data_normed.shape[0] / batch_size))
    for idx_c in tqdm.trange(data_normed.shape[-2], desc="channel wise processing"):
        for idx_z in range(data_normed.shape[2]):
            for idx_x in range(num_batches):
                start = idx_x * batch_size
                end = min((idx_x + 1) * batch_size, data_normed.shape[0])
                data_batch = data_normed[start:end, :, idx_z, idx_c, :].to(device)

                l2 = torch.linalg.norm(data_batch[:, None] - db_pattern_normed[None, :,  None], dim=-1)
                vals, indices = torch.min(l2, dim=1)
                batch_t1t2b1b0_vals = t1t2b1b0_vals[indices]

                b1[start:end, :, idx_z, idx_c] = batch_t1t2b1b0_vals[..., 2].cpu()
                l2_res[start:end, :, idx_z, idx_c] = vals

    logger.info("B1 smoothing")
    b1_map = smooth_map(b1, kernel_size=min(b1.shape[:2]) // 32)

    nifti_save(b1, img_aff=affine, path_to_dir=path, file_name="reg_b1_estimate")
    nifti_save(b1_map, img_aff=affine, path_to_dir=path, file_name="reg_b1_smoothed")

    # if data_shape.__len__() == 5:
    #     b1_map = torch.mean(b1_map, dim=-1)
    l2_residual_upper_limit = 0.15
    weights = (l2_residual_upper_limit - torch.clamp(l2_res, 0, l2_residual_upper_limit)) / l2_residual_upper_limit
    weights = smooth_map(weights, kernel_size=5)
    b1_map = torch.nan_to_num(
        torch.sum(weights * b1_map, dim=-1) / torch.sum(weights, dim=-1)
    )

    nifti_save(weights, img_aff=affine, path_to_dir=path, file_name="reg_b1_weights")
    nifti_save(b1_map, img_aff=affine, path_to_dir=path, file_name="reg_b1_combined")

    db_pattern_normed = torch.reshape(db_pattern_normed, db_shape)
    db_pattern_normed = torch.movedim(db_pattern_normed, -2, 2)
    db_pattern_normed = torch.reshape(db_pattern_normed, (-1, *db_pattern_normed.shape[-2:]))
    l2_res = torch.zeros(data_normed.shape[:-1])

    logger.info("Regularized fit")
    # do slice wise processing
    for idx_z in tqdm.trange(data_normed.shape[2], desc="Processing, slice wise with b1 reg."):
        for idx_c in range(data_normed.shape[3]):
            data_batch = data_normed[:, :, idx_z, idx_c].to(device)
            b1_batch = b1_map[:, :, idx_z]
            # want the database to be pulled towards the b1 regularization
            b1_loss = torch.abs(b1_batch[None] - b1_vals[:, None, None])
            _, b1_indices = torch.min(b1_loss, dim=0)
            db_batch = db_pattern_normed[:, b1_indices, :]
            # now db dims match with data batch
            # db [t1t2b0, nx, ny, nc, t], data_batch [nx, ny, nc, t]
            loss = torch.linalg.norm(
                torch.abs(data_batch[None]) - torch.abs(db_batch), dim=-1
            )
            vals, indices = torch.min(loss, dim=0)

            # loss = torch.linalg.vecdot(data_batch[None], db_batch, dim=-1)
            # dot_mag = torch.abs(loss)
            # dot_phase = torch.abs(torch.angle(loss))
            # dot = (1 - phase_weighting) * dot_mag - phase_weighting * dot_phase
            # vals, indices = torch.max(dot, dim=0)

            batch_t1t2b0 = t1t2b0_vals[indices]

            l2_res[:, :, idx_z, idx_c] = vals
            t2[:, :, idx_z, idx_c] = batch_t1t2b0[..., 1].cpu()

    r2 = torch.nan_to_num(1 / t2)
    t2 = 1e3 * t2

    logger.info(f"Combining channels")
    weights = (l2_residual_upper_limit - torch.clamp(l2_res, 0, l2_residual_upper_limit)) / l2_residual_upper_limit
    weights = smooth_map(weights, kernel_size=5)

    # weighted averaging
    r2_combined = torch.nan_to_num(
        torch.sum(weights * r2, dim=-1) / torch.sum(weights, dim=-1)
    )

    logger.info(f"Saving channel fits")
    # reshape & save
    names = ["ch_optimize_residual", "ch_r2", "ch_t2", "weights", "r2"]
    for i, r in enumerate([l2_res, r2, t2, weights, r2_combined]):
        nifti_save(
            r, img_aff=affine,
            path_to_dir=path, file_name=names[i]
        )


def load_input_field(in_type: InputType, settings: EmcFitSettings, data_shape: tuple):
    match in_type:
        case InputType.B1:
            p_in = settings.input_b1
        case InputType.B0:
            p_in = settings.input_b0
        case InputType.B1ERR:
            p_in = settings.input_b1_err
        case _:
            msg = f"Unknown input type: {in_type.name}"
            logger.error(msg)
            raise ValueError(msg)
    path_in = plib.Path(p_in)
    if not path_in.is_file():
        logger.info(f"No {in_type.name} input given or input invalid, estimating {in_type.name} from data")
        data = None
    else:
        map = torch.from_numpy(nifti_load(path_in)[0])
        if settings.process_slice:
            map = map[:, :, map.shape[2] // 2, None]
        while map.ndim < len(data_shape) - 2:
            map = map.unsqueeze(-1)
        data = map.expand(data_shape[:-2])
    if data is not None:
        match in_type:
            case InputType.B1:
                if data.max() > 10:
                    data /= 100
                data /= settings.tx_factor
            # case InputType.B1ERR:
            #     if data.max() > 100:
            #         data /= 100
        logger.info(f"\t\t- {in_type.name} Input value range: {(data[data.abs() > 1e-6].min().item(), data.max().item())}")
    return data


def wrap_cli(settings: EmcFitSettings):
    """
    Function to wrap the command line arguments dataclass, load data and call the core function
    """
    device = torch.device("cuda:0") if settings.use_gpu and torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"set device: {device}")

    logger.info(f"Load data")
    in_path = plib.Path(settings.input_data).absolute()
    if ".nii" in in_path.suffixes:
        data, img = nifti_load(in_path)
        data = torch.from_numpy(data)
        affine = torch.from_numpy(img.affine)
        logger.info(f"\t\t- found data dims: {data.shape}")
        if data.ndim < 5:
            # first assume combined channel input
            data.unsqueeze_(-2)
            logger.info(f"\t\t- assumed compressed channel dim: {data.shape}")
        if data.ndim < 5:
            # assume compressed slices
            data.unsqueeze_(2)
            logger.info(f"\t\t- assumed compressed slice dim: {data.shape}")
    else:
        data = torch_load(settings.input_data)
        affine = torch_load(settings.input_affine) if settings.input_affine else torch.eye(4)

    if settings.process_slice:
        data = data[:, :, data.shape[2] // 3, None]

    logger.info(f"load database")
    db = DB.load(settings.input_database)
    vals = db.get_t1_t2_b1_b0_values()
    logger.info(f"\t\t- Value range:")
    for i in range(4):
        logger.info(f"\t\t\t {['T1', 'T2', 'B1', 'B0'][i]}: {(vals[i].min().item(), vals[i].max().item())}")

    path = plib.Path(settings.out_path)

    logger.info(f"Prepare data")
    # get torch tensors
    db_torch_mag, db_torch_phase = db.get_torch_tensors_t1t2b1b0e()
    # get t2 and b1 values
    t1_vals, t2_vals, b1_vals, b0_vals = db.get_t1_t2_b1_b0_values()

    # fft reconned k-space
    if not settings.input_in_image_space:
        logger.info(f"FFT data into image space")
        data = fft_to_img(data, dims=(0, 1))

    # set database
    db_pattern = db_torch_mag * torch.exp(1j * db_torch_phase)

    b1_data = load_input_field(in_type=InputType.B1, settings=settings, data_shape=data.shape)
    b1_err_data = load_input_field(in_type=InputType.B1ERR, settings=settings, data_shape=data.shape)
    b0_data = load_input_field(in_type=InputType.B0, settings=settings, data_shape=data.shape)

    if settings.rsos_channel_combine:
        # b1_data = root_sum_of_squares(b1_data, dim_channel=-1).unsqueeze(-1) if b1_data is not None else None
        data = root_sum_of_squares(input_data=data, dim_channel=-2).unsqueeze(-2)

    fit_mese(
        data_xyzce=data, db_t1t2b1b0=db_pattern,
        t1_vals=t1_vals, t2_vals=t2_vals, b1_vals=b1_vals, b0_vals=b0_vals,
        path_out=path, b1_data=b1_data, b1_err_data=b1_err_data, b1_err_thr=settings.b1_error_cutoff,
        b0_data=b0_data,
        device=device, input_affine=affine,
        lr_reg=settings.low_rank_regularisation
    )


def normalize_data(data, dim: int = -1):
    norm = torch.linalg.norm(data, dim=dim, keepdims=True)
    return torch.nan_to_num(data / norm, posinf=0.0, nan=0.0)


def prep_dims_bg_field(data: torch.Tensor, shape):
    if data is None:
        data_alloc = torch.zeros(shape[:-2])
        data_alloc_sm = torch.zeros(shape[:-2])
        fit = True
    else:
        while data.ndim > 3:
            data = data[..., 0]
        while data.ndim < 3:
            data = data.unsqueeze(-1)
        data = data.expand(shape[:3])
        data_alloc = data_alloc_sm = data.clone()
        fit = False
    return data_alloc, data_alloc_sm, fit


def estimate_fields_from_db(
        data_5d: torch.Tensor, db_t1t2b1b0: torch.Tensor, device: torch.device,
        data_b1: torch.Tensor = None, data_b0: torch.Tensor = None,
        b1_vals: torch.Tensor = None, b0_vals: torch.Tensor = None,
        b1_err_data: torch.Tensor = None, b1_err_thr: float = 0.15,
        batch_size: int = 10):
    if data_b1 is not None and data_b0 is not None:
        return data_b1, data_b1, data_b0, data_b0

    logger.info(f"l2 fit - background field estimate")

    if data_5d.shape[-2] > 1:
        data_5d = root_sum_of_squares(data_5d, dim_channel=-2).unsqueeze(-2)

    data_5d = normalize_data(data_5d, dim=-1)

    num_batches = int(np.ceil(data_5d.shape[0] / batch_size))
    data_b1_out, data_b1_sm, fit_b1 = prep_dims_bg_field(data_b1, data_5d.shape)
    data_b0, data_b0_sm, fit_b0 = prep_dims_bg_field(data_b0, data_5d.shape)

    if b1_err_data is not None:
        # if b1 error maps are given we calculate a combined b1+ map from the input, the error maps and the emc esimate,
        # hence we want to fit b1 from the dictionary
        fit_b1 = True
    if not fit_b1:
        b1_vals_device = b1_vals.to(device)
    if not fit_b0:
        b0_vals_device = b0_vals.to(device)

    db_shape = db_t1t2b1b0.shape
    nt1, nt2, nb1, nb0, nt = db_shape

    db_t1t2b1b0 = db_t1t2b1b0.to(device=device, dtype=data_5d.dtype)

    if data_5d.ndim < 5:
        # if data is still smaller we throw an error
        msg = f"Input data assumed to be at least 4D (x y z (c) e), got: {data_5d.shape}"
        logger.error(msg)
        raise AttributeError(msg)

        # logger.info(f"Channel wise processing - channel {idx_c + 1} / {data_5d.shape[-2]}")
    for idx_z in tqdm.trange(data_5d.shape[2], desc="slice wise processing"):
        # we process per B0 simulated value, otherwise the memory explodes
        for idx_x in range(num_batches):
            start = idx_x * batch_size
            end = min((idx_x + 1) * batch_size, data_5d.shape[0])
            data_batch = data_5d[start:end, :, idx_z, 0, :].to(device)
            d_shape = data_batch.shape
            data_batch = data_batch.view(-1, d_shape[-1])

            if fit_b1 and fit_b0:
                # we estimate both at once just brute force the minimum l2
                l2 = torch.cdist(
                    data_batch.view(-1, d_shape[-1]),
                    db_t1t2b1b0.view(-1,db_shape[-1])
                )

                vals, indices = torch.min(l2, dim=-1)
                indices = unflatten_batched_indices(indices, db_shape[:-1])
                data_b1_out[start:end, :, idx_z] = b1_vals[indices[..., 2].cpu()].reshape(d_shape[:-1])
                data_b0[start:end, :, idx_z] = b0_vals[indices[..., 3].cpu()].reshape(d_shape[:-1])
            elif not fit_b1:
                # b1 is given, we create a batched database which is as close as possible at each given b1 value per pixel
                b1_batch = data_b1[start:end, :, idx_z].to(device)
                # want the database to be pulled towards the b1 regularization,
                # b1-batch [nx, ny]
                b1_loss = torch.abs(b1_batch[None] - b1_vals_device[:, None, None])
                _, b1_indices = torch.min(b1_loss, dim=0)

                db_batch = db_t1t2b1b0[:, :, b1_indices]
                # some fiddling around with dims, we want the spatial dims to go to the back and db dims to front
                db_batch = db_batch.permute(0, 1, 4, 2, 3, 5)
                # merge respective db and data dims
                db_batch = db_batch.reshape(nt1 * nt2 * nb0, -1, nt)

                # match
                l2 = torch.linalg.norm(
                    data_batch[None] - db_batch, dim=-1
                )
                vals, indices = torch.min(l2, dim=0)
                indices = unflatten_batched_indices(indices, (nt1, nt2, nb0))

                data_b0[start:end, :, idx_z] = b0_vals[indices[..., 2].cpu()].reshape(d_shape[:-1])

            else:
                # b0 is given, we create a batched database which is as close as possible at each given b0 value per pixel
                b0_batch = data_b0[start:end, :, idx_z].to(device)
                # want the database to be pulled towards the b0 regularization,
                # b1-batch [nx, ny]
                b0_loss = torch.abs(b0_batch[:, :, None] - b0_vals_device[None, None, :])

                # usually this is not possible due to memory
                # (much more b1 values simulated and assigning for each point along y axis is costly)
                # thus we use a combined penalty instead

                # match
                l2 = torch.cdist(
                    data_batch.view(-1, d_shape[-1]),
                    db_t1t2b1b0.view(-1, db_shape[-1])
                )

                loss = l2.view(*d_shape[:-1], *db_shape[:-1]) + b0_loss[:, :, None, None, None]
                loss = loss.view(*d_shape[:-1], -1)
                vals, indices = torch.min(loss, dim=-1)
                indices = unflatten_batched_indices(indices, (nt1, nt2, nb1, nb0))

                data_b1_out[start:end, :, idx_z] = b1_vals[indices[..., 2].cpu()]
            # field_sm[:, :, idx_z, idx_c] = smooth_map(field_alloc, kernel_size=min(field_alloc.shape[:2]) // 30)
    if fit_b0:
        data_b0_sm = torch.from_numpy(gaussian_filter(data_b0.numpy(), sigma=10, axes=(0, 1, 2)))
    else:
        data_b0_sm = data_b0
    if fit_b1:
        if b1_err_data is not None:
            # calculate combined map
            # we use a weighted averaging, with a linear weighting factor between 0 and b1_err_thr the threshold.
            # i.e. for 0 b1 error we use the b1 input, for b1_err_thr error we use the b1 emc estimate
            weights = torch.clamp(b1_err_data.abs(), 0.0, b1_err_thr) / b1_err_thr
            data_b1 = weights * data_b1_out + (1 - weights) * data_b1
        else:
            data_b1 = data_b1_out
        data_b1_sm = torch.from_numpy(gaussian_filter(data_b1.numpy(), sigma=10, axes=(0, 1, 2)))
    else:
        data_b1_sm = data_b1
    return data_b1_sm, data_b1, data_b0_sm, data_b0


def unflatten_batched_indices(flat_indices, shape):
    """
    Convert batched flattened indices back to individual dimension indices.

    Args:
        flat_indices (torch.Tensor): Batched flattened indices with shape [b1, b2, ..., flat_inds]
        shape (tuple): Original tensor shape to unflatten into

    Returns:
        torch.Tensor: Indices for each dimension with shape [b1, b2, flat_inds, len(shape)]
    """
    # Ensure flat_indices is a tensor
    if not isinstance(flat_indices, torch.Tensor):
        flat_indices = torch.tensor(flat_indices)

    # Prepare output tensor
    out_indices = torch.zeros(
        *flat_indices.shape,
        len(shape),
        dtype=torch.long,
        device=flat_indices.device
    )

    # Iterate through dimensions in reverse
    for dim in range(len(shape) - 1, -1, -1):
        # Compute index for current dimension
        out_indices[..., dim] = flat_indices % shape[dim]

        # Update flat_indices by integer division
        flat_indices //= shape[dim]

    return out_indices


def regularised_fit(
        data: torch.Tensor, db_t1t2b1b0: torch.Tensor,
        t1_vals: torch.Tensor, t2_vals: torch.Tensor,
        b1_data: torch.Tensor, b1_vals: torch.Tensor,
        b0_data: torch.Tensor, b0_vals: torch.Tensor,
        device: torch.device, lr_reg: int = None) -> (torch.Tensor, torch.Tensor):
    logger.info("Regularized fit")
    # prepare the database for b1 and b0 regularization
    db_shape = db_t1t2b1b0.shape
    # db_b1_reg = torch.movedim(db_t1t2b1b0, -2, 2)
    # db_b1_reg = torch.reshape(db_b1_reg, (-1, *db_b1_reg.shape[-2:])).to(device)
    # db_reg = db_t1t2b1b0.view(-1, db_t1t2b1b0.shape[-3:]).to(device)
    db_t1t2b1b0 = db_t1t2b1b0.to(device)
    b1_vals = b1_vals.to(device)
    b0_vals = b0_vals.to(device)
    t1_vals = t1_vals.to(device)
    t2_vals = t2_vals.to(device)

    # allocate
    l2_res = torch.zeros(data.shape[:-1])
    t2 = torch.zeros(data.shape[:-1])

    # create broadcastable meshgrid for database indexing
    t1_indices,  t2_indices = torch.meshgrid(
        torch.arange(db_shape[0], device=db_t1t2b1b0.device),
        torch.arange(db_shape[1], device=db_t1t2b1b0.device),
        indexing="ij"
    )
    # broadcast
    t1_indices = t1_indices[:, :, None, None].expand(*db_shape[:2], *data.shape[:2])
    t2_indices = t2_indices[:, :, None, None].expand(*db_shape[:2], *data.shape[:2])

    # do slice wise processing
    logger.info(f"Processing, slice wise with B1 reg. and low-rank regularisation set to {lr_reg}")
    for idx_z in range(data.shape[2]):
        logger.info(f"______")
        logger.info(f"Processing, slice wise :: Slice {idx_z + 1} / {data.shape[2]}")
        for idx_c in tqdm.trange(data.shape[3], desc="Channel wise processing."):
            data_batch = data[:, :, idx_z, idx_c].to(device)
            b1_batch = b1_data[:, :, idx_z, idx_c].to(device)
            # want the database to be pulled towards the b1 and b0 regularization,
            # b1-batch [nx, ny]
            b1_loss = torch.abs(b1_batch[None] - b1_vals[:, None, None])
            _, b1_indices = torch.min(b1_loss, dim=0)

            b0_batch = b0_data[:, :, idx_z, idx_c].to(device)
            # b0-batch [nx, ny]
            b0_loss = torch.abs(b0_batch[None] - b0_vals[:, None, None])
            _, b0_indices = torch.min(b0_loss, dim=0)
            # we now have indices specifying the sub-database to take for every point (channel and slice wise processing)
            # thus [nx, ny]

            # we want to grab the sub-database generated by the respective b0 and b1
            b0_indices = b0_indices[None, None].expand_as(t1_indices)
            b1_indices = b1_indices[None, None].expand_as(t1_indices)
            db_batch = db_t1t2b1b0[t1_indices, t2_indices, b1_indices, b0_indices, :]

            # now db dims match with data batch
            if lr_reg is not None:
                # use a low rank patch based constraint to stabilise the fitting
                batch_t2, batch_l2_res = match_lr_constrained(
                    data_batch_xyt=data_batch, db_batch_t1t2xyt=db_batch,
                    t1_vals=t1_vals, t2_vals=t2_vals, rank=lr_reg
                )
            else:
                # just match pixel wise
                batch_t2, batch_l2_res = match(
                    data_batch_xyt=data_batch, db_batch_t1t2xyt=db_batch,
                    t1_vals=t1_vals, t2_vals=t2_vals
                )

            l2_res[:, :, idx_z, idx_c] = batch_l2_res
            t2[:, :, idx_z, idx_c] = batch_t2.cpu()

            # loss = torch.linalg.vecdot(data_batch[None], db_batch, dim=-1)
            # dot_mag = torch.abs(loss)
            # dot_phase = torch.abs(torch.angle(loss))
            # dot = (1 - phase_weighting) * dot_mag - phase_weighting * dot_phase
            # vals, indices = torch.max(dot, dim=0)

    return t2, l2_res

def match(
        data_batch_xyt: torch.Tensor, db_batch_t1t2xyt: torch.Tensor,
        t1_vals: torch.Tensor, t2_vals: torch.Tensor):
    db_batch_t1t2xyt = db_batch_t1t2xyt.view(-1, *data_batch_xyt.shape)
    loss = torch.linalg.norm(
        torch.abs(data_batch_xyt[None]) - torch.abs(db_batch_t1t2xyt), dim=-1
    )
    vals, indices = torch.min(loss, dim=0)

    indices = unflatten_batched_indices(indices, (t1_vals.shape[0], t2_vals.shape[0]))
    t1s = t1_vals[indices[..., 0]]
    t2s = t2_vals[indices[..., 1]]
    return t2s, vals


def match_lr_constrained(
        data_batch_xyt: torch.Tensor, db_batch_t1t2xyt: torch.Tensor,
        t1_vals: torch.Tensor, t2_vals: torch.Tensor,
        rank: int = 2, patch_size: int = 3):
    shape = data_batch_xyt.shape
    # we want to rearrange the data to form small local neighborhoods,
    # for which we presume the b1 input value of its central point (i.e. the same database batch)
    indices, matrix_shape = get_linear_indices(
        k_space_shape=shape[:2], patch_shape=(patch_size, patch_size), sample_directions=(1, 1)
    )

    indices = indices.view(matrix_shape)
    in_db = db_batch_t1t2xyt.reshape(
        *db_batch_t1t2xyt.shape[:2],
        -1, # flatten the spatial dims such that we can use the same indexing
        db_batch_t1t2xyt.shape[-1]
    )
    in_patches = data_batch_xyt.view(-1, shape[-1])
    # we now have all neighborhoods (2nd dim) of spatial dims (1st dim)
    # we process these in batches, otherwise memory explodes
    batch_size = 500
    num_batches = int(np.ceil(indices.shape[0] / batch_size))
    # allocate
    t2s = torch.zeros(matrix_shape)
    t1s = torch.zeros(matrix_shape)
    l2_res = torch.zeros(matrix_shape)

    for ib in range(num_batches):
        start = ib * batch_size
        end = min((ib + 1) * batch_size, indices.shape[0])

        batch_indices = indices[start:end]

        # build patches via indexing
        patches = in_patches[batch_indices]
        # and format to the correct matrix shape
        # patches = patches.reshape((*matrix_shape, shape[-1]))
        # now we should have neighbrohood and temporal dims to the back, as needed for torch batch computations
        # thus we call an svd
        u, s, vh = torch.linalg.svd(patches, full_matrices=False)

        del patches
        torch.cuda.empty_cache()

        # and truncate the singular values for a low rank approximation
        s[..., rank:] = 0

        # reconstruct the patches
        lr_patches = u @ torch.diag_embed(s) @ vh
        del u, s, vh
        torch.cuda.empty_cache()

        # do the fitting pixel wise within the patch
        lr_patches = normalize_data(lr_patches, dim=-1)

        # for each patch, we want to map the sub-databases picked of each neighborhood point
        db_batch_matrix = in_db[:, :, batch_indices].view(-1, *batch_indices.shape, in_db.shape[-1])
        torch.cuda.empty_cache()

        # db_batch_matrix = db_batch_matrix.reshape(
        #     db_batch_matrix.shape[0],
        #     *matrix_shape,
        #     shape[-1]
        # )

        loss = torch.linalg.norm(lr_patches[None] - db_batch_matrix, dim=-1)
        # loss = torch.zeros(db_batch_matrix.shape[:-1])
        # for i in range(loss.shape[0]):
        #     tmp = lr_patches - db_batch_matrix[i]
        #     loss[i] = torch.linalg.norm(tmp, dim=-1).cpu()
        #     # del tmp
        #     torch.cuda.empty_cache()

        vals, min_indices = torch.min(loss, dim=0)
        del loss, db_batch_matrix
        torch.cuda.empty_cache()

        # get values
        min_indices = unflatten_batched_indices(min_indices, (t1_vals.shape[0], t2_vals.shape[0]))
        t1s[start:end] = t1_vals[min_indices[..., 0]].cpu()
        t2s[start:end] = t2_vals[min_indices][..., 1].cpu()
        l2_res[start:end] = vals.cpu()

    # allocate outputs
    weights = torch.zeros(in_patches[..., 0].shape, dtype=l2_res.dtype)
    batch_t2 = torch.zeros(in_patches[..., 0].shape, dtype=t2s.dtype)
    batch_res_l2 = torch.zeros(in_patches[..., 0].shape, dtype=l2_res.dtype)

    # weighted averaging
    weights = weights.index_add(0, indices.view(-1), (1 / l2_res).view(-1))
    batch_t2 = batch_t2.index_add(0, indices.view(-1), (t2s / l2_res).view(-1))
    batch_res_l2 = batch_res_l2.index_add(0, indices.view(-1), l2_res.view(-1))
    count_matrix = torch.bincount(indices.view(-1))

    # reshape to original
    batch_t2 = batch_t2.reshape(shape[:-1])
    weights = weights.reshape(shape[:-1])
    batch_t2 = torch.nan_to_num(batch_t2 / weights, nan=0.0, posinf=0.0)

    batch_res_l2 = batch_res_l2.reshape(shape[:-1])
    count_matrix = count_matrix.reshape(shape[:-1])
    count_matrix[count_matrix < 1e-3] = 1
    batch_res_l2 = batch_res_l2 / count_matrix

    return batch_t2, batch_res_l2


def fit_mese(
        data_xyzce: torch.Tensor, db_t1t2b1b0: torch.Tensor,
        t1_vals: torch.Tensor, t2_vals: torch.Tensor, b1_vals: torch.Tensor, b0_vals: torch.Tensor,
        path_out: plib.Path,
        b1_data: torch.Tensor = None, b0_data = None,
        b1_err_data: torch.Tensor = None, b1_err_thr: float = 0.15,
        device: torch.device = torch.get_default_device(),
        lr_reg: int = None,
        input_affine: torch.Tensor = torch.eye(4)):
    """
    MESE dictionary fit method.
    """

    logger.info(f"Fit MESE")
    # __ Some preparations
    # for now ensure magnitude data
    data_xyzce = data_xyzce.abs().to(dtype=torch.float32)
    # normalize data
    data_xyzce = normalize_data(data_xyzce, dim=-1)
    # save shape
    db_shape = db_t1t2b1b0.shape
    # for now ensure magnitude data
    db_t1t2b1b0 = db_t1t2b1b0.abs().to(dtype=torch.float32)
    # normalise database
    db_t1t2b1b0 = normalize_data(db_t1t2b1b0, dim=-1)
    # set up values
    # if b1_data is not None:
    #     b1_data *= 1.32

    # check for additional input if not available estimate for RSOS data
    # (assuming B1 transmit and B0 variation is approximately equal for all receive channels)
    b1_data, b1_est, b0_data, b0_est = estimate_fields_from_db(
        data_5d=data_xyzce, db_t1t2b1b0=db_t1t2b1b0,
        data_b1=b1_data, b1_vals=b1_vals,
        data_b0=b0_data, b0_vals=b0_vals,
        b1_err_data=b1_err_data, b1_err_thr=b1_err_thr,
        device=device, batch_size=1 # if b0_data is not None else 10
    )
    for i, d in enumerate([b1_data, b1_est, b0_data, b0_est]):
        nifti_save(
            d, img_aff=input_affine, path_to_dir=path_out,
            file_name=["b1_estimate_smoothed", "b1_estimate", "b0_estimate_smoothed", "b0_estimate"][i]
        )
    if data_xyzce.ndim - 1 > b1_data.ndim:
        # need to expand back to channels
        b1_data = b1_data.unsqueeze(-1).expand_as(data_xyzce[..., 0])

    if data_xyzce.ndim - 1 > b0_data.ndim:
        # need to expand back to channels
        b0_data = b0_data.unsqueeze(-1).expand_as(data_xyzce[..., 0])

    # now use this for input to to the regularised fit
    t2, l2_res = regularised_fit(
        data=data_xyzce, db_t1t2b1b0=db_t1t2b1b0, t1_vals=t1_vals, t2_vals=t2_vals,
        b1_data=b1_data, b1_vals=b1_vals, b0_data=b0_data, b0_vals=b0_vals,
        lr_reg=lr_reg, device=device
    )

    # __ Cleanup and Combination
    r2 = torch.nan_to_num(1 / t2, posinf=0.0, nan=0.0, neginf=0.0)
    t2 = 1e3 * t2

    logger.info(f"Combining channels")
    l2_res = l2_res**2
    l2_residual_upper_limit = l2_res.median() / 3
    weights = (l2_residual_upper_limit - torch.clamp(l2_res, 0, l2_residual_upper_limit)) / l2_residual_upper_limit
    weights = smooth_map(weights, kernel_size=5)

    # weighted averaging - would just give the same if used with rsos i.e. one channel
    r2_combined = torch.nan_to_num(
        torch.sum(weights * r2, dim=-1) / torch.sum(weights, dim=-1)
    )

    logger.info(f"Saving channel fits")
    # reshape & save
    names = ["optimize_residual", "r2", "t2", "weights", "r2_combined"]
    for i, r in enumerate([l2_res, r2, t2, weights, r2_combined]):
        nifti_save(
            r, img_aff=input_affine,
            path_to_dir=path_out, file_name=names[i]
        )


def main():
    # Setup CLI Program
    setup_program_logging(name="EMC Dictionary Grid Search", level=logging.INFO)
    # Setup parser
    parser, prog_args = setup_parser(
        prog_name="EMC Dictionary Grid Search for non - combined data",
        dict_config_dataclasses={"settings": EmcFitSettings}
    )
    # Get settings
    settings = EmcFitSettings.from_cli(args=prog_args.settings, parser=parser)
    settings.display()

    try:
        wrap_cli(settings=settings)
    except Exception as e:
        logging.exception(e)
        parser.print_help()
#
# def main():
#     path = plib.Path(
#         "/data/pt_np-jschmidt/data/30_projects/01_pulseq_mese_r2/01_data/1_invivo/2025-05-19/processing/mese_cfa/fit"
#     )
#     data = torch_load(
#            "/data/pt_np-jschmidt/data/30_projects/01_pulseq_mese_r2/01_data/1_invivo/"
#            "2025-05-19/raw/mese_cfa/k_space_rmos.pt"
#     )
#     data = data[:, :, data.shape[2] // 2, None].clone()
#     b1_data, _ = nifti_load(
#         "/data/pt_np-jschmidt/data/30_projects/01_pulseq_mese_r2/01_data/1_invivo/"
#         "2025-05-19/processing/mese_cfa/fit/b1_estimate_smoothed.nii"
#     )
#     b1_data = torch.from_numpy(b1_data)
#     affine = torch_load(
#            "/data/pt_np-jschmidt/data/30_projects/01_pulseq_mese_r2/01_data/1_invivo/"
#            "2025-05-19/raw/mese_cfa/affine.pt"
#     )
#     db = DB.load(
#         "/data/pt_np-jschmidt/data/30_projects/01_pulseq_mese_r2/02_emc/cfa/db_mese_cfa.pkl"
#     )
#     # get torch tensors
#     db_torch_mag, db_torch_phase = db.get_torch_tensors_t1t2b1b0e()
#     # get t2 and b1 values
#     t1_vals, t2_vals, b1_vals, b0_vals = db.get_t1_t2_b1_b0_values()
#
#     # fft reconned k-space
#     data = fft_to_img(data, dims=(0, 1))
#
#     # set database
#     db_pattern = db_torch_mag * torch.exp(1j * db_torch_phase)
#
#     fit_mese(
#         data_xyzce=data, db_t1t2b1b0=db_pattern, b1_data=b1_data,
#         t1_vals=t1_vals, t2_vals=t2_vals, b1_vals=b1_vals, b0_vals=b0_vals,
#         path_out=path, device=torch.device("cuda:0"),
#         input_affine=affine
#     )



if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s', datefmt='%I:%M:%S', level=logging.INFO
    )
    main()
