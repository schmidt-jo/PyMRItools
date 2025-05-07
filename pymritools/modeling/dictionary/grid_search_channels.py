import logging
import pathlib as plib

import tqdm
import numpy as np
import torch
import json

from pymritools.modeling.dictionary.setup import setup_db, setup_b1, setup_path, setup_input, setup_b0
from pymritools.config import setup_program_logging, setup_parser
from pymritools.config.emc import EmcFitSettings
from pymritools.config.database import DB
from pymritools.utils import nifti_save, fft, ifft, root_sum_of_squares, torch_load, nifti_load

log_module = logging.getLogger(__name__)


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
    data_fft = fft(data, dims=(0, 1))
    # convolve
    data_fft *= kernel
    # fft back
    result = ifft(data_fft, dims=(0, 1))
    if not torch.is_complex(data):
        result = torch.abs(result)
        result = torch.clamp(result, min=data.min(), max=data.max())
    return result


def fit_megesse():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    log_module.info(f"set device: {device}")
    log_module.info(f"Load data")
    data = torch_load(
        "/data/pt_np-jschmidt/data/00_phantom_scan_data/2025-03-06_megesse_tests/raw/megesse_cesp_axial/gnc/"
        "img_gnc_cplx_slice.pt"
    )
    # data = torch.squeeze(data)
    affine = torch_load(
        "/data/pt_np-jschmidt/data/00_phantom_scan_data/2025-03-06_megesse_tests/raw/megesse_cesp_axial/affine.pt"
    )
    log_module.info(f"load database")
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

    log_module.info("Get Params")
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

    log_module.info("normalize data")
    # data_rsos = root_sum_of_squares(input_data=data, dim_channel=-2)
    data_rsos = data

    log_module.info("prep database")

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

    log_module.info("Allocate data")

    t2 = torch.zeros(data_normed_se.shape[:-1])
    r2p = torch.zeros(data_normed_se.shape[:-1])

    b1 = torch.zeros(data_normed_se.shape[:-1])
    b0 = torch.zeros(data_normed_se.shape[:-1])

    loss_residual = torch.zeros(data_normed_se.shape[:-1])
    phase_offset = torch.zeros(data_normed_se.shape[:-1])

    log_module.info(f"rough estimate R2*")
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

    log_module.info(f"rough estimate R2+")
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

    log_module.info(f"rough estimate R2 & R2'")
    rough_r2 = 0.5 * (r2s + r2dag)
    rough_r2p = 0.5 * (r2s - r2dag)
    nifti_save(rough_r2, img_aff=affine, path_to_dir=path, file_name="r2_rough")
    nifti_save(rough_r2p, img_aff=affine, path_to_dir=path, file_name="r2p_rough")

    log_module.info(f"Estimate rough B0 / B1")
    for idx_z in range(data_normed_se.shape[2]):
        log_module.info(f"Process slice: {idx_z + 1} / {data_normed_se.shape[2]}")
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

    log_module.info("B1 smoothing")
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

    log_module.info("prep data")
    data_norm = torch.linalg.norm(data_rsos, dim=-1, keepdim=True)
    data_normed = torch.nan_to_num(data_rsos / data_norm)
    # data_normed = torch.abs(data_normed).unsqueeze(-2)
    data_normed = data_normed.unsqueeze(-2)
    b1_map = b1_map.unsqueeze(-1)
    # sample r2p
    r2p_att = torch.exp(-gre_attenuation_times[None] * r2p_vals[:, None]).to(device)
    log_module.info("sample r2p db")
    db_r2p = torch.zeros((*db_shape[:-1], r2p_att.shape[0], db_shape[-1]), dtype=db_pattern.dtype)

    for idx_t2 in tqdm.trange(t2_vals.shape[0]):
        # for each t2 value we sample the database, this batching is only for computational and memory efficiency
        db_batch = db_pattern[:, idx_t2].to(device)
        db_batch = db_batch.unsqueeze(-2) * r2p_att[None, None, None]
        db_batch = torch.nan_to_num(db_batch / torch.linalg.norm(db_batch, dim=-1, keepdim=True))
        db_r2p[:, idx_t2] = db_batch.cpu()
    # now has dims [t1, t2, ny, b0, t]
    # r2p samples have dims [r2p, t] - > want [t1, t2, ny, b0, r2p, t]

    log_module.info("B1 Regularized matching")
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

    # log_module.info("B0 smoothing")
    # b0_map = smooth_map(b0, kernel_size=min(b1.shape[:2]) // 32)
    # nifti_save(b0, img_aff=affine, path_to_dir=path, file_name="se_b0_estimate")
    # nifti_save(b0_map, img_aff=affine, path_to_dir=path, file_name="se_b0_smoothed")
    #
    # log_module.info("B0 Regularized matching")
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

    log_module.info(f"Combining channels")
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

    log_module.info(f"Saving channel fits")
    # reshape & save
    names = ["ch_optimize_residual", "ch_r2", "ch_t2", "weights", "r2", "r2p", "r2s", "b0", "b1", "phase_offset"]
    for i, r in enumerate([loss_residual, r2, t2, weights, r2_combined, r2p_combined, r2s, b0, b1, phase_offset]):
        nifti_save(
            r, img_aff=affine,
            path_to_dir=path, file_name=names[i]
        )


def fit_revisited():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    log_module.info(f"set device: {device}")
    log_module.info(f"Load data")
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
    log_module.info(f"load database")
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

    log_module.info(f"l2 fit - b1 estimate")
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

    log_module.info("B1 smoothing")
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

    log_module.info("Regularized fit")
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

    log_module.info(f"Combining channels")
    weights = (l2_residual_upper_limit - torch.clamp(l2_res, 0, l2_residual_upper_limit)) / l2_residual_upper_limit
    weights = smooth_map(weights, kernel_size=5)

    # weighted averaging
    r2_combined = torch.nan_to_num(
        torch.sum(weights * r2, dim=-1) / torch.sum(weights, dim=-1)
    )

    log_module.info(f"Saving channel fits")
    # reshape & save
    names = ["ch_optimize_residual", "ch_r2", "ch_t2", "weights", "r2"]
    for i, r in enumerate([l2_res, r2, t2, weights, r2_combined]):
        nifti_save(
            r, img_aff=affine,
            path_to_dir=path, file_name=names[i]
        )


def fit_mese():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    log_module.info(f"set device: {device}")
    log_module.info(f"Load data")
    data = torch_load(
        "/data/pt_np-jschmidt/data/30_projects/01_pulseq_mese_r2/01_data/0_phantom_data/2025-05-06/processing/"
        "mese_vfa-a3p5ws/recon/ac_loraks_bcs_recon_acc-3p50_r-300.pt"
    )
    # data = data[:, :, data.shape[2] // 2, None]
    affine = torch_load(
        "/data/pt_np-jschmidt/data/30_projects/01_pulseq_mese_r2/01_data/0_phantom_data/2025-05-06/raw/"
        "mese_vfa-a3p5ws/affine.pt"
    )
    log_module.info(f"load database")
    db = DB.load(
        "/data/pt_np-jschmidt/data/30_projects/01_pulseq_mese_r2/02_emc/vfa/db_mese_vfa.pkl"
    )
    path = plib.Path(
        "/data/pt_np-jschmidt/data/30_projects/01_pulseq_mese_r2/01_data/0_phantom_data/2025-05-06/"
        "estimation/mese_vfa_a3p5ws/loraks_b1_in"
    )

    # get torch tensors
    db_torch_mag, db_torch_phase = db.get_torch_tensors_t1t2b1b0e()
    # get t2 and b1 values
    t1_vals, t2_vals, b1_vals, b0_vals = db.get_t1_t2_b1_b0_values()

    # fft reconned k-space
    data = fft(data, dims=(0, 1))
    # normalize data
    data_rsos = root_sum_of_squares(input_data=data, dim_channel=-2)
    data_norm_rsos = torch.linalg.norm(data_rsos, dim=-1, keepdim=True)
    data_rsos_normed = torch.nan_to_num(data_rsos / data_norm_rsos)

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

    t2 = torch.zeros(data_normed.shape[:-1])
    b1 = torch.zeros(data_normed.shape[:-1])
    l2_res = torch.zeros(data_normed.shape[:-1])

    t2_rsos = torch.zeros(data_rsos_normed.shape[:-1])
    b1_rsos = torch.zeros(data_rsos_normed.shape[:-1])
    l2_res_rsos = torch.zeros(data_rsos_normed.shape[:-1])

    log_module.info(f"l2 fit - b1 estimate")
    batch_size = 10
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

    log_module.info(f"l2 fit - b1 estimate - rsos")
    for idx_z in range(data_rsos_normed.shape[2]):
        for idx_x in range(num_batches):
            start = idx_x * batch_size
            end = min((idx_x + 1) * batch_size, data_rsos_normed.shape[0])
            data_batch = data_rsos_normed[start:end, :, idx_z, :].to(device)

            l2 = torch.linalg.norm(data_batch[:, None] - db_pattern_normed[None, :, None], dim=-1)
            vals, indices = torch.min(l2, dim=1)
            batch_t1t2b1b0_vals = t1t2b1b0_vals[indices]

            b1_rsos[start:end, :, idx_z] = batch_t1t2b1b0_vals[..., 2].cpu()
            l2_res_rsos[start:end, :, idx_z] = vals

    log_module.info("B1 smoothing")
    b1_map = smooth_map(b1, kernel_size=min(b1.shape[:2]) // 32)
    b1_map_rsos = smooth_map(b1_rsos, kernel_size=min(b1_rsos.shape[:2]) // 32)

    nifti_save(b1, img_aff=affine, path_to_dir=path, file_name="reg_b1_estimate")
    nifti_save(b1_map, img_aff=affine, path_to_dir=path, file_name="reg_b1_smoothed")
    nifti_save(b1_rsos, img_aff=affine, path_to_dir=path, file_name="rsos_reg_b1_estimate")
    nifti_save(b1_map_rsos, img_aff=affine, path_to_dir=path, file_name="rsos_reg_b1_smoothed")

    b1_map = torch.from_numpy(nifti_load(
        "/data/pt_np-jschmidt/data/30_projects/01_pulseq_mese_r2/01_data/0_phantom_data/2025-05-06/"
        "estimation/mese_cfa/rsos_reg_b1_smoothed.nii"
    )[0]).unsqueeze(-1).expand(b1.shape)
    log_module.info("Regularized fit")
    t1t2b0_vals = torch.tensor([
        [t1, t2, b0] for t1 in t1_vals for t2 in t2_vals for b0 in b0_vals
    ], device=device)
    # prepare the database for b1 regularization
    db_pattern_normed = torch.reshape(db_pattern_normed, db_shape)
    db_pattern_normed = torch.movedim(db_pattern_normed, -2, 2)
    db_pattern_normed = torch.reshape(db_pattern_normed, (-1, *db_pattern_normed.shape[-2:]))
    l2_res = torch.zeros(data_normed.shape[:-1])

    # do slice wise processing
    for idx_z in tqdm.trange(data_normed.shape[2], desc="Processing, slice wise with b1 reg."):
        for idx_c in range(data_normed.shape[3]):
            data_batch = data_normed[:, :, idx_z, idx_c].to(device)
            b1_batch = b1_map[:, :, idx_z, idx_c]
            # want the database to be pulled towards the b1 regularization, b1-batch [nx, ny]
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

    for idx_z in tqdm.trange(data_rsos_normed.shape[2], desc="Processing, slice wise with b1 reg."):
        data_batch = data_rsos_normed[:, :, idx_z].to(device)
        b1_batch = b1_map_rsos[:, :, idx_z]
        # want the database to be pulled towards the b1 regularization, b1-batch [nx, ny]
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

        l2_res_rsos[:, :, idx_z] = vals
        t2_rsos[:, :, idx_z] = batch_t1t2b0[..., 1].cpu()

    r2 = torch.nan_to_num(1 / t2)
    t2 = 1e3 * t2
    r2_rsos = torch.nan_to_num(1 / t2_rsos)
    t2_rsos = 1e3 * t2_rsos

    log_module.info(f"Combining channels")
    l2_residual_upper_limit = 0.15
    weights = (l2_residual_upper_limit - torch.clamp(l2_res, 0, l2_residual_upper_limit)) / l2_residual_upper_limit
    weights = smooth_map(weights, kernel_size=5)

    # weighted averaging
    r2_combined = torch.nan_to_num(
        torch.sum(weights * r2, dim=-1) / torch.sum(weights, dim=-1)
    )

    log_module.info(f"Saving channel fits")
    # reshape & save
    names = ["ch_optimize_residual", "ch_r2", "ch_t2", "weights", "r2", "r2_rsos", "t2_rsos"]
    for i, r in enumerate([l2_res, r2, t2, weights, r2_combined, r2_rsos, t2_rsos]):
        nifti_save(
            r, img_aff=affine,
            path_to_dir=path, file_name=names[i]
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
        fit(settings=settings)
    except Exception as e:
        logging.exception(e)
        parser.print_help()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    fit_mese()
