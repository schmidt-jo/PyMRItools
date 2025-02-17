import json
import logging
import pathlib as plib

import tqdm
import torch
import numpy as np

from pymritools.config.emc import EmcFitSettings
from pymritools.config.emc.settings import FitSettings
from pymritools.config.database import DB
from pymritools.utils import nifti_save, nifti_load, normalize_data
from pymritools.config import setup_program_logging, setup_parser

log_module = logging.getLogger(__name__)


def setup_input(settings: EmcFitSettings):
    """
        Loads, processes, and normalizes MRI input data from a nifti file, then
        prepares it for analysis by converting to a single batch dimension and
        identifying non-zero voxels for masking.

        Parameters:
        settings (EmcFitSettings): An object containing settings and configuration
        for loading and processing data, including file paths and output names.

        Returns:
        tuple: A tuple containing the following elements:
            - input_data (torch.Tensor): The normalized input data tensor.
            - input_img (nibabel.Nifti1Image): The loaded nifti image object.
            - data_shape (tuple): The original shape of the input data.
            - mask_nii_nonzero (torch.Tensor): A tensor indicating which voxels are non-zero.
            - rho_s (float): Calculated scaling factor related to the signal-to-noise ratio.
            - name (str): The output name derived from the settings.
    """
    # load data
    input_data, input_img = nifti_load(settings.input_data)
    name = settings.out_name
    # get to torch
    input_data = torch.from_numpy(input_data)
    # save shape
    data_shape = input_data.shape
    # normalize and get scaling factor (related to SNR)
    input_data, rho_s = normalize_data(input_data)
    # get into one batch dim: [b, etl]
    input_data = torch.reshape(input_data, (-1, data_shape[-1]))
    # check for masked voxels, dont need to process 0 voxels
    mask_nii_nonzero = torch.sum(torch.abs(input_data), dim=-1) > 1e-6
    return input_data, input_img, data_shape, mask_nii_nonzero, rho_s, name


def setup_db(settings: EmcFitSettings):
    """
        Setups the database based on the provided settings.

        Args:
            settings (EmcFitSettings): Configuration settings for database setup.

        Returns:
        tuple:
            - db: Loaded database object.
            - db_torch_mag: Normalized torch tensor containing magnitude values from the database.
            - rho_db: Flattened normalization parameter values.
            - t1t2b1_vals: Torch tensor of combined T1, T2, and B1 values.
            - t1_vals: List of T1 values.
            - t2_vals: List of T2 values.
            - b1_vals: List of B1 values.
    """
    # load database
    db = DB.load(settings.input_database)
    # get torch tensors
    db_torch_mag, db_torch_phase = db.get_torch_tensors_t1t2b1e()
    # normalize database, use magnitude only for now
    db_torch_mag, rho_db = normalize_data(db_torch_mag)
    # get t2 and b1 values
    t1_vals, t2_vals, b1_vals = db.get_t1_t2_b1_values()
    # cast to combined db dimension
    t1t2b1_vals = torch.tensor([(t1, t2, b1) for t1 in t1_vals for t2 in t2_vals for b1 in b1_vals])
    db_torch_mag = torch.reshape(db_torch_mag, (-1, db_torch_mag.shape[-1]))
    rho_db = rho_db.flatten()
    return db, db_torch_mag, rho_db, t1t2b1_vals, t1_vals, t2_vals, b1_vals


def setup_path(settings: EmcFitSettings):
    """
    Sets up and ensures the output path specified in the settings exists.

    Parameters
    ----------
    settings : EmcFitSettings
        The settings containing the output path to be set up.

    Returns
    -------
    Path
        The absolute output path after ensuring its existence.

    Notes
    -----
    - Logs the output path being set.
    - Creates the directory if it doesn't already exist.
    """
    # set path
    path_out = plib.Path(settings.out_path).absolute()
    log_module.info(f"set output path: {path_out}")
    if not path_out.exists():
        log_module.info(f"mkdir {path_out}".ljust(20))
        path_out.mkdir(exist_ok=True, parents=True)
    return path_out


def setup_b1(settings: EmcFitSettings):
    """
    setup_b1(settings: EmcFitSettings)

    Prepares B1 data for processing based on the input settings.

    Arguments:
        settings: An instance of EmcFitSettings containing the input B1 file path.

    Returns:
        A tuple containing:
            - A tensor of B1 data, scaled appropriately.
            - A name string for identification purposes.
    """
    # set shorthand for b1 processing
    if settings.input_b1:
        b1_data, b1_img = nifti_load(settings.input_b1)
        b1_data = torch.from_numpy(b1_data)
        # get all spatial voxels into one batch dim
        b1_data = torch.flatten(b1_data)
        # check for scaling (percentage or unitless)
        if torch.max(b1_data) > 10:
            b1_data = b1_data / 100
        name = f"b1-in_"
    else:
        b1_data = None
        name = ""
    return b1_data, name


def fit_r2_fn(
        input_data: torch.Tensor, db_mag: torch.Tensor,
        t2_vals: torch.Tensor, b1_vals: torch.Tensor, t1_vals: torch.Tensor = torch.tensor([1.5]),
        b1_data: torch.Tensor = None, batch_size: int = 1000, device: torch.device = torch.get_default_device()):
    # mask nonzero values
    data_shape = input_data.shape
    input_data = input_data.reshape((-1, data_shape[-1]))

    # can save some computations if we get bet data or data with 0 voxels
    mask_nii_nonzero = torch.sum(torch.abs(input_data), dim=-1) > 1e-8
    in_data_masked = input_data[mask_nii_nonzero]
    # get data size
    batch_dim_size = in_data_masked.shape[0]

    b1_in = True if b1_data is not None else False
    if b1_in:
        in_b1_masked = torch.reshape(b1_data, (-1,))[mask_nii_nonzero]
    else:
        in_b1_masked = None

    # set batch processing
    num_batches = int(np.ceil(batch_dim_size / batch_size))

    # get value combinations per curve
    t1t2b1_vals = torch.tensor([(t1, t2, b1) for t1 in t1_vals for t2 in t2_vals for b1 in b1_vals])

    # allocate
    t2 = torch.zeros(batch_dim_size, dtype=t2_vals.dtype, device=torch.device("cpu"))
    l2 = torch.zeros(batch_dim_size, dtype=t2_vals.dtype, device=torch.device("cpu"))
    b1 = torch.zeros(batch_dim_size, dtype=b1_vals.dtype, device=torch.device("cpu"))
    t2_unmasked = torch.zeros(input_data.shape[0], dtype=t2_vals.dtype, device=torch.device("cpu"))
    l2_unmasked = torch.zeros(input_data.shape[0], dtype=t2_vals.dtype, device=torch.device("cpu"))
    b1_unmasked = torch.zeros(input_data.shape[0], dtype=b1_vals.dtype, device=torch.device("cpu"))
    # rho_theta = torch.zeros(batch_dim_size, dtype=rho_s.dtype, device=device)

    # reshape db to [t1, t2, b1, etl]
    db_mag = torch.reshape(db_mag, (t1_vals.shape[0], t2_vals.shape[0], b1_vals.shape[0], -1))

    if not b1_in:
        # take whole database to gpu in case we arent regularizing with b1
        fit_db = db_mag.to(device)[None]
    else:
        fit_db = None
    # batch process
    for idx_b in tqdm.trange(num_batches, desc="Batch Processing"):
        start = idx_b * batch_size
        end = min(start + batch_size, batch_dim_size)
        data_batch = in_data_masked[start:end].to(device)

        # b1 penalty
        if b1_in:
            # if regularizing get b1 from closest matching to b1 input
            b1_batch = in_b1_masked[start:end]
            b1_penalty = torch.sqrt(torch.square(b1_vals[:, None] - b1_batch[None, :]))
            b1_min_idxs = torch.min(b1_penalty, dim=0).indices
            b1[start:end] = b1_vals[b1_min_idxs]
            # reduce db size by matching b1s, and restore batch dim
            fit_db = torch.movedim(db_mag[:, :, b1_min_idxs], 2, 0).to(device)
            # essentially left with [batch, t1, t2, etl], restore b1
            fit_db = fit_db[:, :, :, None]

        # l2 norm difference of magnitude data vs magnitude database
        # calculate difference, dims db [batch, t1s, t2s, b1s, t], nii-batch [batch,t]
        l2_norm_diff = torch.linalg.vector_norm(
            fit_db - data_batch[:, None, None, None, :], dim=-1)

        # find minimum l2 in db dim
        l2_flat = torch.reshape(l2_norm_diff, (end - start, -1))

        min_l2 = torch.min(l2_flat, dim=1)

        min_idx_l2 = min_l2.indices.to(torch.device("cpu"))
        min_vals_l2 = min_l2.values.to(torch.device("cpu"))

        # populate maps
        if not b1_in:
            t2[start:end] = t1t2b1_vals[min_idx_l2, 1]
            b1[start:end] = t1t2b1_vals[min_idx_l2, 2]
        else:
            t2[start:end] = t2_vals[min_idx_l2]
        l2[start:end] = min_vals_l2
        # rho_theta[start:end] = rho_db[min_idx_l2]

    # fill in the whole tensor
    t2_unmasked[mask_nii_nonzero] = t2.cpu()
    l2_unmasked[mask_nii_nonzero] = l2.cpu()
    b1_unmasked[mask_nii_nonzero] = b1.cpu()
    # rho_theta_unmasked[mask_nii_nonzero] = rho_theta.cpu()
    t2_unmasked = torch.reshape(t2_unmasked, data_shape[:-1])
    l2_unmasked = torch.reshape(l2_unmasked, data_shape[:-1])
    b1_unmasked = torch.reshape(b1_unmasked, data_shape[:-1])
    return t2_unmasked, b1_unmasked, l2_unmasked


def fit(settings: FitSettings):
    # set device
    if settings.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    log_module.info(f"Set torch device: {device}")

    # setup path
    path_out = setup_path(settings=settings)

    # setup input data
    input_data, input_img, data_shape, mask_nii_nonzero, rho_s, name = setup_input(settings=settings)
    total_data_size = input_data.shape[0]

    # setup database
    # ToDo: Database calculation is with a fixed set of spin isochromats. If we don't normalize those and
    #       insert a scaling parameter we would find a PD like modeling factor (still SNR dependent)
    db, db_mag, rho_db, t1t2b1_vals, t1_vals, t2_vals, b1_vals = setup_db(settings=settings)

    # setup b1
    b1_data, b1_name = setup_b1(settings=settings)
    b1_in = True if b1_data is not None else False

    # allocate space, no t1 fit for now, dont put on gpu,
    # since we only process the non zero data on gpu and fill in later
    t2_unmasked = torch.zeros(total_data_size, dtype=t1t2b1_vals.dtype, device=torch.device("cpu"))
    l2_unmasked = torch.zeros(total_data_size, dtype=t1t2b1_vals.dtype, device=torch.device("cpu"))
    b1_unmasked = torch.zeros(total_data_size, dtype=t1t2b1_vals.dtype, device=torch.device("cpu"))
    rho_theta_unmasked = torch.zeros(total_data_size, dtype=rho_s.dtype, device=torch.device("cpu"))

    # mask
    in_data_masked = input_data[mask_nii_nonzero]
    batch_dim_size = in_data_masked.shape[0]
    if b1_in:
        in_b1_masked = b1_data[mask_nii_nonzero]

    # set batch processing
    batch_size = settings.batch_size
    num_batches = int(np.ceil(batch_dim_size / batch_size))

    t2 = torch.zeros(batch_dim_size, dtype=t1t2b1_vals.dtype, device=device)
    l2 = torch.zeros(batch_dim_size, dtype=t1t2b1_vals.dtype, device=device)
    b1 = torch.zeros(batch_dim_size, dtype=b1_vals.dtype, device=device)
    rho_theta = torch.zeros(batch_dim_size, dtype=rho_s.dtype, device=device)

    # reshape db to [t1, t2, b1, etl]
    db_mag = torch.reshape(db_mag, (t1_vals.shape[0], t2_vals.shape[0], b1_vals.shape[0], -1))

    if not b1_in:
        # take whole database to gpu in case we arent regularizing with b1
        fit_db = db_mag.to(device)[None]
    # batch process
    for idx_b in tqdm.trange(num_batches, desc="Batch Processing"):
        start = idx_b * batch_size
        end = min(start + batch_size, batch_dim_size)
        data_batch = in_data_masked[start:end].to(device)

        # b1 penalty
        if b1_in:
            # if regularizing get b1 from closest matching to b1 input
            b1_batch = in_b1_masked[start:end]
            b1_penalty = torch.sqrt(torch.square(b1_vals[:, None] - b1_batch[None, :]))
            b1_min_idxs = torch.min(b1_penalty, dim=0).indices
            b1[start:end] = b1_vals[b1_min_idxs]
            # reduce db size by matching b1s, and restore batch dim
            fit_db = torch.movedim(db_mag[:, :, b1_min_idxs], 2, 0).to(device)
            # essentially left with [batch, t1, t2, etl], restore b1
            fit_db = fit_db[:, :, :, None]

        # l2 norm difference of magnitude data vs magnitude database
        # calculate difference, dims db [batch, t1s, t2s, b1s, t], nii-batch [batch,t]
        l2_norm_diff = torch.linalg.vector_norm(
            fit_db - data_batch[:, None, None, None, :], dim=-1)

        # find minimum l2 in db dim
        l2_flat = torch.reshape(l2_norm_diff, (end-start, -1))

        min_l2 = torch.min(l2_flat, dim=1)

        min_idx_l2 = min_l2.indices.to(torch.device("cpu"))
        min_vals_l2 = min_l2.values.to(torch.device("cpu"))

        # populate maps
        if not b1_in:
            t2[start:end] = t1t2b1_vals[min_idx_l2, 1]
            b1[start:end] = t1t2b1_vals[min_idx_l2, 2]
        else:
            t2[start:end] = t2_vals[min_idx_l2]
        l2[start:end] = min_vals_l2
        rho_theta[start:end] = rho_db[min_idx_l2]
    # fill in the whole tensor
    t2_unmasked[mask_nii_nonzero] = t2.cpu()
    l2_unmasked[mask_nii_nonzero] = l2.cpu()
    b1_unmasked[mask_nii_nonzero] = b1.cpu()
    rho_theta_unmasked[mask_nii_nonzero] = rho_theta.cpu()

    # reshape
    if torch.max(t2_unmasked) < 5:
        # cast to ms
        t2_unmasked = 1e3 * t2_unmasked
    t2_unmasked = torch.reshape(t2_unmasked, data_shape[:-1])
    t2_unmasked = t2_unmasked.numpy(force=True)
    l2_unmasked = torch.reshape(l2_unmasked, data_shape[:-1])
    l2_unmasked = l2_unmasked.numpy(force=True)
    b1_unmasked = torch.reshape(b1_unmasked, data_shape[:-1]).numpy(force=True)
    rho_theta_unmasked = torch.reshape(rho_theta_unmasked, data_shape[:-1]).cpu()
    r2 = np.divide(1e3, t2_unmasked, where=t2_unmasked > 1e-4, out=np.zeros_like(t2_unmasked))
    pd = torch.nan_to_num(
        torch.divide(
            torch.squeeze(rho_s),
            torch.squeeze(rho_theta_unmasked)
        ),
        nan=0.0, posinf=0.0
    )
    # we want to calculate histograms for both, and find upper cutoffs of the data values based on the histograms
    # since both might explode
    # pd_hist, pd_bins = torch.histogram(pd.flatten(), bins=200)
    # # find percentage where 95 % of data lie
    # pd_hist_perc = torch.cumsum(pd_hist, dim=0) / torch.sum(pd_hist, dim=0)
    # pd_cutoff_value = pd_bins[torch.nonzero(pd_hist_perc > 0.95)[0].item()]
    # pd = torch.clamp(pd, min=0.0, max=pd_cutoff_value).numpy(force=True)

    # save data
    nifti_save(data=r2, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}r2")
    nifti_save(data=t2_unmasked, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}t2")
    nifti_save(data=b1_unmasked, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}b1")
    nifti_save(data=l2_unmasked, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}l2-err")
    nifti_save(data=rho_s, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}rho-s")
    nifti_save(data=rho_theta_unmasked, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}rho-theta")
    nifti_save(data=pd, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}pd-approx")


def megesse(settings: EmcFitSettings):
    # set device
    if settings.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    log_module.info(f"Set torch device: {device}")

    # setup path
    path_out = setup_path(settings=settings)

    # setup input data
    input_data, input_img, data_shape, mask_nii_nonzero, rho_s, name = setup_input(settings=settings)
    total_data_size = input_data.shape[0]

    # setup database
    db, db_mag, rho_db, t1t2b1_vals, t1_vals, t2_vals, b1_vals = setup_db(settings=settings)

    # setup b1
    b1_data, b1_name = setup_b1(settings=settings)
    if b1_data is None:
        err = f"No B1 input provided."
        log_module.error(err)
        raise ValueError(err)

    # setup tes and se and gre indices
    with open(
            "/data/pt_np-jschmidt/data/03_sequence_dev/build_sequences/2024-11-11_mese_acc_megesse_sym_in-vivo/"
            "megesse_acc-3/megesse_v1p0_acc-3p0_res-0p70-0p70-0p70_te.json", "r") as j_file:
        tes = json.load(j_file)[2:]
    se_idxs = torch.arange(2, input_data.shape[-1], 5)
    gre_idxs = torch.tensor([k for k in range(input_data.shape[-1]) if k not in se_idxs])
    # build gradient echo sampling differences to se
    gre_times = tes[:2]
    for idx_se in range(4):
        for idx_gre in range(4):
            te_gre = tes[gre_idxs[2 + idx_gre + 4 * idx_se]]
            te_se = tes[se_idxs[idx_se]]
            gre_times.append(np.abs(te_gre - te_se))
    gre_times = torch.tensor(gre_times)
    # prepare for fit
    a = torch.ones((gre_times.shape[0], 2), dtype=torch.float64)
    a[:, 1] = gre_times
    a = a.to(device)
    ata = torch.matmul(a.T, a)
    ata_inv = torch.linalg.inv(ata)

    # allocate space, no t1 fit for now, dont put on gpu,
    # since we only process the non zero data on gpu and fill in later
    t2_unmasked = torch.zeros(total_data_size, dtype=t1t2b1_vals.dtype, device=torch.device("cpu"))
    r2p_unmasked = torch.zeros(total_data_size, dtype=t1t2b1_vals.dtype, device=torch.device("cpu"))
    l2_unmasked = torch.zeros(total_data_size, dtype=t1t2b1_vals.dtype, device=torch.device("cpu"))
    b1_unmasked = torch.zeros(total_data_size, dtype=t1t2b1_vals.dtype, device=torch.device("cpu"))
    rho_theta_unmasked = torch.zeros(total_data_size, dtype=rho_s.dtype, device=torch.device("cpu"))

    # mask
    in_data_masked = input_data[mask_nii_nonzero]
    batch_dim_size = in_data_masked.shape[0]
    in_b1_masked = b1_data[mask_nii_nonzero]

    # set batch processing
    batch_size = settings.batch_size
    num_batches = int(np.ceil(batch_dim_size / batch_size))

    # allocate masked batched output
    t2 = torch.zeros(batch_dim_size, dtype=t1t2b1_vals.dtype, device=device)
    r2p = torch.zeros(batch_dim_size, dtype=t1t2b1_vals.dtype, device=device)
    l2 = torch.zeros(batch_dim_size, dtype=t1t2b1_vals.dtype, device=device)
    b1 = torch.zeros(batch_dim_size, dtype=b1_vals.dtype, device=device)
    rho_theta = torch.zeros(batch_dim_size, dtype=rho_s.dtype, device=device)

    # calculate
    # reshape the database to b1 and t2 dimensions
    db_mag = torch.reshape(db_mag, (t2_vals.shape[0], b1_vals.shape[0], -1)).to(device)
    # take sub database with spin echo only datapoints
    db_se = db_mag[..., se_idxs]
    # normalize sub-database
    db_se, db_se_norm = normalize_data(db_se)
    # batch process
    for idx_b in tqdm.trange(num_batches, desc="Batch Processing"):
        start = idx_b * batch_size
        end = min(start + batch_size, batch_dim_size)
        # take batch of data
        data_batch = in_data_masked[start:end].to(device)
        # take SE only data
        data_se = data_batch[:, se_idxs]
        # normalize
        data_se, data_se_norm = normalize_data(data_se)

        # reduce database size from knowledge of b1
        # take batch of b1 input
        b1_batch = in_b1_masked[start:end]
        # calculate b1 values of database closest matching input, per voxel
        b1_penalty = torch.sqrt(torch.square(b1_vals[:, None] - b1_batch[None, :]))
        b1_min_idxs = torch.min(b1_penalty, dim=0).indices
        # assign those to the b1 output
        b1[start:end] = b1_vals[b1_min_idxs]
        # assign fit database to be used, this is the database for SE only echoes with the closest matching b1 indexes
        fit_db = torch.movedim(db_se[:, b1_min_idxs], 1, 0)

        # l2 norm difference of magnitude data vs magnitude database only SE data
        # calculate difference, dims db [batch-dim, t2s, t], nii-batch [batch-dim,t]
        l2_norm_diff = torch.linalg.vector_norm(
            fit_db - data_se[:, None, :], dim=-1)

        # find minimum index in db dim
        min_l2 = torch.min(l2_norm_diff, dim=1)

        # fit se data
        min_idx_l2 = min_l2.indices.to(torch.device("cpu"))
        min_vals_l2 = min_l2.values.to(torch.device("cpu"))

        # populate maps
        t2[start:end] = t2_vals[min_idx_l2]
        l2[start:end] = min_vals_l2
        rho_theta[start:end] = rho_db[min_idx_l2]

        # now we want to fit the gre data,
        # for this we want to take the whole signal and the whole found signal db curve and calculate r2p from
        # the attenuation on gre samples,
        # for this we want to normalize the data and the se database by the same norm
        # we used for the SE only data fitting, i.e. align SE signal samples
        s_t = torch.divide(data_batch, data_se_norm[:, None])
        db_t = torch.divide(db_mag[min_idx_l2, b1_min_idxs], db_se_norm[min_idx_l2, b1_min_idxs, None])
        y = torch.log(s_t[..., gre_idxs] / db_t[..., gre_idxs])
        # solve
        beta = torch.einsum(
            "jk, ij -> ik",
            ata_inv,
            torch.einsum("jk, ik -> ij", a.T, y),
        )
        r2p[start:end] = - beta[:, 1]

    # fill in the whole tensor
    t2_unmasked[mask_nii_nonzero] = t2.cpu()
    r2p_unmasked[mask_nii_nonzero] = r2p.cpu()
    l2_unmasked[mask_nii_nonzero] = l2.cpu()
    b1_unmasked[mask_nii_nonzero] = b1.cpu()
    rho_theta_unmasked[mask_nii_nonzero] = rho_theta.cpu()

    # reshape
    if torch.max(t2_unmasked) < 5:
        # cast to ms
        t2_unmasked = 1e3 * t2_unmasked
    t2_unmasked = torch.reshape(t2_unmasked, data_shape[:-1])
    t2_unmasked = t2_unmasked.numpy(force=True)
    r2p_unmasked = torch.reshape(r2p_unmasked, data_shape[:-1])
    r2p_unmasked = r2p_unmasked.numpy(force=True)
    l2_unmasked = torch.reshape(l2_unmasked, data_shape[:-1])
    l2_unmasked = l2_unmasked.numpy(force=True)
    b1_unmasked = torch.reshape(b1_unmasked, data_shape[:-1]).numpy(force=True)
    rho_theta_unmasked = torch.reshape(rho_theta_unmasked, data_shape[:-1]).cpu()
    r2 = np.divide(1e3, t2_unmasked, where=t2_unmasked > 1e-3, out=np.zeros_like(t2_unmasked))
    r2star = r2 + r2p_unmasked
    pd = torch.nan_to_num(
        torch.divide(
            torch.squeeze(rho_s),
            torch.squeeze(rho_theta_unmasked)
        ),
        nan=0.0, posinf=0.0
    )
    # we want to calculate histograms for both, and find upper cutoffs of the data values based on the histograms
    # since both might explode
    pd_hist, pd_bins = torch.histogram(pd.flatten(), bins=200)
    # find percentage where 95 % of data lie
    pd_hist_perc = torch.cumsum(pd_hist, dim=0) / torch.sum(pd_hist, dim=0)
    pd_cutoff_value = pd_bins[torch.nonzero(pd_hist_perc > 0.95)[0].item()]
    pd = torch.clamp(pd, min=0.0, max=pd_cutoff_value).numpy(force=True)

    # save data
    nifti_save(data=r2, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}r2")
    nifti_save(data=r2p_unmasked, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}r2p")
    nifti_save(data=r2star, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}r2s")
    nifti_save(data=t2_unmasked, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}t2")
    nifti_save(data=b1_unmasked, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}b1")
    nifti_save(data=l2_unmasked, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}l2-err")
    nifti_save(data=rho_s, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}rho-s")
    nifti_save(data=rho_theta_unmasked, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}rho-theta")
    nifti_save(data=pd, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}pd-approx")


def main():
    # Setup CLI Program
    setup_program_logging(name="EMC Dictionary Grid Search", level=logging.INFO)
    # Setup parser
    parser, prog_args = setup_parser(
        prog_name="EMC Dictionary Grid Search",
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


def grid_combined_r2_r2s_search():
    # Setup CLI Program
    setup_program_logging(name="EMC Dictionary Combined R2 / R2* Search", level=logging.INFO)
    # Setup parser
    parser, prog_args = setup_parser(
        prog_name="EMC Dictionary Combined R2 / R2* Search",
        dict_config_dataclasses={"settings": EmcFitSettings}
    )
    # Get settings
    settings = EmcFitSettings.from_cli(args=prog_args.settings, parser=parser)
    settings.display()

    try:
        megesse(settings=settings)
    except Exception as e:
        logging.exception(e)
        parser.print_help()


if __name__ == '__main__':
    main()
