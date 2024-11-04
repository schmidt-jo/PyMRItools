import logging
import pathlib as plib

import torch

from pymritools.config.emc import EmcFitSettings
from pymritools.config.database import DB
from pymritools.utils import normalize_data, nifti_load

log_module = logging.getLogger(__name__)


def load_data(settings: EmcFitSettings):
    # set path
    path_out = plib.Path(settings.out_path).absolute()
    log_module.info(f"set output path: {path_out}")
    if not path_out.exists():
        log_module.info(f"mkdir {path_out}".ljust(20))
        path_out.mkdir(exist_ok=True, parents=True)
    # set shorthand for b1 processing
    b1_in = False
    b1_data = None
    if settings.input_b1:
        b1_in = True

    # set device
    if settings.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    log_module.info(f"Set torch device: {device}")

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

    # load database
    db = DB.load(settings.input_database)
    # get torch tensors
    db_torch_mag, db_torch_phase = db.get_torch_tensors_t1t2b1e()
    # normalize database, use magnitude only for now
    db_torch_mag, rho_db = normalize_data(db_torch_mag)
    # get t2 and b1 values
    t1_vals, t2_vals, b1_vals = db.get_t1_t2_b1_values()
    # cast to combined db dimension
    t1t2b1_vals = torch.tensor([(t1, t2, b1) for t1 in t1_vals for t2 in t2_vals  for b1 in b1_vals])
    db_torch_mag = torch.reshape(db_torch_mag, (-1, db_torch_mag.shape[-1]))
    rho_db = rho_db.flatten()

    # load B1 if given
    if b1_in:
        b1_data, b1_img = nifti_load(settings.input_b1)
        b1_data = torch.from_numpy(b1_data)
        # get all spatial voxels into one batch dim
        b1_data = torch.flatten(b1_data)
        # check for scaling (percentage or unitless)
        if torch.max(b1_data) > 10:
            b1_data = b1_data / 100
        if name:
            name = f"{name}_".replace(".", "p")
        name = f"{name}b1-in_"

    return input_data, mask_nii_nonzero, rho_s, db_torch_mag, rho_db, t1t2b1_vals, b1_data, name

