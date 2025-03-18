
import pathlib as plib
import logging

import torch
import nibabel as nib
from pymritools.config.database import DB
from pymritools.config.emc import EmcFitSettings
from pymritools.utils import nifti_save, nifti_load, normalize_data, torch_load

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
    path_in = plib.Path(settings.input_data)
    if not path_in.is_file():
        raise FileNotFoundError(path_in)
    if ".nii" in path_in.suffixes:
        input_data, input_img = nifti_load(settings.input_data)
        input_data = torch.from_numpy(input_data)
    elif ".pt" in path_in.suffixes:
        input_data = torch_load(settings.input_data)
        path_affine = plib.Path(settings.input_affine)
        if not path_affine.is_file():
            aff = torch.eye(4)
        else:
            aff = torch_load(path_affine)
        input_img = nib.Nifti1Image(input_data.numpy(), aff.numpy())
    else:
        err = f"File suffix '{path_in.suffixes}' is not supported. Chose .nii or .pt files"
        raise AttributeError(err)

    name = settings.out_name
    # get to torch
    # save shape
    data_shape = input_data.shape
    # normalize and get scaling factor (related to SNR)
    input_data, rho_s = normalize_data(input_data)
    # get into one batch dim: [b, etl]
    input_data = torch.reshape(input_data, (-1, data_shape[-1]))
    # check for masked voxels, dont need to process 0 voxels
    mask_nii_nonzero = torch.sum(torch.abs(input_data), dim=-1) > 1e-6
    return input_data, input_img, data_shape, mask_nii_nonzero, rho_s, name


def setup_db(settings: EmcFitSettings, return_complex: bool = False):
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
    db_torch_mag, db_torch_phase = db.get_torch_tensors_t1t2b1b0e()
    # get t2 and b1 values
    t1_vals, t2_vals, b1_vals, b0_vals = db.get_t1_t2_b1_b0_values()
    # cast to combined db dimension
    t1t2b1b0_vals = torch.tensor([
        (t1, t2, b1, b0) for t1 in t1_vals for t2 in t2_vals for b1 in b1_vals for b0 in b0_vals
    ])
    if return_complex:
        db_pattern = db_torch_mag + 1j * torch.exp(db_torch_phase)
    else:
        db_pattern = db_torch_mag
    # normalize database, use magnitude only for now
    db_pattern, rho_db = normalize_data(db_pattern)
    db_pattern = torch.reshape(db_pattern, (-1, db_pattern.shape[-1]))
    rho_db = rho_db.flatten()
    return db, db_pattern, rho_db, t1t2b1b0_vals, t1_vals, t2_vals, b1_vals, b0_vals


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


def setup_b0(settings: EmcFitSettings):
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
    if settings.input_b0:
        b0_data, b0_img = nifti_load(settings.input_b0)
        b0_data = torch.from_numpy(b0_data)
        # get all spatial voxels into one batch dim
        b0_data = torch.flatten(b0_data)
        name = f"b0-in_"
    else:
        b0_data = None
        name = ""
    return b0_data, name
