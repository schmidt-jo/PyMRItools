import nibabel as nib
import pathlib as plib
import logging

import numpy as np
import torch

log_module = logging.getLogger(__name__)


def nifti_load(path_to_file: str | plib.Path) -> (np.ndarray, nib.Nifti1Image):
    """
        Loads a NIfTI file from the specified path.

        Args:
            path_to_file (str | plib.Path): The path to the NIfTI file. This can be a string or a pathlib.Path object.

        Returns:
            tuple: A tuple containing:
                - data (numpy.ndarray): The image data loaded from the NIfTI file.
                - img (nibabel.Nifti1Image): The NIfTI image object.

        Raises:
            FileNotFoundError: If the specified file is not found.
            AttributeError: If the specified file is not a NIfTI file
    """
    path_to_file = plib.Path(path_to_file).absolute()
    if not path_to_file.is_file():
        err = f"could not find file: {path_to_file}"
        log_module.error(err)
        raise FileNotFoundError(err)
    if not ".nii" in path_to_file.suffixes:
        err = f"file {path_to_file} not found to be a nifti file."
        log_module.error(err)
        raise AttributeError(err)
    img = nib.load(path_to_file.as_posix())
    data = img.get_fdata()
    return data, img


def nifti_save(
        data: np.ndarray | torch.Tensor, img_aff: nib.Nifti1Image | np.ndarray | torch.Tensor,
        path_to_dir: str | plib.Path, file_name: str):
    """
    Saves a NIfTI file to the specified directory with the given file name.

    Args:
        data (numpy.ndarray | torch.Tensor): The image data to be saved in the NIfTI format.
        img_aff (nibabel.Nifti1Image or np.ndarray | torch.Tensor): The NIfTI image object or affine matrix.
        path_to_dir (str | plib.Path): The directory where the NIfTI file will be saved. This can be a string or a pathlib.Path object.
                                       The directory will be created if it does not exist.
        file_name (str): The name of the file to be saved.

    Returns:
        None

    Raises:
        ValueError: If the provided data is not a numpy.ndarray or if the file_name is not a valid string.

    Logs:
        Creates an info log entry stating the path to the saved NIfTI file.
    """
    path_to_dir = plib.Path(path_to_dir).absolute()
    path_to_dir.mkdir(parents=True, exist_ok=True)
    file_path = path_to_dir.joinpath(file_name).with_suffix(".nii")
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    if torch.is_tensor(img_aff):
        img_aff = img_aff.cpu().numpy()
    if isinstance(img_aff, np.ndarray):
        img_aff = nib.Nifti1Image(data, affine=img_aff)
    else:
        img_aff = nib.Nifti1Image(data, affine=img_aff.affine)

    log_module.info(f"Write file : {file_path}")
    nib.save(img_aff, file_path.as_posix())


def setup_program_logging(name: str, level: int = logging.INFO):
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=level)
    # run some exclusions we dont want to expose to the user log
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("simple_parsing").setLevel(logging.WARNING)

    num_chars_name = len(name)
    num_chars_fill = 50
    logging.info("".ljust(num_chars_fill, "_"))
    logging.info(name.ljust(int((num_chars_fill + num_chars_name)/ 2), "_").rjust(num_chars_fill, "_"))
    logging.info("".ljust(num_chars_fill, "_"))
