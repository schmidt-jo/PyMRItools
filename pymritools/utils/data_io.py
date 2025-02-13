import sys
import os
import nibabel as nib
import pathlib as plib
import logging
import numpy as np
import torch

log_module = logging.getLogger(__name__)


def set_save_path(path_to_dir: str | plib.Path, file_name: str, suffix: str):
    path_to_dir = plib.Path(path_to_dir).absolute()
    path_to_dir.mkdir(parents=True, exist_ok=True)
    file_path = path_to_dir.joinpath(file_name).with_suffix(suffix)
    log_module.info(f"Write file : {file_path}")
    return file_path


def set_load_path(path_to_file: str | plib.Path, suffix: str):
    path_to_file = plib.Path(path_to_file).absolute()
    if not path_to_file.is_file():
        err = f"could not find file: {path_to_file}"
        log_module.error(err)
        raise FileNotFoundError(err)
    if not suffix in path_to_file.suffixes:
        err = f"file {path_to_file} not found to be a {suffix} file."
        log_module.error(err)
        raise AttributeError(err)
    log_module.info(f"Load file: {path_to_file}")
    return path_to_file


def numpy_save(
        data: np.ndarray, path_to_file: str | plib.Path, file_name: str):
    if not isinstance(data, np.ndarray):
        err = f"data must be of type numpy.ndarray, but found {type(data)}"
        log_module.error(err)
        raise ValueError(err)

    file_path = set_save_path(path_to_file, file_name, suffix=".npy")
    np.save(file_path, data)


def numpy_load(path_to_file: str | plib.Path) -> np.ndarray:
    path_to_file = set_load_path(path_to_file, suffix=".npy")
    return np.load(path_to_file)


def torch_save(
        data: torch.Tensor | np.ndarray, path_to_file: str | plib.Path, file_name: str):
    if not torch.is_tensor(data):
        data = torch.from_numpy(data.copy())
    file_path = set_save_path(path_to_file, file_name, suffix=".pt")
    torch.save(data, file_path.as_posix())


def torch_load(path_to_file: str | plib.Path) -> torch.Tensor:
    path_to_file = set_load_path(path_to_file, suffix=".pt")
    return torch.load(path_to_file, map_location="cpu")


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
    path_to_file = set_load_path(path_to_file, suffix=".nii")
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
    file_path = set_save_path(path_to_dir, file_name, suffix=".nii")

    if torch.is_tensor(data):
        data = data.cpu().numpy()
    if data.dtype == bool:
        data = data.astype(np.int32)
    if torch.is_tensor(img_aff):
        img_aff = img_aff.cpu().numpy()
    if isinstance(img_aff, np.ndarray):
        img_aff = nib.Nifti1Image(data, affine=img_aff)
    else:
        img_aff = nib.Nifti1Image(data, affine=img_aff.affine)

    nib.save(img_aff, file_path.as_posix())


class HidePrints:
    """
    We use the logging module to track all logs.
    Some packages (like twixtools) are written with print statements.
    We try to fix and pull request to get them integrated.
    Meanwhile this helper prevents unnecessary print() calls to spam our log
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
