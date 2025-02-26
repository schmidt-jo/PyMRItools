import numpy as np
import logging
import pathlib as plib

import torch

from pymritools.utils import fft, torch_load, torch_save
from pymritools.config import setup_program_logging, setup_parser
from pymritools.config.seqprog.rawdata import RMOS

log_module = logging.getLogger(__name__)


def remove_oversampling(data: np.ndarray | torch.Tensor, data_in_k_space: bool = True,
                        read_dir: int = 0, os_factor: int = 2):
    if np.abs(os_factor - 1) < 1e-9:
        return data
    log_module.debug(f"remove oversampling")
    # get read direction dimension
    nx = data.shape[read_dir]

    if data_in_k_space:
        # need to transform
        data = fft(input_data=data, img_to_k=False, axes=(read_dir,))

    # data in freq domain, do removal
    lower_idx = int((os_factor - 1) / (2 * os_factor) * nx)
    upper_idx = int((os_factor + 1) / (2 * os_factor) * nx)
    if not torch.is_tensor(data):
        move_func = np.moveaxis
    else:
        move_func = torch.movedim
    data = move_func(data, read_dir, 0)[lower_idx:upper_idx]
    data = move_func(data, 0, read_dir)
    if data_in_k_space:
        # data was in k domain originally, hence we move back
        data = fft(input_data=data, img_to_k=True, axes=(read_dir,))
    return data


def rmos(settings: RMOS):
    path_in = plib.Path(settings.input_file)
    data = torch_load(path_in)
    if settings.dim < 1:
        log_module.info(f"No read direction given, deducing from shape, taking biggest dim.")
        ds = torch.tensor(data.shape)
        read_dir = torch.where(ds == ds.max())[0].item()
    else:
        read_dir = settings.dim
    log_module.info(f"found dims: {data.shape}, set read_dir = {read_dir}")
    data = remove_oversampling(
        data=data, data_in_k_space=settings.data_in_kspace, read_dir=read_dir, os_factor=settings.os_factor
    )
    torch_save(data, path_to_file=settings.out_path, file_name=f"{path_in.stem}_rmos")


def main():
    # setup logging
    setup_program_logging(name="remove oversampling", level=logging.INFO)
    # setup parser
    parser, args = setup_parser(
        prog_name="Remove_oversampling",
        dict_config_dataclasses={"settings": RMOS}
    )
    # get config
    settings = RMOS.from_cli(args=args.settings, parser=parser)
    settings.display()

    try:
        rmos(settings=settings)
    except Exception as e:
        parser.print_help()
        logging.exception(e)
        exit(-1)
