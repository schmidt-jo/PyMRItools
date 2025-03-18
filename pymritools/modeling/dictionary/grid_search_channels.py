import logging

import tqdm
import numpy as np
import torch

from pymritools.modeling.dictionary.setup import setup_db, setup_b1, setup_path, setup_input, setup_b0
from pymritools.config import setup_program_logging, setup_parser
from pymritools.config.emc import EmcFitSettings
from pymritools.utils import nifti_save, fft

log_module = logging.getLogger(__name__)


def fit(settings: EmcFitSettings):
    # setup path
    path = setup_path(settings=settings)
    # load in and setup data
    input_data, input_img, data_shape, mask_nii_nonzero, rho_s, name = setup_input(settings=settings)
    # load in B1 if given
    b1_data, name = setup_b1(settings=settings)
    # load in B0 if given
    b0_data, name = setup_b0(settings=settings)
    # load in database
    db, db_torch, rho_db, t1t2b1b0_vals, t1_vals, t2_vals, b1_vals, b0_vals = setup_db(
        settings=settings, return_complex=torch.is_complex(input_data)
    )

    if settings.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{settings.gpu_device}')
    else:
        device = torch.device('cpu')
    log_module.info(f"Setting device: {device}")
    db_torch = db_torch.to(device=device, dtype=input_data.dtype)
    t1t2b1b0_vals = t1t2b1b0_vals.to(device)

    # allocate
    result_r2 = torch.zeros((input_data.shape[0],), dtype=t2_vals.dtype, device="cpu")
    result_r1 = torch.zeros((input_data.shape[0],), dtype=t1_vals.dtype, device="cpu")
    result_b1 = torch.zeros((input_data.shape[0],), dtype=b1_vals.dtype, device="cpu")
    result_b0 = torch.zeros((input_data.shape[0],), dtype=b0_vals.dtype, device="cpu")
    result_optimize = torch.zeros((input_data.shape[0]), dtype=torch.float32, device="cpu")

    # for now brute force all values
    batch_size = settings.batch_size
    num_batches = int(np.ceil(input_data.shape[0] / batch_size))
    # do slice wise processing
    for idx_b in tqdm.trange(num_batches, desc="Processing"):
        # probably need to batch the database to fit memory
        start = idx_b * batch_size
        end = min((idx_b + 1) * batch_size, input_data.shape[0])
        bs = end - start
        data_batch = input_data[start:end].to(device)
        # data and db are normalized, both might be complex
        # calculate dot, dot product is not sensitive to phase offsets,
        # which is what we need for channel wise processing.
        # Since channels approximately all have their respective offset
        dot = torch.linalg.vecdot(data_batch[:, None], db_torch[None, :], dim=-1)
        # need absolute value - phase offset dependency otherwise hides in imag channel
        dot = torch.abs(dot)
        # find maximum along db dimension
        max_vals, indices = torch.max(dot, dim=1)
        batch_t1t2b1b0 = t1t2b1b0_vals[indices].cpu()
        result_optimize[start:end] = max_vals
        result_r1[start:end] = 1 / batch_t1t2b1b0[:, 0]
        result_r2[start:end] = 1 / batch_t1t2b1b0[:, 1]
        result_b1[start:end] = batch_t1t2b1b0[:, 2]
        result_b0[start:end] = batch_t1t2b1b0[:, 3]

    # reshape & save
    names = ["optimize_residual", "r1", "r2", "b1", "b0"]
    for i, r in enumerate([result_optimize, result_r1, result_r2, result_b1, result_b0]):
        r = torch.reshape(r, data_shape[:-1])
        nifti_save(r, img_aff=input_img, path_to_dir=path, file_name=names[i])


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
    main()

