from pymritools.config.modeling import MEXPSettings
from pymritools.utils import nifti_load, nifti_save
from pymritools.config import setup_program_logging ,setup_parser
import logging
import pathlib as plib
import torch


log_module = logging.getLogger(__name__)


def fit(settings: MEXPSettings):
    # set output path
    log_module.info(f"set output path: {settings.out_path}")
    path_out = plib.Path(settings.out_path).absolute()
    if not path_out.exists():
        log_module.info(f"mkdir {path_out.as_posix()}")
        path_out.mkdir(exist_ok=True, parents=True)

    # set device
    if settings.use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{settings.gpu_device}")
    else:
        device = torch.device("cpu")
    log_module.info(f"setting torch device: {device}")

    # load data
    if not settings.input_data:
        err = "no input file given"
        log_module.error(err)
        raise ValueError(err)
    input_data, input_img = nifti_load(settings.input_data)
    input_data = torch.from_numpy(input_data).to(device).to(dtype=torch.float32)

    # save input shape
    shape = input_data.shape

    # setup tensors, [xyz, t]
    if shape.__len__() < 2:
        err = (f"module set up  to deal with at least 2 dimensions (x and t, time assumed in last dim), "
               f"but found shape: {shape}")
        log_module.error(err)
        raise AttributeError(err)

    # assume mono - exponential decay with S(t) = S0 exp(-R t)
    # want to least square fit for R (and S0) -> solve Y = Ab, with Y = log(S), b = (log(S0), -R)
    # assume time in last dim, reshape space to batch dimension
    input_data = torch.reshape(input_data, (-1, shape[-1]))
    n, t = input_data.shape

    # get echo times
    te = torch.tensor(settings.echo_times).to(dtype=input_data.dtype, device=device)
    if te.shape[0] != t:
        err = (f"number of echo times ({te.shape[0]}) does not match number of time points "
               f"in input data ({t})")
        log_module.error(err)
        raise AttributeError(err)
    if te.min() > 1:
        msg = f"Assuming echo times given ({te.tolist()}) to be in ms. Try to adopt to seconds."
        log_module.warning(msg)
        te *= 1e-3
    # build tensors
    y = torch.nan_to_num(
        torch.log(input_data),
        nan=0.0, posinf=0.0, neginf=0.0
    )

    a = torch.ones((n, t, 2), dtype=input_data.dtype, device=device)
    a[:, :, 1] = te[None, :]

    # solve, according to torch documentation using lstsq is faster and more stable than matrix multiplications
    # dims: a [n, t, 2], b [n, 2, 1], y [n, t, 1]
    b = torch.linalg.lstsq(a, y[:, :, None]).solution

    # b, dims [n, 2, 1]
    s_0 = torch.squeeze(torch.exp(b[:, 0]))
    r = - torch.squeeze(b[:, 1])

    s_0 = torch.reshape(s_0, shape[:-1])
    r = torch.reshape(r, shape[:-1])

    # save
    nifti_save(data=s_0, img_aff=input_img, path_to_dir=path_out, file_name="S0")
    nifti_save(data=r, img_aff=input_img, path_to_dir=path_out, file_name="R")


def main():
    setup_program_logging(name="Exponential Decay Fitting", level=logging.INFO)

    # setup parser
    parser, prog_args = setup_parser(
        prog_name="Exponential Decay Fitting",
        dict_config_dataclasses={"settings": MEXPSettings}
    )

    # get settings
    settings = MEXPSettings.from_cli(args=prog_args.settings, parser=parser)
    settings.display()

    try:
        fit(settings=settings)
    except Exception as e:
        logging.exception(e)
        parser.print_help()


if __name__ == '__main__':
    main()

