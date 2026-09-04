from pymritools.config.modeling import MEXPSettings
from pymritools.utils import nifti_load, nifti_save
from pymritools.config import setup_program_logging ,setup_parser
import logging
import pathlib as plib
import torch

log_module = logging.getLogger(__name__)


def process(settings: MEXPSettings):
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

    s0, r = fit(input_data, torch.tensor(settings.echo_times))

    # save
    nifti_save(data=s0, img_aff=input_img, path_to_dir=path_out, file_name="S0")
    nifti_save(data=r, img_aff=input_img, path_to_dir=path_out, file_name="R")


def fit(data: torch.Tensor, te: torch.Tensor):
    """
    Fits a mono-exponential decay model to the input data.

    This function performs a least squares fitting of a mono-exponential decay model to
    input data, where the signal decay is modeled as S(t) = S0 * exp(-R * t). The time
    dimension is assumed to be the last dimension of the input tensor, and the function
    batches all non-time dimensions for processing. It estimates the parameters S0 and
    R for each pixel or point in the input dataset.

    Arguments:
        data (torch.Tensor): Input tensor containing the signal data. The last dimension
            is assumed to represent the time axis.
        te (torch.Tensor): Tensor containing echo times [s] corresponding to the time points
            in the last dimension of the input data.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - A tensor of the same shape as the input (excluding the last dimension),
              representing the estimated S0 values.
            - A tensor of the same shape as the input (excluding the last dimension),
              representing the estimated R values.

    Raises:
        AttributeError: If the input tensor does not have at least 2 dimensions or if the
            number of echo times does not match the number of time points in the input data.
    """
    # we assume time dimension is the last dimension and batch the rest
    # use device of input tensor -> thus GPU computation is set via inputing a CUDA tensor
    device = data.device

    # save input shape
    shape = data.shape
    if shape.__len__() < 2:
        err = (f"module set up  to deal with at least 2 dimensions (x and t, time assumed in last dim), "
               f"but found shape: {shape}")
        log_module.error(err)
        raise AttributeError(err)
    if shape[-1] != te.shape[0]:
        err = (f"number of echo times ({te.shape[0]}) does not match number of time points "
               f"in input data ({shape[-1]})")
        log_module.error(err)
        raise AttributeError(err)

    # assume mono - exponential decay with S(t) = S0 exp(-R t)
    # want to least square fit for R (and S0) -> solve Y = Ab, with Y = log(S), b = (log(S0), -R)
    # assume time in last dim, reshape space to batch dimension
    input_data = torch.reshape(data, (-1, shape[-1]))
    n, t = input_data.shape

    if te.min() > 1:
        msg = f"Assuming echo times given ({te.tolist()}) to be in s. Try to adopt to seconds."
        log_module.warning(msg)
        te *= 1e-3

    te = te.to(dtype=input_data.dtype, device=device)

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

    return s_0, r


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
        process(settings=settings)
    except Exception as e:
        logging.exception(e)
        parser.print_help()


if __name__ == '__main__':
    main()

